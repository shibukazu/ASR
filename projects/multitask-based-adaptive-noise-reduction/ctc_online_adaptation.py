import copy
from logging import config, getLogger

import hydra
import mlflow
import numpy as np
import torch
from conf import logging_conf
from ctc_model import CausalConformerMultitaskCTCAdapterModel
from data import CSJVAD1PathAdaptationDataset
from hydra.core.hydra_config import HydraConfig
from omegaconf import DictConfig
from rich.logging import RichHandler
from tokenizer import SentencePieceTokenizer
from torchmetrics.functional import char_error_rate

# from torchmetrics.functional import word_error_rate
from tqdm import tqdm
from util.mlflow import log_params_from_omegaconf_dict

from utils import calc_slope


class DataParallel(torch.nn.DataParallel):
    def __getattr__(self, name):
        try:
            return super().__getattr__(name)
        except AttributeError:
            return getattr(self.module, name)


DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def vad_subsample(vad, kernel_size, stride):
    n_subsample = (len(vad) - kernel_size + stride) // stride
    subsampled_vad = []
    for i in range(n_subsample):
        sub = vad[i * stride : i * stride + kernel_size]
        if len(sub) // 2 + 1 <= sum(sub):
            subsampled_vad.append(1)
        else:
            subsampled_vad.append(0)
    return subsampled_vad


def forward(
    cfg: DictConfig,
    model,
    x,
    x_len,
    subsampled_vad,
    subsampled_vad_len,
    bce_criterion,
):
    bx = x.unsqueeze(0).to(DEVICE)
    bx_len = torch.tensor([x_len]).to(DEVICE)
    bsubsampled_vad = subsampled_vad.unsqueeze(0).to(DEVICE)

    _, _, bsubsampled_vad_probs = model(
        bx=bx,
        bx_len=bx_len,
    )  # bvad_probs: [B, T]
    vad_loss = bce_criterion(bsubsampled_vad_probs.squeeze(-1), bsubsampled_vad)
    vad_loss = vad_loss / bx.shape[0]

    loss = vad_loss

    return loss


def eval(
    cfg: DictConfig,
    model,
    tokenizer,
    x,
    x_len,
    y,
    y_len,
):
    model.eval()

    bx = x.unsqueeze(0).to(DEVICE)
    bx_len = torch.tensor([x_len]).to(DEVICE)
    by = y.unsqueeze(0).to(DEVICE)
    by_len = torch.tensor([y_len]).to(DEVICE)

    with torch.no_grad():
        bhyp_token_ids = model.greedy_inference(bx=bx, bx_len=bx_len)
        bans_token_ids = [by[i, : by_len[i]].tolist() for i in range(by.shape[0])]
        bhyp_text = tokenizer.batch_token_ids_to_text(bhyp_token_ids)
        bans_text = tokenizer.batch_token_ids_to_text(bans_token_ids)

        cer = char_error_rate(bhyp_text, bans_text) * bx.shape[0]

        # for hyp_text, ans_text in zip(bhyp_text, bans_text):
        #    print(f"hyp: {hyp_text}")
        #    print(f"ans: {ans_text}")

        return cer


@hydra.main(version_base=None, config_path="conf/ctc", config_name=None)
def main(cfg: DictConfig):
    CONF_NAME = HydraConfig.get().job.config_name
    EXPERIMENT_NAME = CONF_NAME
    ARTIFACT_LOCATION = f"./artifacts/{EXPERIMENT_NAME}"
    experiment = mlflow.get_experiment_by_name(EXPERIMENT_NAME)
    if experiment is None:
        experiment_id = mlflow.create_experiment(name=EXPERIMENT_NAME, artifact_location=ARTIFACT_LOCATION)
    else:
        experiment_id = experiment.experiment_id

    torch.backends.cudnn.benchmark = False
    with mlflow.start_run(experiment_id=experiment_id) as mlflow_run:
        LOG_DIR = mlflow_run.info.artifact_uri
        config.dictConfig(logging_conf.config_generator(LOG_DIR))
        logger = getLogger()
        logger.handlers[0] = RichHandler(markup=True)  # pretty formatting
        # save parameters from hydra to mlflow
        log_params_from_omegaconf_dict(cfg)

        tokenizer = SentencePieceTokenizer(
            model_file_path=cfg.tokenizer.model_file_path,
        )

        if cfg.dataset.name == "CSJ":
            adaptation_dataset = CSJVAD1PathAdaptationDataset(
                json_file_path=cfg.dataset.train.json_file_path,
                resampling_rate=16000,
                tokenizer=tokenizer,
                spec_aug=None,
            )
        else:
            raise NotImplementedError

        # torch.autograd.set_detect_anomaly(True)
        if cfg.model.name == "CausalConformerMultitaskCTCAdapterModel":
            adapter_pretrained_model_path = cfg.model.pretrained_model_path
            with open(adapter_pretrained_model_path, "rb") as f:
                cpt = torch.load(f)
            adapter_pretrained_model_state = cpt["model"]
            adapter_pretrained_model_args = cpt["model_args"]
            adapter_pretrained_model = CausalConformerMultitaskCTCAdapterModel(
                **adapter_pretrained_model_args,
            )
            adapter_pretrained_model.load_state_dict(adapter_pretrained_model_state)
        else:
            raise NotImplementedError
        adapter_pretrained_model = DataParallel(adapter_pretrained_model)

        MAX_NUM_ADAPTATION = cfg.train.num_adaptation

        # サンプル内のCER, Improve平均値およびSlopeを格納する配列を用意する
        # 以下の配列の各和をnum_sampleで割ったものをサンプル間平均値とする
        total_inner_sample_avg_baseline_cer = 0.0

        total_inner_sample_avg_curr_after_cers = [0.0 for _ in range(MAX_NUM_ADAPTATION)]
        total_inner_sample_avg_next_after_cers = [0.0 for _ in range(MAX_NUM_ADAPTATION)]
        total_inner_sample_avg_curr_improves = [0.0 for _ in range(MAX_NUM_ADAPTATION)]
        total_inner_sample_avg_next_improves = [0.0 for _ in range(MAX_NUM_ADAPTATION)]
        total_inner_sample_curr_slopes = [0.0 for _ in range(MAX_NUM_ADAPTATION)]
        total_inner_sample_next_slopes = [0.0 for _ in range(MAX_NUM_ADAPTATION)]

        sample_count = 0  # サンプル間平均を計算するために利用される

        # 各長時間音声（ノイズ, 話者の組）ごとに調べる
        for sample_idx, (
            _,
            xs,
            ys,
            x_lens,
            y_lens,
            _,
            vads,
            _,
            _,
            _,
        ) in enumerate(adaptation_dataset):
            sample_count += 1
            NUM_UTTERANCES = len(xs)
            if NUM_UTTERANCES <= 1:
                continue
            bar = tqdm(NUM_UTTERANCES * MAX_NUM_ADAPTATION)
            bar.set_description(f"{sample_idx} / {len(adaptation_dataset)} th Sample: ")

            # Baseline CERを計算する
            torch.cuda.empty_cache()
            model = copy.deepcopy(adapter_pretrained_model)
            model.to(DEVICE)
            baseline_cers = [0 for _ in range(NUM_UTTERANCES)]
            for utterance_idx in range(NUM_UTTERANCES):
                baseline_cer = eval(
                    cfg,
                    model,
                    tokenizer,
                    xs[utterance_idx],
                    x_lens[utterance_idx],
                    ys[utterance_idx],
                    y_lens[utterance_idx],
                )
                baseline_cers[utterance_idx] = baseline_cer
            total_inner_sample_avg_baseline_cer += sum(baseline_cers) / len(baseline_cers)
            mlflow.log_metric(
                "baseline_cer",
                total_inner_sample_avg_baseline_cer / sample_count if sample_count > 0 else 0.0,
                step=sample_idx,
            )

            # 各Adaptationの回数ごとに調べる
            for num_adaptation in range(MAX_NUM_ADAPTATION):
                # dataおよびAdaptation回数の間での独立性を担保する
                torch.cuda.empty_cache()
                model = copy.deepcopy(adapter_pretrained_model)
                model.to(DEVICE)
                bce_criterion = torch.nn.BCELoss(reduction="sum")

                # adapter専用のoptimizerを用意する
                optimizers = []
                if cfg.model.name == "CausalConformerMultitaskCTCAdapterModel":
                    for block_idx in range(len(model.encoder.adapter_blocks)):
                        optimizer = torch.optim.Adam(
                            model.encoder.adapter_blocks[block_idx].adapter.parameters(),
                            lr=cfg.train.optimize.lr,
                            weight_decay=cfg.train.optimize.weight_decay,
                            betas=(cfg.train.optimize.beta1, cfg.train.optimize.beta2),
                            eps=cfg.train.optimize.eps,
                        )
                        optimizers.append(optimizer)
                else:
                    raise NotImplementedError

                # サンプル内平均値を計算するための配列および変数
                curr_after_cers = [0.0 for _ in range(NUM_UTTERANCES)]
                next_after_cers = [0.0 for _ in range(NUM_UTTERANCES)]
                curr_improves = [0.0 for _ in range(NUM_UTTERANCES)]
                next_improves = [0.0 for _ in range(NUM_UTTERANCES)]

                # 1発話ずつAdaptationする
                for curr_utterance_idx in range(0, NUM_UTTERANCES - 1):
                    next_utterance_idx = curr_utterance_idx + 1
                    next_x = xs[next_utterance_idx]
                    next_x_len = x_lens[next_utterance_idx]
                    next_y = ys[next_utterance_idx]
                    next_y_len = y_lens[next_utterance_idx]

                    curr_x = xs[curr_utterance_idx]
                    curr_x_len = x_lens[curr_utterance_idx]
                    curr_y = ys[curr_utterance_idx]
                    curr_y_len = y_lens[curr_utterance_idx]
                    curr_vad = vads[curr_utterance_idx]

                    # FrameレベルでのAdaptationをNUM_ADAPTATION回行う
                    model.train()
                    for _ in range(num_adaptation):
                        buffer_size = cfg.train.buffer_size
                        for frame_idx in range(0, len(curr_x), buffer_size):
                            if frame_idx + buffer_size > len(curr_x):
                                continue
                            _curr_x = curr_x[frame_idx : frame_idx + buffer_size]
                            _curr_x_len = len(_curr_x)
                            _curr_vad = curr_vad[frame_idx : frame_idx + buffer_size]
                            _curr_subsampled_vad = vad_subsample(
                                vad_subsample(
                                    _curr_vad, cfg.model.subsampling_kernel_size1, cfg.model.subsampling_stride1
                                ),
                                cfg.model.subsampling_kernel_size2,
                                cfg.model.subsampling_stride2,
                            )
                            _curr_subsampled_vad = torch.tensor(_curr_subsampled_vad, dtype=torch.float32)
                            _curr_subsampled_vad_len = len(_curr_subsampled_vad)

                            _curr_loss = forward(
                                cfg=cfg,
                                model=model,
                                x=_curr_x,
                                x_len=_curr_x_len,
                                subsampled_vad=_curr_subsampled_vad,
                                subsampled_vad_len=_curr_subsampled_vad_len,
                                bce_criterion=bce_criterion,
                            )
                            _curr_loss.backward()
                            for optimizer in optimizers:
                                optimizer.step()
                                optimizer.zero_grad()

                    # 現在の発話でのAFTER CERを計算する
                    curr_after_cer = eval(cfg, model, tokenizer, curr_x, curr_x_len, curr_y, curr_y_len).item()
                    curr_after_cers[curr_utterance_idx] = curr_after_cer
                    curr_improves[curr_utterance_idx] = baseline_cers[curr_utterance_idx] - curr_after_cer

                    # 1つ先の発話でのAFTER CERを計算する
                    next_after_cer = eval(cfg, model, tokenizer, next_x, next_x_len, next_y, next_y_len).item()
                    next_after_cers[next_utterance_idx] = next_after_cer
                    next_improves[next_utterance_idx] = baseline_cers[next_utterance_idx] - next_after_cer

                    bar.update(1)

                # サンプル内の平均CERを求める
                total_inner_sample_avg_curr_after_cers[num_adaptation] += sum(curr_after_cers[:-1]) / (
                    NUM_UTTERANCES - 1
                )
                total_inner_sample_avg_next_after_cers[num_adaptation] += sum(next_after_cers[1:]) / (
                    NUM_UTTERANCES - 1
                )
                total_inner_sample_avg_curr_improves[num_adaptation] += sum(curr_improves[:-1]) / (NUM_UTTERANCES - 1)
                total_inner_sample_avg_next_improves[num_adaptation] += sum(next_improves[1:]) / (NUM_UTTERANCES - 1)

                # サンプル内のslopeを求める
                x = np.arange(NUM_UTTERANCES - 1)
                curr_slope = calc_slope(x, curr_improves[:-1])
                next_slope = calc_slope(x, next_improves[1:])
                total_inner_sample_curr_slopes[num_adaptation] += curr_slope
                total_inner_sample_next_slopes[num_adaptation] += next_slope

            for num_adaptation in range(MAX_NUM_ADAPTATION):
                mlflow.log_metric(
                    f"{num_adaptation}_avg_curr_slope",
                    total_inner_sample_curr_slopes[num_adaptation] / sample_count if sample_count > 0 else 0.0,
                    step=sample_idx,
                )
                mlflow.log_metric(
                    f"{num_adaptation}_avg_next_slope",
                    total_inner_sample_next_slopes[num_adaptation] / sample_count if sample_count > 0 else 0.0,
                    step=sample_idx,
                )
                mlflow.log_metric(
                    f"{num_adaptation}_avg_curr_after_cer",
                    total_inner_sample_avg_curr_after_cers[num_adaptation] / sample_count if sample_count > 0 else 0.0,
                    step=sample_idx,
                )
                mlflow.log_metric(
                    f"{num_adaptation}_avg_next_after_cer",
                    total_inner_sample_avg_next_after_cers[num_adaptation] / sample_count if sample_count > 0 else 0.0,
                    step=sample_idx,
                )
                mlflow.log_metric(
                    f"{num_adaptation}_avg_curr_improve",
                    total_inner_sample_avg_curr_improves[num_adaptation] / sample_count if sample_count > 0 else 0.0,
                    step=sample_idx,
                )
                mlflow.log_metric(
                    f"{num_adaptation}_avg_next_improve",
                    total_inner_sample_avg_next_improves[num_adaptation] / sample_count if sample_count > 0 else 0.0,
                    step=sample_idx,
                )

            del model


if __name__ == "__main__":
    main()
