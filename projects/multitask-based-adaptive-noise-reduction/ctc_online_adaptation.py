import copy
from logging import config, getLogger

import hydra
import mlflow
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

        NUM_ADAPTATION = cfg.train.num_adaptation
        total_baseline_cer = 0.0

        total_curr_after_cers = [0.0 for _ in range(NUM_ADAPTATION)]
        total_next_after_cers = [0.0 for _ in range(NUM_ADAPTATION)]

        baseline_count = 0
        count = 0
        for sample_idx, (
            _,
            xs,
            ys,
            x_lens,
            y_lens,
            audio_sec,
            vads,
            vad_lens,
            subsampled_vads,
            subsampled_vad_lens,
        ) in enumerate(adaptation_dataset):
            bar = tqdm(total=len(xs))
            bar.set_description(f"{sample_idx} / {len(adaptation_dataset)} th Sample: ")
            torch.cuda.empty_cache()

            # data間での独立性を担保する
            model = copy.deepcopy(adapter_pretrained_model)
            model.to(DEVICE)
            bce_criterion = torch.nn.BCELoss(reduction="sum")

            # adapter専用のoptimizer, schedulerを用意する
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

            NUM_UTTERANCES = len(xs)

            # Adaptation前のCERを計算する
            for utterance_idx in range(NUM_UTTERANCES):
                baseline_count += 1

                baseline_cer = eval(
                    cfg,
                    model,
                    tokenizer,
                    xs[utterance_idx],
                    x_lens[utterance_idx],
                    ys[utterance_idx],
                    y_lens[utterance_idx],
                )
                total_baseline_cer += baseline_cer

            # 1発話ずつAdaptationを行う
            # 現在の発話でAdaptation -> 1つ先の発話のCERを計算 (改善率を計算する)
            num_utterances = len(xs)
            for curr_utterance_idx in range(0, num_utterances - 1):
                count += 1

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

                for adaptation_idx in range(NUM_ADAPTATION):
                    # [SUB] Adaptationを行う
                    # [SUB] FrameレベルでのAdaptationを行う
                    model.train()
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

                        curr_loss = forward(
                            cfg=cfg,
                            model=model,
                            x=_curr_x,
                            x_len=_curr_x_len,
                            subsampled_vad=_curr_subsampled_vad,
                            subsampled_vad_len=_curr_subsampled_vad_len,
                            bce_criterion=bce_criterion,
                        )
                        curr_loss.backward()
                        for optimizer in optimizers:
                            optimizer.step()
                            optimizer.zero_grad()

                    # [SUB] 比較のために現在の発話でのAFTER CERを計算する
                    curr_after_cer = eval(cfg, model, tokenizer, curr_x, curr_x_len, curr_y, curr_y_len)
                    total_curr_after_cers[adaptation_idx] += curr_after_cer

                    # 1つ先の発話でのAFTER CERを計算する
                    next_after_cer = eval(cfg, model, tokenizer, next_x, next_x_len, next_y, next_y_len)
                    total_next_after_cers[adaptation_idx] += next_after_cer

                bar.update(1)

            mlflow.log_metric(
                "baseline_cer",
                total_baseline_cer.item() / baseline_count if baseline_count > 0 else 0.0,
                step=sample_idx,
            )

            for adaptation_idx in range(NUM_ADAPTATION):
                mlflow.log_metric(
                    f"{adaptation_idx}th avg_curr_after_cer",
                    total_curr_after_cers[adaptation_idx].item() / count if count > 0 else 0.0,
                    step=sample_idx,
                )
                mlflow.log_metric(
                    f"{adaptation_idx}th avg_next_after_cer",
                    total_next_after_cers[adaptation_idx].item() / count if count > 0 else 0.0,
                    step=sample_idx,
                )

            del model


if __name__ == "__main__":
    main()
