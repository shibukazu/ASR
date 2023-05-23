import copy
from logging import config, getLogger

import hydra
import mlflow
import torch
from conf import logging_conf
from ctc_model import (
    CausalConformerMultitaskCTCAdapterModel,
    CausalConformerMultitaskCTCLLAdapterModel,
    CausalConformerMultitaskCTCModel,
)
from data import CSJVADAdaptationDataset
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
        bhyp_token_ids = model.streaming_greedy_inference(
            bx=bx, bx_len=bx_len, num_previous_frames=cfg.model.num_previous_frames
        )
        bans_token_ids = [by[i, : by_len[i]].tolist() for i in range(by.shape[0])]
        bhyp_text = tokenizer.batch_token_ids_to_text(bhyp_token_ids)
        bans_text = tokenizer.batch_token_ids_to_text(bans_token_ids)

        cer = char_error_rate(bhyp_text, bans_text) * bx.shape[0]

        for hyp_text, ans_text in zip(bhyp_text, bans_text):
            print(f"hyp: {hyp_text}")
            print(f"ans: {ans_text}")

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
            adaptation_dataset = CSJVADAdaptationDataset(
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
        elif cfg.model.name == "CausalConformerMultitaskCTCLLAdapterModel":
            base_model_path = cfg.model.base_model_path
            with open(base_model_path, "rb") as f:
                base_model_cpt = torch.load(f)
            base_model_state = base_model_cpt["model"]
            base_model_args = base_model_cpt["model_args"]
            base_model = CausalConformerMultitaskCTCModel(
                **base_model_args,
            )
            base_model.load_state_dict(base_model_state)

            adapter_pretrained_model_path = cfg.model.model_path
            with open(adapter_pretrained_model_path, "rb") as f:
                adapter_pretrained_model_cpt = torch.load(f)
            adapter_state = adapter_pretrained_model_cpt["adapter"]
            adapter_pretrained_model_args = adapter_pretrained_model_cpt["model_args"]
            adapter_pretrained_model = CausalConformerMultitaskCTCLLAdapterModel(
                **adapter_pretrained_model_args,
            )

            adapter_pretrained_model.base_model_injection(base_model)
            adapter_pretrained_model.adapter.load_state_dict(adapter_state)
        else:
            raise NotImplementedError
        adapter_pretrained_model = DataParallel(adapter_pretrained_model)

        NUM_ADAPTATION = cfg.train.num_adaptation

        total_prev_cer = 0.0
        total_after_cers = [0.0 for _ in range(NUM_ADAPTATION)]
        total_improvements = [0.0 for _ in range(NUM_ADAPTATION)]
        count = 0
        for sample_idx, (_, x, y, x_len, y_len, audio_sec, vad, vad_len) in enumerate(adaptation_dataset):
            count += 1
            bar = tqdm(total=len(x))
            bar.set_description(f"{sample_idx} / {len(adaptation_dataset)} th Sample: ")
            torch.cuda.empty_cache()

            # data間での独立性を担保する
            model = copy.deepcopy(adapter_pretrained_model)
            model.to(DEVICE)

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
            elif cfg.model.name == "CausalConformerMultitaskCTCLLAdapterModel":
                adapter_optimizer = torch.optim.Adam(
                    model.adapter.parameters(),
                    lr=cfg.train.optimize.lr,
                    weight_decay=cfg.train.optimize.weight_decay,
                    betas=(cfg.train.optimize.beta1, cfg.train.optimize.beta2),
                    eps=cfg.train.optimize.eps,
                )
                optimizers.append(adapter_optimizer)
            else:
                raise NotImplementedError

            # 適応前のCERを計算する
            prev_cer = eval(cfg, model, tokenizer, x, x_len, y, y_len)
            total_prev_cer += prev_cer
            logger.info(f"prev_cer: {prev_cer}")

            for adaptation_idx in range(NUM_ADAPTATION):
                # adaptationする
                model.train()

                buffer_size = cfg.train.buffer_size
                accum_frame = 0
                for frame_idx in range(0, len(x), buffer_size):
                    accum_frame += buffer_size
                    _x = x[frame_idx : frame_idx + buffer_size]
                    _vad = vad[frame_idx : frame_idx + buffer_size]
                    _x_len = len(_x)
                    if _x_len < buffer_size:
                        continue  # avoid error because of the last frame
                    _subsampled_vad = vad_subsample(
                        vad_subsample(_vad, cfg.model.subsampling_kernel_size1, cfg.model.subsampling_stride1),
                        cfg.model.subsampling_kernel_size2,
                        cfg.model.subsampling_stride2,
                    )
                    _subsampled_vad = torch.tensor(_subsampled_vad, dtype=torch.float32)
                    _subsampled_vad_len = len(_subsampled_vad)

                    bce_criterion = torch.nn.BCELoss(reduction="sum")
                    loss = forward(
                        cfg=cfg,
                        model=model,
                        x=_x,
                        x_len=_x_len,
                        subsampled_vad=_subsampled_vad,
                        subsampled_vad_len=_subsampled_vad_len,
                        bce_criterion=bce_criterion,
                    )
                    loss.backward()
                    bar.update(_x_len)

                    if accum_frame > cfg.train.accum_frame:
                        for optimizer in optimizers:
                            optimizer.step()
                            optimizer.zero_grad()
                        accum_frame = 0

                # 適応後のCERを計算する
                after_cer = eval(cfg, model, tokenizer, x, x_len, y, y_len)
                logger.info(f"after_cer: {after_cer}")
                total_after_cers[adaptation_idx] += after_cer

                # improvementを計算する
                improvement = prev_cer - after_cer
                logger.info(f"improvement: {improvement}")
                total_improvements[adaptation_idx] += improvement

                bar.set_postfix(
                    {"avg_impr": total_improvements[adaptation_idx].item() / count if count > 0 else 0.0}
                )

            mlflow.log_metric(
                "avg_prev_cer",
                total_prev_cer.item() / count if count > 0 else 0.0,
                step=sample_idx,
            )
            for adaptation_idx in range(NUM_ADAPTATION):
                mlflow.log_metric(
                    f"{adaptation_idx}th avg_improvement",
                    total_improvements[adaptation_idx].item() / count if count > 0 else 0.0,
                    step=sample_idx,
                )

                mlflow.log_metric(
                    f"{adaptation_idx}th avg_after_cer",
                    total_after_cers[adaptation_idx].item() / count if count > 0 else 0.0,
                    step=sample_idx,
                )

            del model


if __name__ == "__main__":
    main()
