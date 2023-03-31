import math
import os
from logging import config, getLogger

import hydra
import mlflow
import torch
from conf import logging_conf
from ctc_model import CausalConformerVADAdapterCTCModel
from data import CSJDataset, CSJVADPretrainDataset, get_dataloader, get_vad_pretrain_dataloader
from hydra.core.hydra_config import HydraConfig
from modules.spec_aug import SpecAug
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


def forward(
    cfg: DictConfig,
    model,
    bx,
    bx_len,
    by,
    by_len,
    bsubsampled_vad,
    bsubsampled_vad_len,
    ctc_criterion,
    bce_criterions,
):
    bx = bx.to(DEVICE)
    by = by.to(DEVICE)
    bsubsampled_vad = bsubsampled_vad.to(DEVICE)

    blog_probs, bsubsampled_x_len, bsubsampled_vad_probs = model(
        bx=bx,
        bx_len=bx_len,
    )  # bvad_probs: [B, num_adapter_blocks, T]
    bsubsampled_vad_probs = bsubsampled_vad_probs.transpose(0, 1)  # bvad_probs: [num_adapter_blocks, B, T]
    # print(blog_probs.shape, bsubsampled_x_len.shape, bvad_probs.shape, bvad.shape)
    ctc_loss = ctc_criterion(
        log_probs=blog_probs.transpose(0, 1),
        targets=by,
        input_lengths=bsubsampled_x_len.to(DEVICE),
        target_lengths=by_len.to(DEVICE),
    )
    ctc_loss = ctc_loss / bx.shape[0]

    vad_loss = 0
    # 各ブロックごとにVADのLossを計算
    for bce_criterion, bsubsampled_vad_prob in zip(bce_criterions, bsubsampled_vad_probs):  # bvad_prob: [B, T]
        raw_vad_loss = bce_criterion(bsubsampled_vad_prob, bsubsampled_vad)  # [B, T]
        # padding部分を0にする
        # bsubsampled_vad_lenの各長さより後ろの部分を0にする
        for i, subsampled_vad_len in enumerate(bsubsampled_vad_len):
            raw_vad_loss[i, subsampled_vad_len:] = 0
        vad_loss += raw_vad_loss.sum()

    vad_loss = vad_loss / bx.shape[0]

    loss = ctc_loss + vad_loss

    return loss


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
        vocab_size = tokenizer.num_tokens

        if cfg.dataset.name == "CSJ":
            spec_aug = SpecAug(
                freq_mask_max_length=cfg.model.spec_aug.freq_mask_max_length,
                time_mask_max_length=cfg.model.spec_aug.time_mask_max_length,
                num_freq_mask=cfg.model.spec_aug.num_freq_mask,
                num_time_mask=cfg.model.spec_aug.num_time_mask,
            )
            train_dataset = CSJVADPretrainDataset(
                json_file_path=cfg.dataset.train.json_file_path,
                resampling_rate=16000,
                tokenizer=tokenizer,
                spec_aug=spec_aug,
            )
            dev_dataset = CSJDataset(
                json_file_path=cfg.dataset.dev.json_file_path,
                resampling_rate=16000,
                tokenizer=tokenizer,
                spec_aug=None,
            )
        else:
            raise NotImplementedError

        train_dataloader = get_vad_pretrain_dataloader(
            train_dataset,
            batch_sec=cfg.train.batch_sec,
            batch_text_len=cfg.train.batch_text_len,
            num_workers=4,
            pin_memory=True,
            pad_idx=tokenizer.pad_token_id,
        )

        dev_dataloader = get_dataloader(
            dev_dataset,
            batch_sec=cfg.train.batch_sec,
            batch_text_len=cfg.train.batch_text_len,
            num_workers=4,
            pin_memory=True,
            pad_idx=tokenizer.pad_token_id,
        )

        # torch.autograd.set_detect_anomaly(True)
        model_args = {
            "input_size": cfg.model.encoder.input_size,
            "subsampled_input_size": cfg.model.encoder.subsampled_input_size,
            "num_conformer_blocks": cfg.model.encoder.num_conformer_blocks,
            "ff_hidden_size": cfg.model.encoder.ff_hidden_size,
            "conv_hidden_size": cfg.model.encoder.conv_hidden_size,
            "conv_kernel_size": cfg.model.encoder.conv_kernel_size,
            "mha_num_heads": cfg.model.encoder.mha_num_heads,
            "num_adapter_blocks": cfg.model.encoder.num_adapter_blocks,
            "adapter_hidden_size": cfg.model.encoder.adapter_hidden_size,
            "dropout": cfg.model.encoder.dropout,
            "subsampling_kernel_size1": cfg.model.encoder.subsampling_kernel_size1,
            "subsampling_stride1": cfg.model.encoder.subsampling_stride1,
            "subsampling_kernel_size2": cfg.model.encoder.subsampling_kernel_size2,
            "subsampling_stride2": cfg.model.encoder.subsampling_stride2,
            "num_previous_frames": cfg.model.encoder.num_previous_frames,
            "is_timewise_ln": cfg.model.encoder.is_timewise_ln,
            "vocab_size": vocab_size,
            "blank_idx": tokenizer.blank_token_id,
        }
        if cfg.model.name == "CausalConformerVADAdapterCTC":
            model = CausalConformerVADAdapterCTCModel(**model_args)
        else:
            raise NotImplementedError
        model = DataParallel(model).to(DEVICE)
        optimizer = torch.optim.Adam(
            model.parameters(),
            lr=cfg.train.optimize.lr,
            weight_decay=cfg.train.optimize.weight_decay,
            betas=(cfg.train.optimize.beta1, cfg.train.optimize.beta2),
            eps=cfg.train.optimize.eps,
        )
        if cfg.train.optimize.do_schedule:
            scheduler = torch.optim.lr_scheduler.LambdaLR(
                optimizer,
                lambda s: cfg.model.encoder.subsampled_input_size**-0.5
                * min((s + 1) ** -0.5, (s + 1) * cfg.train.optimize.warmup_steps**-1.5),
            )

        ctc_criterion = torch.nn.CTCLoss(blank=tokenizer.blank_token_id, reduction="sum")
        bce_criterions = []
        for i in range(cfg.model.encoder.num_adapter_blocks):
            # BCELossはpaddingに対応していないため、reduction=noneにした上で、後でpadding部分を除外する
            bce_criterions.append(torch.nn.BCELoss(reduction="none"))

        NUM_EPOCH = cfg.train.num_epoch
        num_steps = 0

        for i in range(1, NUM_EPOCH + 1):
            torch.cuda.empty_cache()

            model.train()
            epoch_train_loss = 0
            accum_sec = 0
            bar = tqdm(total=len(train_dataset))
            bar.set_description(f"Train Epoch {i}  ")
            for _, bx, by, bx_len, by_len, baudio_sec, bsubsampled_vad, bsubsampled_vad_len in train_dataloader:
                accum_sec += sum(baudio_sec)
                loss = forward(
                    cfg=cfg,
                    model=model,
                    bx=bx,
                    by=by,
                    bx_len=bx_len,
                    by_len=by_len,
                    bsubsampled_vad=bsubsampled_vad,
                    bsubsampled_vad_len=bsubsampled_vad_len,
                    ctc_criterion=ctc_criterion,
                    bce_criterions=bce_criterions,
                )
                loss.backward()
                epoch_train_loss += loss.item() * bx.shape[0]

                for param_group in optimizer.param_groups:
                    bar.set_postfix({"loss": loss.item(), "lr": param_group["lr"], "step": num_steps})
                bar.update(bx.shape[0])

                if accum_sec > cfg.train.accum_sec:
                    grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.train.optimize.max_grad_norm)
                    if math.isnan(grad_norm):
                        logger.error("grad norm is nan. Do not update model.")
                        logger.error(f"loss value: {loss.item()}")
                    else:
                        optimizer.step()
                        if cfg.train.optimize.do_schedule:
                            scheduler.step()
                            num_steps += 1
                        optimizer.zero_grad()
                        accum_sec = 0

            logger.info(f"Train loss: {epoch_train_loss / len(train_dataset)}")
            mlflow.log_metric("train_loss", epoch_train_loss / len(train_dataset), step=i)
            # 学習がうまくいっているか確認するためにdevデータで評価する
            # これはAdaptationとは別
            model.eval()
            epoch_dev_cer = 0

            bar = tqdm(total=len(dev_dataset))
            bar.set_description(f"Valid Epoch {i}  ")
            torch.cuda.empty_cache()
            with torch.no_grad():
                for _, bx, by, bx_len, by_len, baudio_sec in dev_dataloader:

                    if i >= 5 and i % 5 == 0 and cfg.do_decode:
                        if cfg.decoder.type == "streaming_greedy":
                            bx = bx.to(DEVICE)
                            bhyp_token_ids = model.streaming_greedy_inference(bx=bx, bx_len=bx_len)
                        elif cfg.decoder.type == "greedy":
                            bx = bx.to(DEVICE)
                            bhyp_token_ids = model.greedy_inference(bx=bx, bx_len=bx_len)
                        else:
                            raise NotImplementedError
                        bans_token_ids = [by[i, : by_len[i]].tolist() for i in range(by.shape[0])]
                        bhyp_text = tokenizer.batch_token_ids_to_text(bhyp_token_ids)
                        bans_text = tokenizer.batch_token_ids_to_text(bans_token_ids)

                        epoch_dev_cer += char_error_rate(bhyp_text, bans_text) * bx.shape[0]
                        # epoch_dev_wer += word_error_rate(bhyp_text, bans_text) * bx.shape[0]

                    bar.update(by.shape[0])

                # logger.info(f"Dev loss: {epoch_dev_loss / len(dev_dataset)}")
                # mlflow.log_metric("dev_loss", epoch_dev_loss / len(dev_dataset), step=i)

                if i >= 5 and i % 5 == 0 and cfg.do_decode:
                    mlflow.log_metric("dev_cer", epoch_dev_cer / len(dev_dataset), step=i)
                    logger.info(f"Dev CER: {epoch_dev_cer / len(dev_dataset)}")

            if i >= 5 and i % 5 == 0:
                torch.save(
                    {
                        "model": model.module.state_dict(),
                        "model_args": model_args,
                        "optimizer": optimizer.state_dict(),
                        "scheduler": scheduler.state_dict() if cfg.train.optimize.do_schedule else None,
                        "dev_cer": epoch_dev_cer / len(dev_dataset),
                    },
                    os.path.join(mlflow_run.info.artifact_uri, f"model_{i}.pth"),
                )


if __name__ == "__main__":
    main()

# sil sp_
