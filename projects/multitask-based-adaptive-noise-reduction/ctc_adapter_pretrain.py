import os
from logging import config, getLogger

import hydra
import mlflow
import torch
from conf import logging_conf
from ctc_model import CausalConformerMultitaskCTCAdapterModel
from data import CSJVADPretrainDataset, get_vad_pretrain_dataloader
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
    bsubsampled_vad,
    bsubsampled_vad_len,
    bce_criterion,
):
    bx = bx.to(DEVICE)
    bsubsampled_vad = bsubsampled_vad.to(DEVICE)

    _, _, bsubsampled_vad_probs = model(
        bx=bx,
        bx_len=bx_len,
    )  # bvad_probs: [B, T]
    raw_vad_loss = bce_criterion(bsubsampled_vad_probs.squeeze(-1), bsubsampled_vad)  # [B, T]
    for i, subsampled_vad_len in enumerate(bsubsampled_vad_len):
        raw_vad_loss[i, subsampled_vad_len:] = 0
    vad_loss = raw_vad_loss.sum() / bx.shape[0]

    loss = vad_loss

    return loss


def eval(
    cfg: DictConfig,
    model,
    tokenizer,
    eval_dataloader,
):
    model.eval()
    torch.cuda.empty_cache()
    cer = 0
    with torch.no_grad():
        for _, bx, by, bx_len, by_len, baudio_sec, bsubsampled_vad, bsubsampled_vad_len in eval_dataloader:
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

            cer += char_error_rate(bhyp_text, bans_text) * bx.shape[0]

    cer /= len(eval_dataloader.dataset)
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
            eval_dataset = CSJVADPretrainDataset(
                json_file_path=cfg.dataset.eval.json_file_path,
                resampling_rate=16000,
                tokenizer=tokenizer,
                spec_aug=None,
            )
            eval_ref_dataset = CSJVADPretrainDataset(
                json_file_path=cfg.dataset.eval_ref.json_file_path,
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

        eval_dataloader = get_vad_pretrain_dataloader(
            eval_dataset,
            batch_sec=cfg.train.batch_sec,
            batch_text_len=cfg.train.batch_text_len,
            num_workers=4,
            pin_memory=True,
            pad_idx=tokenizer.pad_token_id,
        )

        eval_ref_dataloader = get_vad_pretrain_dataloader(
            eval_ref_dataset,
            batch_sec=cfg.train.batch_sec,
            batch_text_len=cfg.train.batch_text_len,
            num_workers=4,
            pin_memory=True,
            pad_idx=tokenizer.pad_token_id,
        )

        # torch.autograd.set_detect_anomaly(True)
        if cfg.model.name == "CausalConformerMultitaskCTCAdapterModel":
            pretrained_model_path = cfg.model.pretrained_model_path
            with open(pretrained_model_path, "rb") as f:
                cpt = torch.load(f)
            pretrained_model_state = cpt["model"]
            pretrained_model_args = cpt["model_args"]
            adapter_pretrained_model_args = pretrained_model_args
            adapter_pretrained_model_args["adapter_hidden_size"] = cfg.model.adapter_hidden_size
            adapter_pretrained_model_args["num_adapter_blocks"] = cfg.model.num_adapter_blocks
            pretrained_model = CausalConformerMultitaskCTCAdapterModel(
                **adapter_pretrained_model_args,
            ).to(DEVICE)
            # モデルのロードを行う
            # 一部のblockをadapter付きブロックに置き換えているため、手動でロードする
            adapter_count = 0
            for name, param in tqdm(pretrained_model.named_parameters()):
                if name.startswith("encoder.adapter_blocks."):
                    if ".adapter." in name:
                        if cfg.model.adapter_init == "identity":
                            if "weight" in name:
                                adapter_count += 1
                                torch.nn.init.eye_(param)
                            elif "bias" in name:
                                torch.nn.init.zeros_(param)
                            else:
                                raise NotImplementedError
                        elif cfg.model.adapter_init == "random":
                            pass
                        else:
                            raise NotImplementedError
                    else:
                        idx = int(name.split(".")[2])
                        modified_idx = idx
                        modified_name = name.replace(f".{idx}.", f".{modified_idx}.")
                        modified_name = modified_name.replace("adapter_blocks", "conformer_blocks")
                        param.data = pretrained_model_state[modified_name]
                elif name.startswith("encoder.conformer_blocks."):
                    idx = int(name.split(".")[2])
                    modified_idx = cfg.model.num_adapter_blocks + idx
                    modified_name = name.replace(f".{idx}.", f".{modified_idx}.")
                    param.data = pretrained_model_state[modified_name]
                else:
                    param.data = pretrained_model_state[name]
            assert adapter_count == cfg.model.num_adapter_blocks * 2, f"{adapter_count}"
        else:
            raise NotImplementedError
        pretrained_model = DataParallel(pretrained_model).to(DEVICE)
        # adapter専用のoptimizer, schedulerを用意する
        optimizers = []
        for block_idx in range(len(pretrained_model.encoder.adapter_blocks)):
            optimizer = torch.optim.Adam(
                pretrained_model.encoder.adapter_blocks[block_idx].adapter.parameters(),
                lr=cfg.train.optimize.lr,
                weight_decay=cfg.train.optimize.weight_decay,
                betas=(cfg.train.optimize.beta1, cfg.train.optimize.beta2),
                eps=cfg.train.optimize.eps,
            )
            optimizers.append(optimizer)

        bce_criterion = torch.nn.BCELoss(reduction="none")

        NUM_EPOCH = cfg.train.num_epoch
        num_steps = 0

        for i in range(1, NUM_EPOCH + 1):
            # --- start pre eval ---
            prev_cer = eval(cfg, pretrained_model, tokenizer, eval_dataloader)
            prev_ref_cer = eval(cfg, pretrained_model, tokenizer, eval_ref_dataloader)

            # --- start adapter pretrain ---
            pretrained_model.train()
            torch.cuda.empty_cache()
            epoch_train_loss = 0
            accum_sec = 0
            bar = tqdm(total=len(train_dataset))
            bar.set_description(f"Train Epoch {i}  ")
            for _, bx, by, bx_len, by_len, baudio_sec, bsubsampled_vad, bsubsampled_vad_len in train_dataloader:
                accum_sec += sum(baudio_sec)
                loss = forward(
                    cfg=cfg,
                    model=pretrained_model,
                    bx=bx,
                    bx_len=bx_len,
                    bsubsampled_vad=bsubsampled_vad,
                    bsubsampled_vad_len=bsubsampled_vad_len,
                    bce_criterion=bce_criterion,
                )
                loss.backward()
                epoch_train_loss += loss.item() * bx.shape[0]

                for param_group in optimizer.param_groups:
                    bar.set_postfix({"loss": loss.item(), "step": num_steps})
                bar.update(bx.shape[0])

                if accum_sec > cfg.train.accum_sec:
                    for optimizer in optimizers:
                        optimizer.step()
                        optimizer.zero_grad()
                    num_steps += 1
                    accum_sec = 0

            # --- start after eval ---
            after_cer = eval(cfg, pretrained_model, tokenizer, eval_dataloader)
            after_ref_cer = eval(cfg, pretrained_model, tokenizer, eval_ref_dataloader)

            improvement = prev_cer - after_cer
            ref_improvement = prev_ref_cer - after_ref_cer

            # --- logging ---

            logger.info(f"Epoch {i}  train_loss: {epoch_train_loss / len(train_dataset)}")
            mlflow.log_metric("train_loss", epoch_train_loss / len(train_dataset), step=i)
            logger.info(f"Epoch {i}  prev_cer: {prev_cer}")
            mlflow.log_metric("prev_cer", prev_cer, step=i)
            logger.info(f"Epoch {i}  after_cer: {after_cer}")
            mlflow.log_metric("after_cer", after_cer, step=i)
            logger.info(f"Epoch {i}  improvement: {improvement}")
            mlflow.log_metric("improvement", improvement, step=i)
            logger.info(f"Epoch {i}  prev_ref_cer: {prev_ref_cer}")
            mlflow.log_metric("prev_ref_cer", prev_ref_cer, step=i)
            logger.info(f"Epoch {i}  after_ref_cer: {after_ref_cer}")
            mlflow.log_metric("after_ref_cer", after_ref_cer, step=i)
            logger.info(f"Epoch {i}  ref_improvement: {ref_improvement}")
            mlflow.log_metric("ref_improvement", ref_improvement, step=i)

            torch.save(
                {
                    "model": pretrained_model.module.state_dict(),
                    "model_args": pretrained_model_args,
                    "optimizer": optimizer.state_dict(),
                },
                os.path.join(mlflow_run.info.artifact_uri, f"model_{i}.pth"),
            )


if __name__ == "__main__":
    main()
