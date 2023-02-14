import math
import os
from logging import config, getLogger

import hydra
import mlflow
import torch
from conf import logging_conf
from data import LibriSpeechDataset, YesNoDataset, get_dataloader
from hydra.core.hydra_config import HydraConfig
from model import CausalConformerModel, TorchAudioConformerModel
from modules.spec_aug import SpecAug
from omegaconf import DictConfig
from rich.logging import RichHandler
from tokenizer import SentencePieceTokenizer
from torchaudio.transforms import RNNTLoss
from torchmetrics.functional import char_error_rate, word_error_rate
from tqdm import tqdm
from util.mlflow import log_params_from_omegaconf_dict


class DataParallel(torch.nn.DataParallel):
    def __getattr__(self, name):
        try:
            return super().__getattr__(name)
        except AttributeError:
            return getattr(self.module, name)


@hydra.main(version_base=None, config_path="conf", config_name=None)
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

        DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        logger.info(f"Training on {DEVICE}.")

        tokenizer = SentencePieceTokenizer(
            model_file_path=cfg.tokenizer.model_file_path,
        )
        blank_idx = tokenizer.blank_token_id
        vocab_size = tokenizer.num_tokens
        if cfg.dataset.name == "YesNo":
            train_dataset = YesNoDataset(
                json_file_path=cfg.dataset.train.json_file_path, resampling_rate=16000, tokenizer=tokenizer
            )
            dev_dataset = YesNoDataset(
                json_file_path=cfg.dataset.dev.json_file_path, resampling_rate=16000, tokenizer=tokenizer
            )

        elif cfg.dataset.name == "Librispeech":
            spec_aug = SpecAug(
                freq_mask_max_length=cfg.model.spec_aug.freq_mask_max_length,
                time_mask_max_length=cfg.model.spec_aug.time_mask_max_length,
                num_freq_mask=cfg.model.spec_aug.num_freq_mask,
                num_time_mask=cfg.model.spec_aug.num_time_mask,
            )
            train_dataset = LibriSpeechDataset(
                json_file_path=cfg.dataset.train.json_file_path,
                resampling_rate=16000,
                tokenizer=tokenizer,
                spec_aug=spec_aug,
            )
            dev_dataset = LibriSpeechDataset(
                json_file_path=cfg.dataset.dev.json_file_path,
                resampling_rate=16000,
                tokenizer=tokenizer,
                spec_aug=spec_aug,
            )
        else:
            raise NotImplementedError

        train_dataloader = get_dataloader(
            train_dataset,
            batch_sec=cfg.train.batch_sec,
            num_workers=4,
            pin_memory=True,
            pad_idx=tokenizer.pad_token_id,
        )
        dev_dataloader = get_dataloader(
            dev_dataset,
            batch_sec=cfg.train.batch_sec,
            num_workers=4,
            pin_memory=True,
            pad_idx=tokenizer.pad_token_id,
        )

        # torch.autograd.set_detect_anomaly(True)
        model_args = {
            "vocab_size": vocab_size,
            "blank_idx": blank_idx,
            "encoder_input_size": cfg.model.encoder.input_size,
            "encoder_subsampled_input_size": cfg.model.encoder.subsampled_input_size,
            "encoder_num_conformer_blocks": cfg.model.encoder.num_conformer_blocks,
            "encoder_ff_hidden_size": cfg.model.encoder.ff_hidden_size,
            "encoder_conv_hidden_size": cfg.model.encoder.conv_hidden_size,
            "encoder_conv_kernel_size": cfg.model.encoder.conv_kernel_size,
            "encoder_mha_num_heads": cfg.model.encoder.mha_num_heads,
            "encoder_dropout": cfg.model.encoder.dropout,
            "encoder_subsampling_kernel_size1": cfg.model.encoder.subsampling_kernel_size1,
            "encoder_subsampling_stride1": cfg.model.encoder.subsampling_stride1,
            "encoder_subsampling_kernel_size2": cfg.model.encoder.subsampling_kernel_size2,
            "encoder_subsampling_stride2": cfg.model.encoder.subsampling_stride2,
            "encoder_num_previous_frames": cfg.model.encoder.num_previous_frames,
            "embedding_size": cfg.model.predictor.embedding_size,
            "predictor_hidden_size": cfg.model.predictor.hidden_size,
            "predictor_num_layers": cfg.model.predictor.num_layers,
            "jointnet_hidden_size": cfg.model.jointnet.hidden_size,
            "decoder_buffer_size": cfg.decoder.buffer_size,
        }
        if cfg.model.name == "CausalConformer":
            model = CausalConformerModel(**model_args)
        elif cfg.model.name == "TorchAudioConformer":
            model = TorchAudioConformerModel(**model_args)
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
        scheduler = torch.optim.lr_scheduler.LambdaLR(
            optimizer,
            lambda s: cfg.model.encoder.subsampled_input_size**-0.5
            * min((s + 1) ** -0.5, (s + 1) * cfg.train.optimize.warmup_steps**-1.5),
        )
        criterion = RNNTLoss(blank=blank_idx, reduction="sum")

        NUM_EPOCH = cfg.train.num_epoch
        num_steps = 0

        for i in range(1, NUM_EPOCH + 1):
            torch.cuda.empty_cache()

            model.train()
            epoch_train_loss = 0
            accum_sec = 0
            bar = tqdm(total=len(train_dataset))
            bar.set_description(f"Train Epoch {i}  ")
            for _, benc_input, bpred_input, benc_input_length, bpred_input_length, baudio_sec in train_dataloader:
                accum_sec += sum(baudio_sec)
                benc_input = benc_input.to(DEVICE)
                bpred_input = bpred_input.to(DEVICE)

                bpadded_output, bsubsampled_enc_input_length = model(
                    padded_enc_input=benc_input,
                    enc_input_lengths=benc_input_length,
                    padded_pred_input=bpred_input,
                    pred_input_lengths=bpred_input_length,
                )
                loss = criterion(
                    logits=bpadded_output,
                    targets=bpred_input,
                    logit_lengths=bsubsampled_enc_input_length.to(DEVICE),
                    target_lengths=bpred_input_length.to(DEVICE),
                )
                loss = loss / bpred_input.shape[0]
                loss.backward()
                epoch_train_loss += loss.item() * bpred_input.shape[0]

                for param_group in optimizer.param_groups:
                    bar.set_postfix({"loss": loss.item(), "lr": param_group["lr"], "step": num_steps})
                bar.update(bpred_input.shape[0])

                if accum_sec > cfg.train.accum_sec:
                    grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.train.optimize.max_grad_norm)
                    if math.isnan(grad_norm):
                        logger.error("grad norm is nan. Do not update model.")
                        logger.error(f"loss value: {loss.item()}")
                    else:
                        optimizer.step()
                        scheduler.step()
                        optimizer.zero_grad()
                        accum_sec = 0
                        num_steps += 1

            logger.info(f"Train loss: {epoch_train_loss / len(train_dataset)}")
            mlflow.log_metric("train_loss", epoch_train_loss / len(train_dataset), step=i)

            model.eval()
            epoch_dev_loss = 0
            epoch_dev_cer = 0
            epoch_dev_wer = 0
            bar = tqdm(total=len(dev_dataset))
            bar.set_description(f"Valid Epoch {i}")
            with torch.no_grad():
                for _, benc_input, bpred_input, benc_input_length, bpred_input_length, baudio_sec in dev_dataloader:
                    benc_input = benc_input.to(DEVICE)
                    bpred_input = bpred_input.to(DEVICE)

                    bpadded_output, bsubsampled_enc_input_length = model(
                        padded_enc_input=benc_input,
                        enc_input_lengths=benc_input_length,
                        padded_pred_input=bpred_input,
                        pred_input_lengths=bpred_input_length,
                    )

                    loss = criterion(
                        logits=bpadded_output,
                        targets=bpred_input,
                        logit_lengths=bsubsampled_enc_input_length.to(DEVICE),
                        target_lengths=bpred_input_length.to(DEVICE),
                    )
                    loss = loss / bpred_input.shape[0]

                    epoch_dev_loss += loss.item() * bpred_input.shape[0]

                    if i >= 30 and i % 5 == 0 and cfg.do_decode:
                        if cfg.decoder.type == "streaming_greedy":
                            bhyp_token_indices = model.streaming_greedy_inference(
                                enc_inputs=benc_input, enc_input_lengths=benc_input_length
                            )
                        elif cfg.decoder.type == "non_streaming_greedy":
                            bhyp_token_indices = model.greedy_inference(
                                enc_inputs=benc_input, enc_input_lengths=benc_input_length
                            )
                        else:
                            raise NotImplementedError
                        bans_token_indices = [
                            bpred_input[i, : bpred_input_length[i]].tolist() for i in range(bpred_input.shape[0])
                        ]
                        bhyp_text = tokenizer.batch_token_ids_to_text(bhyp_token_indices)
                        bans_text = tokenizer.batch_token_ids_to_text(bans_token_indices)

                        epoch_dev_cer += char_error_rate(bhyp_text, bans_text) * benc_input.shape[0]
                        epoch_dev_wer += word_error_rate(bhyp_text, bans_text) * benc_input.shape[0]

                    bar.update(bpred_input.shape[0])

                logger.info(f"Dev loss: {epoch_dev_loss / len(dev_dataset)}")
                mlflow.log_metric("dev_loss", epoch_dev_loss / len(dev_dataset), step=i)

                if i >= 20 and i % 5 == 0 and cfg.do_decode:
                    mlflow.log_metric("dev_cer", epoch_dev_cer / len(dev_dataset), step=i)
                    logger.info(f"Dev CER: {epoch_dev_cer / len(dev_dataset)}")
                    mlflow.log_metric("dev_wer", epoch_dev_wer / len(dev_dataset), step=i)
                    logger.info(f"Dev WER: {epoch_dev_wer / len(dev_dataset)}")

            if i % 5 == 0:
                torch.save(
                    {
                        "model": model.module.state_dict(),
                        "model_args": model_args,
                        "optimizer": optimizer.state_dict(),
                        "scheduler": scheduler.state_dict(),
                        "dev_loss": epoch_dev_loss / len(dev_dataset),
                    },
                    os.path.join(mlflow_run.info.artifact_uri, f"model_{i}.pth"),
                )


if __name__ == "__main__":
    main()
