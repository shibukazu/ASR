import os
from logging import config, getLogger

import hydra
import mlflow
import torch
from conf import logging_conf
from data import LibriSpeechDataset, YesNoDataset
from hydra.core.hydra_config import HydraConfig
from model import CausalConformerModel
from modules.scheduler import TransformerLR
from modules.spec_aug import SpecAug
from omegaconf import DictConfig
from rich.logging import RichHandler
from torchaudio.functional import rnnt_loss
from tqdm import tqdm
from util.mlflow import log_params_from_omegaconf_dict


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

    with mlflow.start_run(experiment_id=experiment_id) as mlflow_run:
        LOG_DIR = mlflow_run.info.artifact_uri
        config.dictConfig(logging_conf.config_generator(LOG_DIR))
        logger = getLogger()
        logger.handlers[0] = RichHandler(markup=True)  # pretty formatting
        # save parameters from hydra to mlflow
        log_params_from_omegaconf_dict(cfg)

        DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        logger.info(f"Training on {DEVICE}.")

        if cfg.dataset.name == "YesNo":
            dataset = YesNoDataset(
                wav_dir_path="datasets/waves_yesno/",
                model_sample_rate=16000,
            )
            train_dataset, dev_dataset = torch.utils.data.random_split(
                dataset, [int(len(dataset) * 0.9), len(dataset) - int(len(dataset) * 0.9)]
            )
            train_dataloader = torch.utils.data.DataLoader(
                train_dataset,
                batch_size=cfg.train.num_batch,
                shuffle=True,
                collate_fn=dataset.collate_fn,
                drop_last=True,
                num_workers=2,
                pin_memory=True,
            )
            dev_dataloader = torch.utils.data.DataLoader(
                dev_dataset,
                batch_size=cfg.train.num_batch,
                shuffle=False,
                collate_fn=dataset.collate_fn,
                drop_last=False,
                num_workers=2,
                pin_memory=True,
            )
            blank_idx = dataset.blank_idx
            vocab_size = len(dataset.idx_to_token)
        elif cfg.dataset.name == "Librispeech":
            spec_aug = SpecAug(
                freq_mask_max_length=cfg.model.spec_aug.freq_mask_max_length,
                time_mask_max_length=cfg.model.spec_aug.time_mask_max_length,
                num_freq_mask=cfg.model.spec_aug.num_freq_mask,
                num_time_mask=cfg.model.spec_aug.num_time_mask,
            )
            train_dataset = LibriSpeechDataset(
                root="datasets/",
                split="train",
                vocab_file_path="vocabs/librispeech_train_960h.json",
                spec_aug=spec_aug,
            )
            dev_dataset = LibriSpeechDataset(
                root="datasets/",
                split=cfg.dataset.dev_split,
                vocab_file_path="vocabs/librispeech_train_960h.json",
                spec_aug=None,
            )

            train_dataloader = torch.utils.data.DataLoader(
                train_dataset,
                batch_size=cfg.train.num_batch,
                shuffle=True,
                collate_fn=train_dataset.collate_fn,
                drop_last=True,
                num_workers=4,
                pin_memory=True,
            )
            dev_dataloader = torch.utils.data.DataLoader(
                dev_dataset,
                batch_size=cfg.train.num_batch,
                shuffle=True,
                collate_fn=train_dataset.collate_fn,
                drop_last=True,
                num_workers=4,
                pin_memory=True,
            )
            blank_idx = train_dataset.blank_idx
            vocab_size = len(train_dataset.idx_to_token)
        else:
            raise NotImplementedError
        torch.autograd.set_detect_anomaly(True)
        model = CausalConformerModel(
            vocab_size=vocab_size,
            blank_idx=blank_idx,
            encoder_input_size=cfg.model.encoder.input_size,
            encoder_subsampled_input_size=cfg.model.encoder.subsampled_input_size,
            encoder_num_conformer_blocks=cfg.model.encoder.num_conformer_blocks,
            encoder_ff_hidden_size=cfg.model.encoder.ff_hidden_size,
            encoder_conv_hidden_size=cfg.model.encoder.conv_hidden_size,
            encoder_conv_kernel_size=cfg.model.encoder.conv_kernel_size,
            encoder_mha_num_heads=cfg.model.encoder.mha_num_heads,
            encoder_dropout=cfg.model.encoder.dropout,
            encoder_subsampling_kernel_size1=cfg.model.encoder.subsampling_kernel_size1,
            encoder_subsampling_stride1=cfg.model.encoder.subsampling_stride1,
            encoder_subsampling_kernel_size2=cfg.model.encoder.subsampling_kernel_size2,
            encoder_subsampling_stride2=cfg.model.encoder.subsampling_stride2,
            encoder_num_previous_frames=cfg.model.encoder.num_previous_frames,
            embedding_size=cfg.model.predictor.embedding_size,
            predictor_hidden_size=cfg.model.predictor.hidden_size,
            predictor_num_layers=cfg.model.predictor.num_layers,
            jointnet_hidden_size=cfg.model.jointnet.hidden_size,
            decoder_buffer_size=cfg.decoder.buffer_size,
        ).to(DEVICE)
        optimizer = torch.optim.Adam(
            model.parameters(),
            lr=cfg.train.optimize.lr,
            weight_decay=cfg.train.optimize.weight_decay,
            betas=(cfg.train.optimize.beta1, cfg.train.optimize.beta2),
            eps=cfg.train.optimize.eps,
        )
        scheduler = TransformerLR(
            optimizer,
            d_model=cfg.model.encoder.subsampled_input_size,
            warmup_steps=cfg.train.optimize.warmup_steps,
        )

        NUM_EPOCH = cfg.train.num_epoch

        for i in range(1, NUM_EPOCH + 1):
            logger.info(f"Epoch {i} has started.")

            model.train()
            epoch_train_loss = 0
            accum_sec = 0
            for benc_input, bpred_input, benc_input_length, bpred_input_length, baudio_sec in tqdm(train_dataloader):
                accum_sec += sum(baudio_sec)
                benc_input = benc_input.to(DEVICE)
                bpred_input = bpred_input.to(DEVICE)

                bpadded_output, bsubsampled_enc_input_length = model(
                    padded_enc_input=benc_input,
                    enc_input_lengths=benc_input_length,
                    padded_pred_input=bpred_input,
                    pred_input_lengths=bpred_input_length,
                )

                loss = rnnt_loss(
                    logits=bpadded_output,
                    targets=bpred_input,
                    logit_lengths=bsubsampled_enc_input_length.to(DEVICE),
                    target_lengths=bpred_input_length.to(DEVICE),
                    blank=blank_idx,
                    reduction="sum",
                )
                loss.backward()
                epoch_train_loss += loss.item()

                if accum_sec > cfg.train.accum_sec:
                    optimizer.step()
                    scheduler.step()
                    optimizer.zero_grad()
                    accum_sec = 0

            logger.info(f"Train loss: {epoch_train_loss / len(train_dataset)}")
            mlflow.log_metric("train_loss", epoch_train_loss / len(train_dataset), step=i)

            model.eval()
            epoch_dev_loss = 0
            with torch.no_grad():
                for benc_input, bpred_input, benc_input_length, bpred_input_length, baudio_sec in tqdm(dev_dataloader):
                    benc_input = benc_input.to(DEVICE)
                    bpred_input = bpred_input.to(DEVICE)

                    bpadded_output, bsubsampled_enc_input_length = model(
                        padded_enc_input=benc_input,
                        enc_input_lengths=benc_input_length,
                        padded_pred_input=bpred_input,
                        pred_input_lengths=bpred_input_length,
                    )
                    loss = rnnt_loss(
                        logits=bpadded_output,
                        targets=bpred_input,
                        logit_lengths=bsubsampled_enc_input_length.to(DEVICE),
                        target_lengths=bpred_input_length.to(DEVICE),
                        blank=blank_idx,
                        reduction="sum",
                    )
                    epoch_dev_loss += loss.item()

                logger.info(f"Dev loss: {epoch_dev_loss / len(dev_dataset)}")
                mlflow.log_metric("dev_loss", epoch_dev_loss / len(dev_dataset), step=i)

            if i % 5 == 0:
                torch.save(
                    {
                        "model": model.state_dict(),
                        "optimizer": optimizer.state_dict(),
                        "scheduler": scheduler.state_dict(),
                        "dev_loss": epoch_dev_loss / len(dev_dataset),
                    },
                    os.path.join(mlflow_run.info.artifact_uri, f"model_{cfg.dataset.name}_{i}.pth"),
                )


if __name__ == "__main__":
    main()
