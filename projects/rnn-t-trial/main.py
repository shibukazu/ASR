import os
from logging import config, getLogger

import hydra
import mlflow
import torch
from conf import logging_conf
from data import LibriLightDataset, YesNoDataset
from hydra.core.hydra_config import HydraConfig
from model import CausalConformerModel
from modules.spec_aug import SpecAug
from omegaconf import DictConfig
from rich.logging import RichHandler
from torchaudio.functional import rnnt_loss
from torchmetrics.functional import char_error_rate, word_error_rate
from util.mlflow import log_params_from_omegaconf_dict


@hydra.main(version_base=None, config_path="conf", config_name=None)
def main(cfg: DictConfig):
    CONF_NAME = HydraConfig.get().job.config_name
    EXPERIMENT_NAME = CONF_NAME
    ARTIFACT_LOCATION = "./artifacts"
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
            train_dataset, test_dataset = torch.utils.data.random_split(
                dataset, [int(len(dataset) * 0.95), len(dataset) - int(len(dataset) * 0.95)]
            )
            train_dataloader = torch.utils.data.DataLoader(
                train_dataset,
                batch_size=cfg.train.num_batch,
                shuffle=True,
                collate_fn=dataset.collate_fn,
                drop_last=True,
            )
            test_dataloader = torch.utils.data.DataLoader(
                test_dataset,
                batch_size=cfg.train.num_batch,
                shuffle=False,
                collate_fn=dataset.collate_fn,
                drop_last=False,
            )
            blank_idx = dataset.blank_idx
            idx_to_token = dataset.idx_to_token
            vocab_size = len(dataset.idx_to_token)
        elif cfg.dataset.name == "LibriLight":
            spec_aug = SpecAug(
                freq_mask_max_length=cfg.model.spec_aug.freq_mask_max_length,
                time_mask_max_length=cfg.model.spec_aug.time_mask_max_length,
                num_freq_mask=cfg.model.spec_aug.num_freq_mask,
                num_time_mask=cfg.model.spec_aug.num_time_mask,
            )
            train_dataset = LibriLightDataset(
                subset="9h", vocab_file_path="vocabs/libri_light_9h.json", spec_aug=spec_aug
            )
            test_dataset = LibriLightDataset(
                subset="1h",
                vocab_file_path="vocabs/libri_light_9h.json",
            )

            train_dataloader = torch.utils.data.DataLoader(
                train_dataset,
                batch_size=cfg.train.num_batch,
                shuffle=True,
                collate_fn=train_dataset.collate_fn,
                drop_last=True,
            )
            test_dataloader = torch.utils.data.DataLoader(
                test_dataset,
                batch_size=cfg.train.num_batch,
                shuffle=False,
                collate_fn=test_dataset.collate_fn,
                drop_last=False,
            )
            blank_idx = train_dataset.blank_idx
            idx_to_token = train_dataset.idx_to_token
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
        optimizer = torch.optim.Adam(model.parameters(), lr=cfg.train.lr, weight_decay=cfg.train.weight_decay)
        # scheduler = None

        NUM_EPOCH = cfg.train.num_epoch
        min_train_wer = 1.0
        for i in range(1, NUM_EPOCH + 1):
            logger.info(f"Epoch {i} has started.")

            model.train()
            epoch_train_loss = 0
            epoch_train_cer = 0
            epoch_train_wer = 0
            accum_step = 0
            for j, (benc_input, bpred_input, benc_input_length, bpred_input_length) in enumerate(train_dataloader):
                optimizer.zero_grad()
                accum_step += 1
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
                epoch_train_loss += loss.item()
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
                bhyp_text = [
                    "".join([idx_to_token[idx] for idx in hyp_token_indices])
                    for hyp_token_indices in bhyp_token_indices
                ]
                bans_text = [
                    "".join([idx_to_token[idx] for idx in ans_token_indices])
                    for ans_token_indices in bans_token_indices
                ]

                # char_error_rate and word_error_rate are automatically reduced by average
                epoch_train_cer += char_error_rate(bhyp_text, bans_text) * benc_input.shape[0]
                epoch_train_wer += word_error_rate(bhyp_text, bans_text) * benc_input.shape[0]

                if accum_step % cfg.train.accum_step == 0:
                    loss.backward()
                    optimizer.step()
                    accum_step = 0

            logger.info(f"Epoch {i} has finished.")
            logger.info(f"Train loss: {epoch_train_loss / len(train_dataset)}")
            mlflow.log_metric("train_loss", epoch_train_loss / len(train_dataset), step=i)
            logger.info(f"Train CER: {epoch_train_cer / len(train_dataset)}")
            mlflow.log_metric("train_cer", epoch_train_cer / len(train_dataset), step=i)
            logger.info(f"Train WER: {epoch_train_wer / len(train_dataset)}")
            mlflow.log_metric("train_wer", epoch_train_wer / len(train_dataset), step=i)

            min_train_wer = min(min_train_wer, epoch_train_wer / len(train_dataset))

            model.eval()
            epoch_test_loss = 0
            epoch_test_cer = 0
            epoch_test_wer = 0
            with torch.no_grad():
                for j, (benc_input, bpred_input, benc_input_length, bpred_input_length) in enumerate(test_dataloader):
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
                    epoch_test_loss += loss.item()

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
                    bhyp_text = [
                        "".join([idx_to_token[idx] for idx in hyp_token_indices])
                        for hyp_token_indices in bhyp_token_indices
                    ]
                    bans_text = [
                        "".join([idx_to_token[idx] for idx in ans_token_indices])
                        for ans_token_indices in bans_token_indices
                    ]

                    epoch_test_cer += char_error_rate(bhyp_text, bans_text) * benc_input.shape[0]
                    epoch_test_wer += word_error_rate(bhyp_text, bans_text) * benc_input.shape[0]

                mlflow.log_metric("test_loss", epoch_test_loss / len(test_dataset), step=i)
                logger.info(f"Test loss: {epoch_test_loss / len(test_dataset)}")
                mlflow.log_metric("test_cer", epoch_test_cer / len(test_dataset), step=i)
                logger.info(f"Test CER: {epoch_test_cer / len(test_dataset)}")
                mlflow.log_metric("test_wer", epoch_test_wer / len(test_dataset), step=i)
                logger.info(f"Test WER: {epoch_test_wer / len(test_dataset)}")

        if min_train_wer > 0.8:
            # learning failed
            exit(1)
        else:
            torch.save(
                {
                    "model_state_dict": model.state_dict(),
                },
                os.path.join(f"model_{cfg.dataset.name}.pth"),
            )


if __name__ == "__main__":
    main()
