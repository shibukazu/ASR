import os
from logging import config, getLogger

import hydra
import mlflow
import torch
from conf import logging_conf
from data import LibriLightDataset, YesNoDataset
from hydra.core.hydra_config import HydraConfig
from model import Model
from omegaconf import DictConfig
from rich.logging import RichHandler
from torchaudio.functional import rnnt_loss
from torchmetrics.functional import char_error_rate
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
            vocab_size = len(dataset.label_to_idx)
            idx_to_label = {v: k for k, v in dataset.label_to_idx.items()}
        elif cfg.dataset.name == "LibriLight":
            dataset = LibriLightDataset(
                subset="1h",
                vocab_file_path="vocabs/libri_light_1h.json",
            )
            train_dataset, test_dataset = torch.utils.data.random_split(
                dataset, [int(len(dataset) * 0.8), len(dataset) - int(len(dataset) * 0.8)]
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
            vocab_size = len(dataset.vocab)
        else:
            raise NotImplementedError

        model = Model(
            vocab_size=vocab_size,
            blank_idx=blank_idx,
            encoder_input_size=cfg.model.encoder.input_size,
            encoder_hidden_size=cfg.model.encoder.hidden_size,
            encoder_num_layers=cfg.model.encoder.num_layers,
            embedding_size=cfg.model.predictor.embedding_size,
            predictor_hidden_size=cfg.model.predictor.hidden_size,
            predictor_num_layers=cfg.model.predictor.num_layers,
            jointnet_hidden_size=cfg.model.jointnet.hidden_size,
        ).to(DEVICE)
        optimizer = torch.optim.Adam(model.parameters(), lr=cfg.train.lr, weight_decay=cfg.train.weight_decay)
        # scheduler = None

        NUM_EPOCH = cfg.train.num_epoch
        for i in range(1, NUM_EPOCH + 1):
            logger.info(f"Epoch {i} has started.")
            epoch_train_loss = 0
            epoch_train_cer = 0
            cnt = 0
            accum_step = 0
            model.train()
            for j, (enc_input, pred_input, enc_input_lengths, pred_input_lengths) in enumerate(train_dataloader):
                optimizer.zero_grad()
                cnt += 1
                accum_step += 1
                enc_input = enc_input.to(DEVICE)
                pred_input = pred_input.to(DEVICE)

                padded_output, subsampled_enc_input_lengths = model(
                    padded_enc_input=enc_input,
                    enc_input_lengths=enc_input_lengths,
                    padded_pred_input=pred_input,
                    pred_input_lengths=pred_input_lengths,
                )

                loss = rnnt_loss(
                    logits=padded_output,
                    targets=pred_input,
                    logit_lengths=subsampled_enc_input_lengths.to(DEVICE),
                    target_lengths=pred_input_lengths.to(DEVICE),
                    blank=blank_idx,
                    reduction="sum",
                )
                epoch_train_loss += loss.item() / enc_input.shape[0]

                hyp_tokens = model.greedy_inference(enc_inputs=enc_input, enc_input_lengths=enc_input_lengths)
                ans_tokens = [pred_input[i, : pred_input_lengths[i]].tolist() for i in range(pred_input.shape[0])]
                hyp_texts = ["".join([idx_to_label[idx] for idx in hyp_token]) for hyp_token in hyp_tokens]
                ans_texts = ["".join([idx_to_label[idx] for idx in ans_token]) for ans_token in ans_tokens]

                epoch_train_cer += char_error_rate(hyp_texts, ans_texts)

                if accum_step % cfg.train.accum_step == 0:
                    loss.backward()
                    optimizer.step()
                    accum_step = 0

            logger.info(f"Epoch {i} has finished.")
            logger.info(f"Train loss: {epoch_train_loss / cnt}")
            mlflow.log_metric("train_loss", epoch_train_loss / cnt, step=i)
            logger.info(f"Train CER: {epoch_train_cer / cnt}")
            mlflow.log_metric("train_cer", epoch_train_cer / cnt, step=i)

            model.eval()
            epoch_test_loss = 0
            epoch_test_cer = 0
            cnt = 0
            with torch.no_grad():
                for j, (enc_input, pred_input, enc_input_lengths, pred_input_lengths) in enumerate(test_dataloader):
                    cnt += 1
                    enc_input = enc_input.to(DEVICE)
                    pred_input = pred_input.to(DEVICE)

                    padded_output, subsampled_enc_input_lengths = model(
                        padded_enc_input=enc_input,
                        enc_input_lengths=enc_input_lengths,
                        padded_pred_input=pred_input,
                        pred_input_lengths=pred_input_lengths,
                    )
                    loss = rnnt_loss(
                        logits=padded_output,
                        targets=pred_input,
                        logit_lengths=subsampled_enc_input_lengths.to(DEVICE),
                        target_lengths=pred_input_lengths.to(DEVICE),
                        blank=blank_idx,
                        reduction="sum",
                    )
                    epoch_test_loss += loss.item() / enc_input.shape[0]

                    hyp_tokens = model.greedy_inference(enc_inputs=enc_input, enc_input_lengths=enc_input_lengths)
                    ans_tokens = [pred_input[i, : pred_input_lengths[i]].tolist() for i in range(pred_input.shape[0])]
                    hyp_texts = ["".join([idx_to_label[idx] for idx in hyp_token]) for hyp_token in hyp_tokens]
                    ans_texts = ["".join([idx_to_label[idx] for idx in ans_token]) for ans_token in ans_tokens]

                    epoch_test_cer += char_error_rate(hyp_texts, ans_texts)

                mlflow.log_metric("test_loss", epoch_test_loss / cnt, step=i)
                logger.info(f"Test loss: {epoch_test_loss / cnt}")
                mlflow.log_metric("test_cer", epoch_test_cer / cnt, step=i)
                logger.info(f"Test CER: {epoch_test_cer / cnt}")

        torch.save(
            {
                "model_state_dict": model.state_dict(),
            },
            os.path.join(f"model_{cfg.dataset.name}.pth"),
        )


if __name__ == "__main__":
    main()
