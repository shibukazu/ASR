import os
import pickle
from logging import config, getLogger

import hydra
import mlflow
import torch
from conf import logging_conf
from data import LibriLightDataset, TEDLIUMRelease2Dataset, get_dataloader
from hydra.core.hydra_config import HydraConfig
from model import Model
from modules.decoders.ctc import greedy_decoder
from modules.transformers.scheduler import TransformerLR
from omegaconf import DictConfig
from quantizer import Quantizer
from rich.logging import RichHandler
from sampler import KLSampler, RandomSampler, TrainKLSampler
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

        DEVICE = torch.device(f"cuda:{cfg.train.cuda}" if torch.cuda.is_available() else "cpu")
        logger.info(f"Training on {DEVICE}.")
        vocab_file_path = cfg.dataset.vocab_file_path
        logger.info("Create Train Dataset.")
        train_dataset = LibriLightDataset(
            root="datasets/librispeech_finetuning",
            subset=cfg.dataset.librilight_split,
            vocab_file_path=vocab_file_path,
        )
        logger.info(f"Train dataset size: {len(train_dataset)}")
        logger.info("Create Test Dataset.")

        with open("kl_divergences.pickle", "rb") as f:
            kl_divergences = pickle.load(f)
        talk_id_kl_list = sorted(kl_divergences.items(), key=lambda x: x[1], reverse=True)
        TARGET_DURATION = float(cfg.dataset.ted_sec)
        ted_total_duration = 0
        target_talk_ids = []

        for talk_id, kl in talk_id_kl_list:
            ted_specific_dataset = TEDLIUMRelease2Dataset(
                root="datasets/TEDLIUM_release2",
                subset="train",
                talk_id=talk_id,
                vocab_file_path="vocabs/librilight_10h.json",
            )
            for idx, x, x_len, y, y_len, text in ted_specific_dataset:
                ted_total_duration += x_len / 16000
            target_talk_ids.append(talk_id)
            if ted_total_duration > TARGET_DURATION:
                break
        test_datasets = []
        for talk_id in target_talk_ids:
            ted_specific_dataset = TEDLIUMRelease2Dataset(
                root="datasets/TEDLIUM_release2",
                subset="train",
                talk_id=talk_id,
                vocab_file_path="vocabs/librilight_10h.json",
            )
            test_collate_fn = ted_specific_dataset.collate_fn
            test_vocab = ted_specific_dataset.vocab
            test_datasets.append(ted_specific_dataset)

        test_dataset = torch.utils.data.ConcatDataset(test_datasets)

        total_duration = 0.0
        for i in range(len(test_dataset)):
            total_duration += test_dataset[i][2] / 16000
        logger.info(f"Total duration of Test dataset: {total_duration / 60:.2f} min")

        logger.info(f"Test dataset size: {len(test_dataset)}")
        test_dataloader = get_dataloader(
            test_dataset, cfg.train.num_batch, shuffle=False, drop_last=False, collate_fn=test_collate_fn
        )

        assert train_dataset.vocab == test_vocab

        # NOTE: 実験の正当性のために毎回Quantizeを行うものとする
        quantizer = Quantizer(cfg.model.quantizer.name, DEVICE)
        limit_sec = float(cfg.selection.limit_sec)
        if cfg.selection.type == "random":
            sampler = RandomSampler(
                quantizer=quantizer,
                train_dataset=train_dataset,
                test_dataset=test_dataset,
                limit=limit_sec,
                device=DEVICE,
            )
        elif cfg.selection.type == "kl":
            sampler = KLSampler(
                quantizer=quantizer,
                train_dataset=train_dataset,
                test_dataset=test_dataset,
                limit=limit_sec,
                device=DEVICE,
            )
        elif cfg.selection.type == "train_kl":
            # This selection is used for debug only.
            sampler = TrainKLSampler(
                quantizer=quantizer,
                dataset=train_dataset,
                limit=limit_sec,
                device=DEVICE,
            )
        else:
            raise ValueError(f"Unknown selection type: {cfg.selection.type}")
        sampled_train_dataset = sampler.sample()
        # show total duration of sampled dataset
        total_duration = 0
        for i in range(len(sampled_train_dataset)):
            total_duration += sampled_train_dataset[i][2] / 16000
        logger.info(f"Total duration of sampled dataset: {total_duration / 60:.2f} min")

        # train datasetからサンプリングされたことを確認する（データリークの防止）
        sampled_train_id = sampled_train_dataset[0][0]
        assert train_dataset[sampled_train_id][5] == sampled_train_dataset[0][5]

        train_dataloader = get_dataloader(
            sampled_train_dataset,
            cfg.train.num_batch,
            shuffle=True,
            drop_last=True,
            collate_fn=train_dataset.collate_fn,
        )
        torch.cuda.empty_cache()
        num_label = len(train_dataset.vocab.keys())
        model = Model(nlabel=num_label, cfg=cfg).to(DEVICE)
        ctc_loss = torch.nn.CTCLoss(reduction="sum", blank=train_dataset.ctc_token_id)
        optimizer = torch.optim.Adam(
            model.parameters(),
            lr=cfg.train.optimize.lr,
            betas=(cfg.train.optimize.beta1, cfg.train.optimize.beta2),
            eps=cfg.train.optimize.eps,
        )
        scheduler = TransformerLR(
            optimizer, d_model=model.hidden_size, warmup_steps=cfg.train.optimize.warmup_steps
        )  # Warmup終了時点でおよそ0.0017になっている

        NUM_EPOCH = cfg.train.num_epoch
        # NUM_ACCUM_STEP = cfg.train.num_accum_step
        NUM_ACCUM_SEC = cfg.train.num_accum_sec

        min_train_wer = 1.0
        min_test_wer = 1.0
        for epoch in range(NUM_EPOCH):
            logger.info(f"Epoch {epoch + 1}/{NUM_EPOCH}")
            model.train()
            train_epoch_loss = 0
            train_epoch_cer = 0
            train_epoch_wer = 0
            train_cnt = 0
            accum_sec = 0.0
            for i, (bidx, bx, bx_len, by, by_len, _) in enumerate(train_dataloader):
                bx = bx.to(DEVICE)
                bx_len = bx_len.to(DEVICE)
                by = by.to(DEVICE)
                by_len = by_len.to(DEVICE)
                log_probs, y_lengths = model(bx, bx_len)
                loss = ctc_loss(log_probs.transpose(1, 0), by, y_lengths, by_len)
                loss.backward()

                accum_sec += bx_len.sum() / 16000

                if accum_sec >= NUM_ACCUM_SEC:
                    optimizer.step()
                    scheduler.step()
                    optimizer.zero_grad()
                    accum_sec = 0.0

                train_epoch_loss += loss.item() / bx.size(0)

                # calculate CER
                hypothesis = torch.argmax(log_probs, dim=-1)
                hypotheses = greedy_decoder(hypothesis, train_dataset.vocab, "[PAD]", "|", "_")
                answers = greedy_decoder(by, train_dataset.vocab, "[PAD]", "|", "_")
                train_epoch_cer += char_error_rate(hypotheses, answers)
                train_epoch_wer += word_error_rate(hypotheses, answers)

                train_cnt += 1

            mlflow.log_metric("train_loss", train_epoch_loss / train_cnt, step=epoch)
            mlflow.log_metric("train_cer", train_epoch_cer / train_cnt, step=epoch)
            mlflow.log_metric("train_wer", train_epoch_wer / train_cnt, step=epoch)

            min_train_wer = min(min_train_wer, train_epoch_wer / train_cnt)
            mlflow.log_metric("min_train_wer", min_train_wer, step=epoch)

            logger.info(f"Train Loss: {train_epoch_loss / train_cnt:.4f}")
            logger.info(f"Train CER: {train_epoch_cer / train_cnt:.4f}")
            logger.info(f"Train WER: {train_epoch_wer / train_cnt:.4f}")

            model.eval()
            test_epoch_loss = 0
            test_epoch_cer = 0
            test_epoch_wer = 0
            test_cnt = 0
            with torch.no_grad():
                for i, (bidx, bx, bx_len, by, by_len, _) in enumerate(test_dataloader):
                    bx = bx.to(DEVICE)
                    bx_len = bx_len.to(DEVICE)
                    by = by.to(DEVICE)
                    by_len = by_len.to(DEVICE)
                    log_probs, y_lengths = model(bx, bx_len)
                    loss = ctc_loss(log_probs.transpose(1, 0), by, y_lengths, by_len)
                    test_epoch_loss += loss.item() / bx.size(0)

                    # calculate CER
                    hypothesis = torch.argmax(log_probs, dim=-1)
                    hypotheses = greedy_decoder(hypothesis, train_dataset.vocab, "[PAD]", "|", "_")
                    answers = greedy_decoder(by, train_dataset.vocab, "[PAD]", "|", "_")
                    test_epoch_cer += char_error_rate(hypotheses, answers)
                    test_epoch_wer += word_error_rate(hypotheses, answers)

                    test_cnt += 1

            mlflow.log_metric("test_loss", test_epoch_loss / test_cnt, step=epoch)
            mlflow.log_metric("test_cer", test_epoch_cer / test_cnt, step=epoch)
            mlflow.log_metric("test_wer", test_epoch_wer / test_cnt, step=epoch)

            min_test_wer = min(min_test_wer, test_epoch_wer / test_cnt)
            mlflow.log_metric("min_test_wer", min_test_wer, step=epoch)

            logger.info(f"Test loss: {test_epoch_loss / test_cnt:.4f}")
            logger.info(f"Test CER: {test_epoch_cer / test_cnt:.4f}")
            logger.info(f"Test WER: {test_epoch_wer / test_cnt:.4f}")

            checkpoint_dir = f"cpts/{EXPERIMENT_NAME}/{mlflow_run.info.run_id}"
            os.makedirs(checkpoint_dir, exist_ok=True)
            # torch.save(model.state_dict(), f"{checkpoint_dir}/epoch_{epoch}_{test_epoch_cer / test_cnt:.4f}.pth")
            # mlflow.log_artifact(f"{checkpoint_dir}/epoch_{epoch}_{test_epoch_cer / test_cnt:.4f}.pth")
        mlflow.log_artifact(LOG_DIR)


if __name__ == "__main__":
    main()
