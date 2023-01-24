import pickle
from logging import config, getLogger

import hydra
import mlflow
import torch
from conf import logging_conf
from data import LibriLightDataset, get_dataloader
from hydra.core.hydra_config import HydraConfig
from model import Model
from modules.decoders.ctc import greedy_decoder
from modules.transformers.scheduler import TransformerLR
from omegaconf import DictConfig
from rich.logging import RichHandler
from sampler import RandomSampler
from torchmetrics.functional import char_error_rate, word_error_rate
from util.mlflow import log_params_from_omegaconf_dict

from utils import calculate_tf_idf_over_ds, cos_sim


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
        # ---- Sampling -----
        sampling_pool = LibriLightDataset(
            subset="9h",
            identifier_to_phones_file_path="phones/librispeech_normalized_phones_no_bcl.json",
            vocab_file_path="vocabs/libri-light_9h.json",
        )
        train_collate_fn = sampling_pool.collate_fn
        train_vocab = sampling_pool.vocab
        if cfg.selection.type == "random":
            limit_duration_sec = float(cfg.selection.limit_duration_sec)

            sampler = RandomSampler(
                sampling_pool=sampling_pool,
                limit_duration=limit_duration_sec,
            )
            train_dataset = sampler.sample()
        elif cfg.selection.type == "phone":
            with open("max_sim_sampled_dataset.pkl", "rb") as f:
                train_dataset = pickle.load(f)
        # show total duration
        total_duration = 0
        for i in range(len(train_dataset)):
            total_duration += train_dataset[i][2] / 16000
        logger.info(f"Total duration of sampled dataset: {total_duration / 60:.2f} min")

        # sampling_poolからサンプリングされたことを確認する（データリークの防止）
        sampled_train_id = train_dataset[0][0]
        assert sampling_pool[sampled_train_id][5] == train_dataset[0][5]

        train_dataloader = get_dataloader(
            train_dataset,
            cfg.train.num_batch,
            shuffle=True,
            drop_last=True,
            collate_fn=train_collate_fn,
        )

        test_dataset = LibriLightDataset(
            subset="1h",
            identifier_to_phones_file_path="phones/librispeech_normalized_phones_no_bcl.json",
            vocab_file_path="vocabs/libri-light_9h.json",
        )
        test_collate_fn = test_dataset.collate_fn
        test_vocab = test_dataset.vocab

        test_dataloader = get_dataloader(
            test_dataset,
            cfg.train.num_batch,
            shuffle=False,
            collate_fn=test_collate_fn,
        )

        assert train_vocab == test_vocab

        train_tf_idf = calculate_tf_idf_over_ds(train_dataset)
        test_tf_idf = calculate_tf_idf_over_ds(test_dataset)
        similarity = cos_sim(train_tf_idf, test_tf_idf)
        mlflow.log_metric("tf_idf_similarity", similarity)

        torch.cuda.empty_cache()
        num_label = len(sampling_pool.vocab.keys())
        model = Model(nlabel=num_label, cfg=cfg).to(DEVICE)
        ctc_loss = torch.nn.CTCLoss(reduction="sum", blank=sampling_pool.ctc_token_id)
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
        NUM_ACCUM_SEC = cfg.train.num_accum_sec

        min_train_wer = 1.0

        for epoch in range(NUM_EPOCH):
            logger.info(f"Epoch {epoch + 1}/{NUM_EPOCH}")
            model.train()
            train_epoch_loss = 0
            train_epoch_cer = 0
            train_epoch_wer = 0
            train_cnt = 0
            accum_sec = 0.0
            for i, batch in enumerate(train_dataloader):
                bx = batch[1].to(DEVICE)
                bx_len = batch[2].to(DEVICE)
                by = batch[3].to(DEVICE)
                by_len = batch[4].to(DEVICE)
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
                hypotheses = greedy_decoder(hypothesis, sampling_pool.vocab, "[PAD]", "|", "_")
                answers = greedy_decoder(by, sampling_pool.vocab, "[PAD]", "|", "_")
                train_epoch_cer += char_error_rate(hypotheses, answers)
                train_epoch_wer += word_error_rate(hypotheses, answers)

                train_cnt += 1

            mlflow.log_metric("train_loss", train_epoch_loss / train_cnt, step=epoch)
            mlflow.log_metric("train_cer", train_epoch_cer / train_cnt, step=epoch)
            mlflow.log_metric("train_wer", train_epoch_wer / train_cnt, step=epoch)

            min_train_wer = min(min_train_wer, train_epoch_wer / train_cnt)

            logger.info(f"Train Loss: {train_epoch_loss / train_cnt:.4f}")
            logger.info(f"Train CER: {train_epoch_cer / train_cnt:.4f}")
            logger.info(f"Train WER: {train_epoch_wer / train_cnt:.4f}")

            model.eval()
            test_epoch_loss = 0
            test_epoch_cer = 0
            test_epoch_wer = 0
            test_cnt = 0
            with torch.no_grad():
                for i, batch in enumerate(test_dataloader):
                    bx = batch[1].to(DEVICE)
                    bx_len = batch[2].to(DEVICE)
                    by = batch[3].to(DEVICE)
                    by_len = batch[4].to(DEVICE)
                    log_probs, y_lengths = model(bx, bx_len)
                    loss = ctc_loss(log_probs.transpose(1, 0), by, y_lengths, by_len)
                    test_epoch_loss += loss.item() / bx.size(0)

                    # calculate CER
                    hypothesis = torch.argmax(log_probs, dim=-1)
                    hypotheses = greedy_decoder(hypothesis, sampling_pool.vocab, "[PAD]", "|", "_")
                    answers = greedy_decoder(by, sampling_pool.vocab, "[PAD]", "|", "_")
                    test_epoch_cer += char_error_rate(hypotheses, answers)
                    test_epoch_wer += word_error_rate(hypotheses, answers)

                    test_cnt += 1

            mlflow.log_metric("test_loss", test_epoch_loss / test_cnt, step=epoch)
            mlflow.log_metric("test_cer", test_epoch_cer / test_cnt, step=epoch)
            mlflow.log_metric("test_wer", test_epoch_wer / test_cnt, step=epoch)

            logger.info(f"Test loss: {test_epoch_loss / test_cnt:.4f}")
            logger.info(f"Test CER: {test_epoch_cer / test_cnt:.4f}")
            logger.info(f"Test WER: {test_epoch_wer / test_cnt:.4f}")

        if min_train_wer > 0.8:
            raise Exception("Train WER is too high.")


if __name__ == "__main__":
    main()
