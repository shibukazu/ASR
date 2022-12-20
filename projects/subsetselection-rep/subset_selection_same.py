import copy
import glob
import json
import math
import os
import random
import re
import sys

sys.path.append("../..")
import time
from collections import defaultdict
from logging import config, getLogger

import hydra
import matplotlib.pyplot as plt
import mlflow
import numpy as np
import torch
from conf import logging_conf
from my_model import Model
from omegaconf import DictConfig
from rich.logging import RichHandler
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import DataLoader, Dataset
from torchaudio.functional import resample
from torchmetrics.functional import char_error_rate, word_error_rate
from transformers import Wav2Vec2CTCTokenizer
from util.mlflow import log_params_from_omegaconf_dict

from datasets import load_dataset
from my_modules.decoders import ctc
from my_modules.transformers.scheduler import TransformerLR


def save_wer_and_plot(scores, method: str):
    """
    save WER hist for each selection method
    """
    wers = [score[1] for score in scores]
    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.hist(wers, bins=50, range=(0.0, 1.0))
    ax.set_title(f"WER on {method}")
    ax.set_xlabel("WER")
    mlflow.log_figure(fig, f"selected_wer_{method}.png")


def cowerage_sampler(dataset, scores, retain, input_to_text, other_size):
    asr_dataset = copy.deepcopy(dataset)
    sorted_scores = sorted(scores, key=lambda x: x[1], reverse=True)
    highest_wer = sorted_scores[0][1]
    lowest_wer = sorted_scores[-1][1]
    num_buckets = int(len(sorted_scores) / 10)

    buckets = []
    for i in range(num_buckets):
        bucket = []
        # WERの範囲をnum_buckets等分する
        # 等分であるため、一つもデータが含まれないbucketも存在する
        lower_limit = lowest_wer + ((i - 1) * (highest_wer - lowest_wer) / num_buckets)
        upper_limit = lowest_wer + ((i) * (highest_wer - lowest_wer) / num_buckets)
        for score in sorted_scores:
            if score[1] >= lower_limit and score[1] < upper_limit:
                bucket.append(score)
        buckets.append(bucket)
    selected_scores_tmp = []
    counter = 0
    for bucket in buckets:
        sampled = random.sample(bucket, math.ceil(retain * len(bucket)))
        selected_scores_tmp.append(sampled)
        counter += len(sampled)
    # 他のサンプリング方法以下にする
    while counter > other_size:
        for selected_score_tmp in selected_scores_tmp:
            if counter == other_size:
                break
            else:
                if len(selected_score_tmp) > 0:
                    selected_score_tmp.pop()
                    counter -= 1

    def checker(example, idx):
        # train時とidxが共通しているかチェック
        assert (
            example["text"] == input_to_text[idx]
        ), f"data mismatch idx:{idx}, trained: {input_to_text[idx]}, selection: {example['text']}"

    asr_dataset["train"].map(
        checker,
        with_indices=True,
        num_proc=16,
    )
    selected_scores = []
    for selected_score_tmp in selected_scores_tmp:
        selected_scores.extend(selected_score_tmp)

    save_wer_and_plot(selected_scores, "cowerage")

    selected_idxs = [selected_score[0] for selected_score in selected_scores]
    asr_dataset["train"] = asr_dataset["train"].filter(
        lambda example, index: index in selected_idxs,
        with_indices=True,
        num_proc=16,
    )
    return asr_dataset


def random_sampler(dataset, scores, retain, input_to_text):
    asr_dataset = copy.deepcopy(dataset)

    def checker(example, idx):
        assert (
            example["text"] == input_to_text[idx]
        ), f"data mismatch idx:{idx}, trained: {input_to_text[idx]}, selection: {example['text']}"

    asr_dataset["train"].map(
        checker,
        with_indices=True,
        num_proc=16,
    )

    select_size = int(retain * len(asr_dataset["train"]))
    selected_scores = random.sample(scores, select_size)

    save_wer_and_plot(selected_scores, "random")

    selected_idxs = [selected_score[0] for selected_score in selected_scores]
    asr_dataset["train"] = asr_dataset["train"].filter(
        lambda example, index: index in selected_idxs,
        with_indices=True,
        num_proc=16,
    )
    return asr_dataset


def top_k_sampler(dataset, scores, retain, input_to_text):
    asr_dataset = copy.deepcopy(dataset)
    sorted_scores = sorted(scores, key=lambda x: x[1], reverse=True)
    boundary = int(len(sorted_scores) * retain)

    def checker(example, idx):
        assert (
            example["text"] == input_to_text[idx]
        ), f"data mismatch idx:{idx}, trained: {input_to_text[idx]}, selection: {example['text']}"

    asr_dataset["train"].map(
        checker,
        with_indices=True,
        num_proc=16,
    )
    selected_scores = sorted_scores[0:boundary]

    save_wer_and_plot(selected_scores, "top_k")

    selected_idxs = [selected_score[0] for selected_score in selected_scores]
    asr_dataset["train"] = asr_dataset["train"].filter(
        lambda example, index: index in selected_idxs,
        with_indices=True,
        num_proc=16,
    )
    return asr_dataset


def bottom_k_sampler(dataset, scores, retain, input_to_text):
    asr_dataset = copy.deepcopy(dataset)
    sorted_scores = sorted(scores, key=lambda x: x[1], reverse=False)
    boundary = int(len(sorted_scores) * retain)

    def checker(example, idx):
        assert (
            example["text"] == input_to_text[idx]
        ), f"data mismatch idx:{idx}, trained: {input_to_text[idx]}, selection: {example['text']}"

    asr_dataset["train"].map(
        checker,
        with_indices=True,
        num_proc=16,
    )
    selected_scores = sorted_scores[0:boundary]

    save_wer_and_plot(selected_scores, "bottom_k")

    selected_idxs = [selected_score[0] for selected_score in selected_scores]
    asr_dataset["train"] = asr_dataset["train"].filter(
        lambda example, index: index in selected_idxs,
        with_indices=True,
        num_proc=16,
    )
    return asr_dataset


class TIMITDatasetWav(Dataset):
    """
    TIMIT dataset class for Wav2Vec2.0 encoder
    args:
        dataset: base huggingface dataset sampled by some methoc
    """

    def __init__(self, dataset, vocab_file_path: str, resample_rate: int = 16000, is_train: bool = False):

        self.type = "train" if is_train else "test"

        self.resample_rate = resample_rate

        dataset = dataset.remove_columns(["id"])

        self.tokenizer = Wav2Vec2CTCTokenizer(
            vocab_file_path, unk_token="[UNK]", pad_token="[PAD]", word_delimiter_token="|"
        )

        def prepare_dataset(example):
            audio = example["audio"]["array"].flatten()
            orig_freq = example["audio"]["sampling_rate"]
            x = resample(audio, orig_freq=orig_freq, new_freq=self.resample_rate)
            mean = np.mean(x)
            std = np.std(x)
            z = (x - mean) / std
            assert z.shape == x.shape, "標準化後の形が異なります。"
            assert z.ndim == 1, "標準化後の次元が不正です。"
            assert abs(np.mean(z) - 0) < 1, "標準化後の平均値が不正です。"
            assert abs(np.std(z) - 1) < 1, "標準化後の標準偏差が不正です。"
            example["input_values"] = z
            example["input_length"] = len(example["input_values"])
            example["labels"] = self.tokenizer(example["text"]).input_ids
            return example

        self.dataset = dataset.map(prepare_dataset, num_proc=4)

        with open(vocab_file_path, "r") as vocab_file:
            vocab = json.load(vocab_file)
        self.vocab = vocab
        self.pad_token_id = vocab["[PAD]"]
        self.unk_token_id = vocab["[UNK]"]
        self.ctc_token_id = vocab["_"]

    def __len__(self):
        return len(self.dataset[self.type])

    def __getitem__(self, idx):

        return (
            idx,
            torch.tensor(self.dataset[self.type][idx]["input_values"]),
            torch.tensor(self.dataset[self.type][idx]["labels"]),
        )

    def collate_fn(self, batch):
        idxs, wavs, text_idxs = zip(*batch)
        original_wav_lens = torch.tensor(np.array([len(wav) for wav in wavs]))
        original_text_idx_lens = torch.tensor(np.array([len(text_idx) for text_idx in text_idxs]))
        # padding for spectrogram_db
        padded_wavs = []
        for wav in wavs:
            padded_wav = np.pad(wav, ((0, max(original_wav_lens) - wav.shape[0])), "constant", constant_values=0)
            padded_wavs.append(padded_wav)

        padded_wavs = torch.tensor(np.array(padded_wavs))

        # padding and packing for text_idx
        padded_text_idxs = pad_sequence(text_idxs, batch_first=True, padding_value=self.pad_token_id)

        return idxs, padded_wavs, padded_text_idxs, original_wav_lens, original_text_idx_lens


CONF_NAME = "selection_1"


@hydra.main(version_base=None, config_path="conf", config_name=CONF_NAME)
def main(cfg: DictConfig):
    SELECTION = cfg.selection.method
    ARTIFACT_LOCATION = "/home/shibutani/fs/ASR/projects/subsetselection-rep/artifacts"
    EXPERIMENT_NAME = f"{CONF_NAME}"

    experiment = mlflow.get_experiment_by_name(EXPERIMENT_NAME)
    if experiment is None:
        experiment_id = mlflow.create_experiment(name=EXPERIMENT_NAME, artifact_location=ARTIFACT_LOCATION)
    else:
        experiment_id = experiment.experiment_id

    with mlflow.start_run(experiment_id=experiment_id) as mlflow_run:
        LOG_DIR = f"./logs/{EXPERIMENT_NAME}/{mlflow_run.info.run_id}"
        config.dictConfig(logging_conf.config_generator(LOG_DIR))
        logger = getLogger()
        logger.handlers[0] = RichHandler(markup=True)  # pretty formatting
        # save parameters from hydra to mlflow
        log_params_from_omegaconf_dict(cfg)
        average_wers = defaultdict(list)
        scores = []
        WER_CALCULATION_EPOCH = cfg.selection.wer_calc_epoch
        TOTAL_RUN = cfg.selection.wer_calc_total_run
        # 事前に行ったFineTuningにおけるWERからサブセットセレクションに用いる平均WERを計算する
        for run in range(2, TOTAL_RUN):
            for p in glob.glob("cpts/*.pt"):
                if re.search(f"cpts/timit_finetune_checkpoint_{run}_{WER_CALCULATION_EPOCH-1}", p):
                    cpt = torch.load(p)
                    wers = cpt["input_to_wer"]
                    input_to_text = cpt["input_to_text"]
                    for idx, wer in wers.items():
                        average_wers[idx].append(wer.item())
                    logger.info(f"{p} was used.")

        for idx, v in average_wers.items():
            scores.append((idx, np.mean(v)))

        base_dataset = load_dataset(
            "../../datasets/loading_scripts/timit.py",
            data_dir="../../datasets/TIMIT/",
            cache_dir="/home/shibutani/fs/.cache/huggingface/datasets",
        )
        logger.info(f"base_dataset size: {len(base_dataset['train'])}")
        RETAIN = cfg.selection.retain
        # create huggingface dataset
        if SELECTION == "random":
            logger.info("Selection method 'random' was selected.")
            selected_dataset = random_sampler(base_dataset, scores, RETAIN, input_to_text)
        elif SELECTION == "top_k":
            logger.info("Selection method 'top_k' was selected.")
            selected_dataset = top_k_sampler(base_dataset, scores, RETAIN, input_to_text)
        elif SELECTION == "bottom_k":
            logger.info("Selection method 'bottom_k' was selected.")
            selected_dataset = bottom_k_sampler(base_dataset, scores, RETAIN, input_to_text)
        elif SELECTION == "cowerage":
            logger.info("Selection method 'cowerage' was selected.")
            # To adjust the sampled size, we use dataset sampled by bottom_k
            bottom_k_selected_dataset = bottom_k_sampler(base_dataset, scores, RETAIN, input_to_text)
            selected_dataset = cowerage_sampler(
                base_dataset, scores, RETAIN, input_to_text, other_size=len(bottom_k_selected_dataset["train"])
            )
        else:
            raise (NotImplementedError)

        with open("timit_full_vocab.json") as f:
            full_vocab = json.load(f)
        logger.info(f"full dataset size: {len(base_dataset['train'])}")
        # create pytorch dataset based on sampled huggingface dataset
        train_dataset = TIMITDatasetWav(selected_dataset, "timit_full_vocab.json", is_train=True)
        logger.info(f"train dataset size: {len(train_dataset)}")
        # vocab check
        logger.info(f"train vocab size: {len(train_dataset.vocab)}")
        for key in train_dataset.vocab.keys():
            _ = full_vocab[key]

        test_dataset = TIMITDatasetWav(selected_dataset, "timit_full_vocab.json", is_train=False)
        logger.info(f"test dataset size: {len(test_dataset)}")
        logger.info(f"test vocab size: {len(test_dataset.vocab)}")
        for key in test_dataset.vocab.keys():
            _ = full_vocab[key]

        BATCH_SIZE = cfg.train.batch
        train_dataloader = DataLoader(
            train_dataset,
            batch_size=BATCH_SIZE,
            shuffle=True,
            num_workers=8,
            # 不完全なバッチの無視
            drop_last=True,
            # 高速化?
            pin_memory=True,
            collate_fn=train_dataset.collate_fn,
        )
        test_dataloader = DataLoader(
            test_dataset,
            batch_size=BATCH_SIZE,
            shuffle=True,
            num_workers=8,
            # 不完全なバッチの無視
            drop_last=True,
            # 高速化?
            pin_memory=True,
            collate_fn=test_dataset.collate_fn,
        )

        device = torch.device(f"cuda:{cfg.train.cuda}" if torch.cuda.is_available() else "cpu")
        logger.info(f"This training will be running on {device}.")

        NUM_LABELS = len(train_dataloader.dataset.vocab)
        NUM_EPOCH = cfg.train.epoch

        model = Model(NUM_LABELS).to(device)

        ctc_loss = torch.nn.CTCLoss(reduction="sum", blank=train_dataloader.dataset.ctc_token_id)
        optimizer = torch.optim.Adam(model.parameters(), lr=cfg.train.lr, weight_decay=cfg.train.weight_decay)
        scheduler = TransformerLR(optimizer, warmup_epochs=cfg.train.warmup_epochs)

        logger.info("Model has been initialized.")
        for i in range(NUM_EPOCH):
            logger.info(f"{i} th epoch")
            t0 = time.time()
            model.train()
            epoch_loss = 0
            cnt = 0
            for _, (idxs, padded_wavs, padded_text_idxs, original_wav_lens, original_text_idx_lens) in enumerate(
                train_dataloader
            ):
                cnt += 1
                padded_wavs = padded_wavs.to(device)
                original_wav_lens = original_wav_lens.to(device)
                padded_text_idxs = padded_text_idxs.to(device)
                original_text_idx_lens = original_text_idx_lens.to(device)

                optimizer.zero_grad()

                log_probs, y_lengths = model(x=padded_wavs, x_lengths=original_wav_lens)

                loss = ctc_loss(log_probs.transpose(1, 0), padded_text_idxs, y_lengths, original_text_idx_lens)
                loss.backward()
                optimizer.step()
                # lossはバッチ内平均ロス
                epoch_loss += loss.item() / BATCH_SIZE
            logger.info(f"train loss: {epoch_loss / cnt}")
            mlflow.log_metric(key="Train Loss", value=epoch_loss / cnt, step=i)
            scheduler.step()
            # バッチ内平均ロスの和をイテレーション数で割ることで、一つのデータあたりの平均ロスを求める

            model.eval()
            with torch.no_grad():
                epoch_test_loss = 0
                cnt = 0
                total_cer = 0
                total_wer = 0
                for _, (
                    idxs,
                    padded_wavs,
                    padded_text_idxs,
                    original_wav_lens,
                    original_text_idx_lens,
                ) in enumerate(test_dataloader):
                    cnt += 1
                    padded_wavs = padded_wavs.to(device)
                    original_wav_lens = original_wav_lens.to(device)
                    padded_text_idxs = padded_text_idxs.to(device)
                    original_text_idx_lens = original_text_idx_lens.to(device)

                    log_probs, y_lengths = model(x=padded_wavs, x_lengths=original_wav_lens)
                    loss = ctc_loss(log_probs.transpose(1, 0), padded_text_idxs, y_lengths, original_text_idx_lens)
                    epoch_test_loss += loss.item() / BATCH_SIZE
                    # for CER calculation
                    hypotheses_idxs = log_probs.argmax(dim=2)
                    hypotheses = ctc.simple_decode(
                        hypotheses_idxs, train_dataloader.dataset.vocab, padding="[PAD]", separator="|", blank="_"
                    )
                    teachers = ctc.simple_decode(
                        padded_text_idxs, train_dataloader.dataset.vocab, padding="[PAD]", separator="|", blank="_"
                    )
                    total_cer += char_error_rate(hypotheses, teachers)
                    total_wer += word_error_rate(hypotheses, teachers)

            t1 = time.time()
            logger.info(
                f"test loss: {epoch_test_loss / cnt}\n"
                f"CER: {total_cer / cnt}\n"
                f"WER: {total_wer / cnt}\n"
                f"Time: {t1 - t0} sec\n"
            )
            mlflow.log_metric(key="Test Loss", value=epoch_test_loss / cnt, step=i)
            mlflow.log_metric(key="CER", value=total_cer / cnt, step=i)
            mlflow.log_metric(key="WER", value=total_wer / cnt, step=i)

            if (i + 1) % 10 == 0:
                checkpoint = {
                    "model": model.state_dict(),
                    "optimizer": optimizer.state_dict(),
                    "scheduler": scheduler.state_dict(),
                    "random": random.getstate(),
                    "np_random": np.random.get_state(),  # numpy.randomを使用する場合は必要
                    "torch": torch.get_rng_state(),
                    "torch_random": torch.random.get_rng_state(),
                    "cuda_random": torch.cuda.get_rng_state(),  # gpuを使用する場合は必要
                    "cuda_random_all": torch.cuda.get_rng_state_all(),  # 複数gpuを使用する場合は必要
                }
                CPT_SAVE_DIR = f"cpts/{EXPERIMENT_NAME}/{mlflow_run.info.run_id}"
                CPT_FILE_NAME = f"{CPT_SAVE_DIR}/{i}.pt"
                os.makedirs(CPT_SAVE_DIR, exist_ok=True)
                torch.save(
                    checkpoint,
                    CPT_FILE_NAME,
                )
                mlflow.log_artifact(CPT_FILE_NAME)
    mlflow.log_artifact(LOG_DIR)


if __name__ == "__main__":
    main()
