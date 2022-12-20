import json
import os
from typing import List

import torch
import torchaudio
from transformers import AutoFeatureExtractor, Wav2Vec2CTCTokenizer

import datasets


class TIMITDataset(torch.utils.data.Dataset):
    def __init__(self, vocab_file_path: str, split: str, resampling_rate: int = 16000):
        """
        split: train, test
        """
        self.resampling_rate = resampling_rate
        # dataset実体はhuggingfaceのdatasetsで取得
        self.dataset = datasets.load_dataset(
            "../../datasets/loading_scripts/timit.py",
            data_dir="../../datasets/TIMIT/",
            cache_dir="/home/shibutani/fs/.cache/huggingface/datasets")
        self.dataset = self.dataset[split]
        self.feature_extractor = AutoFeatureExtractor.from_pretrained("facebook/wav2vec2-conformer-rel-pos-large")
        self.extract_vocab(all_text=self.dataset["text"], vocab_file_path=vocab_file_path)
        self.tokenizer = Wav2Vec2CTCTokenizer(
            vocab_file_path, unk_token="[UNK]", pad_token="[PAD]", word_delimiter_token="|"
        )

        def preprocess_function(example):
            audio = example["audio"]["array"].flatten()
            x = self.feature_extractor(audio, sampling_rate=self.resampling_rate).input_values[0]
            x_len = len(x)
            y = self.tokenizer(example["text"]).input_ids
            y_len = len(y)
            example["x"] = x
            example["x_len"] = x_len
            example["y"] = y
            example["y_len"] = y_len

            return example

        self.dataset = self.dataset.map(preprocess_function, num_proc=4)

        with open(vocab_file_path, "r") as vocab_file:
            vocab = json.load(vocab_file)
        self.vocab = vocab
        self.pad_token_id = vocab["[PAD]"]
        self.unk_token_id = vocab["[UNK]"]
        self.ctc_token_id = vocab["_"]

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        example = self.dataset[idx]

        return (
            idx,
            torch.tensor(example["x"]),
            torch.tensor(example["x_len"]),
            torch.tensor(example["y"]),
            torch.tensor(example["y_len"]),
        )

    def collate_fn(self, batch):
        bidx, bx, bx_len, by, by_len = zip(*batch)

        bx = torch.nn.utils.rnn.pad_sequence(bx, batch_first=True, padding_value=0)
        bx_len = torch.tensor(bx_len)
        by = torch.nn.utils.rnn.pad_sequence(by, batch_first=True, padding_value=0)
        by_len = torch.tensor(by_len)

        return bidx, bx, bx_len, by, by_len

    def extract_vocab(self, all_text: List = None, vocab_file_path: str = "./full_vocab.json") -> None:

        all_text = " ".join(all_text)
        vocab_list = list(set(all_text))

        vocab = {v: k for k, v in enumerate(vocab_list)}
        # use | as delimeter in stead of " "
        vocab["|"] = vocab[" "]
        # dekete unused char
        del vocab[" "]
        # add unk and pad token
        vocab["[UNK]"] = len(vocab)
        vocab["[PAD]"] = len(vocab)
        vocab["_"] = len(vocab)

        with open(vocab_file_path, "w") as vocab_file:
            json.dump(vocab, vocab_file)


class LibriSpeechDataset(torch.utils.data.Dataset):
    def __init__(self, root: str, split: str, resampling_rate: int = 16000):
        """
        split:
        train, dev, test, dev-test
        """
        if split == "train":
            dataset_train_clean_100 = torchaudio.datasets.LIBRISPEECH(
                os.path.join(root, "librispeech"), url="train-clean-100")
            dataset_train_clean_360 = torchaudio.datasets.LIBRISPEECH(
                os.path.join(root, "librispeech"), url="train-clean-360")
            dataset_train_other_500 = torchaudio.datasets.LIBRISPEECH(
                os.path.join(root, "librispeech"), url="train-other-500")
            self.dataset = dataset_train_clean_100 + dataset_train_clean_360 + dataset_train_other_500
        elif split == "dev":
            dataset_dev_clean = torchaudio.datasets.LIBRISPEECH(os.path.join(root, "librispeech"), url="dev-clean")
            dataset_dev_other = torchaudio.datasets.LIBRISPEECH(os.path.join(root, "librispeech"), url="dev-other")
            self.dataset = dataset_dev_clean + dataset_dev_other
        elif split == "test":
            dataset_test_clean = torchaudio.datasets.LIBRISPEECH(os.path.join(root, "librispeech"), url="test-clean")
            dataset_test_other = torchaudio.datasets.LIBRISPEECH(os.path.join(root, "librispeech"), url="test-other")
            self.dataset = dataset_test_clean + dataset_test_other
        elif split == "dev-test":
            dataset_dev_clean = torchaudio.datasets.LIBRISPEECH(os.path.join(root, "librispeech"), url="dev-clean")
            dataset_dev_other = torchaudio.datasets.LIBRISPEECH(os.path.join(root, "librispeech"), url="dev-other")
            dataset_test_clean = torchaudio.datasets.LIBRISPEECH(os.path.join(root, "librispeech"), url="test-clean")
            dataset_test_other = torchaudio.datasets.LIBRISPEECH(os.path.join(root, "librispeech"), url="test-other")
            self.dataset = dataset_dev_clean + dataset_dev_other + dataset_test_clean + dataset_test_other
        else:
            raise ValueError("invalid split")
        self.resampling_rate = resampling_rate
        # only for normalizing
        self.feature_extractor = AutoFeatureExtractor.from_pretrained("facebook/wav2vec2-conformer-rel-pos-large")

        # trainかつvocabがない場合は作成
        vocab_file_path = "librispeech.json"
        if split == "train":
            if not os.path.exists(vocab_file_path):
                self.extract_vocab(vocab_file_path=vocab_file_path)

        self.tokenizer = Wav2Vec2CTCTokenizer(
            vocab_file_path, unk_token="[UNK]", pad_token="[PAD]", word_delimiter_token="|"
        )

        with open(vocab_file_path, "r") as vocab_file:
            vocab = json.load(vocab_file)
        self.vocab = vocab
        self.pad_token_id = vocab["[PAD]"]
        self.unk_token_id = vocab["[UNK]"]
        self.ctc_token_id = vocab["_"]

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        example = self.dataset[idx]
        audio = example[0].flatten()
        x = torch.tensor(self.feature_extractor(audio, sampling_rate=self.resampling_rate).input_values[0])
        x_len = torch.tensor(len(x))
        y = torch.tensor(self.tokenizer(example[2]).input_ids)
        y_len = torch.tensor(len(y))

        return (
            idx,
            x,
            x_len,
            y,
            y_len,
        )

    def collate_fn(self, batch):
        bidx, bx, bx_len, by, by_len = zip(*batch)

        bx = torch.nn.utils.rnn.pad_sequence(bx, batch_first=True, padding_value=0)
        bx_len = torch.tensor(bx_len)
        by = torch.nn.utils.rnn.pad_sequence(by, batch_first=True, padding_value=0)
        by_len = torch.tensor(by_len)

        return bidx, bx, bx_len, by, by_len

    def extract_vocab(self, vocab_file_path: str) -> None:
        print("create vocab...")
        all_texts = [example[2] for example in self.dataset]
        all_text = " ".join(all_texts)
        vocab_list = list(set(all_text))

        vocab = {v: k for k, v in enumerate(vocab_list)}
        # use | as delimeter in stead of " "
        vocab["|"] = vocab[" "]
        # delete unused char
        del vocab[" "]
        # add unk and pad token
        vocab["[UNK]"] = len(vocab)
        vocab["[PAD]"] = len(vocab)
        vocab["_"] = len(vocab)

        with open(vocab_file_path, "w") as vocab_file:
            json.dump(vocab, vocab_file)


class TEDLIUM2Dataset(torch.utils.data.Dataset):
    def __init__(self, root: str, split: str, resampling_rate: int = 16000):
        """
        split:
        train, dev, test
        """
        if split == "train":
            self.dataset = torchaudio.datasets.TEDLIUM(
                root, release="release2", subset="train")
        elif split == "dev":
            self.dataset = torchaudio.datasets.TEDLIUM(
                root, release="release2", subset="dev")
        elif split == "test":
            self.dataset = torchaudio.datasets.TEDLIUM(
                root, release="release2", subset="test")
        else:
            raise ValueError("invalid split")
        self.resampling_rate = resampling_rate
        self.feature_extractor = AutoFeatureExtractor.from_pretrained("facebook/wav2vec2-conformer-rel-pos-large")

        # trainかつvocabがない場合は作成
        vocab_file_path = "tedlium2.json"
        if split == "train":
            if not os.path.exists(vocab_file_path):
                self.extract_vocab(vocab_file_path=vocab_file_path)

        self.tokenizer = Wav2Vec2CTCTokenizer(
            vocab_file_path, unk_token="[UNK]", pad_token="[PAD]", word_delimiter_token="|"
        )

        with open(vocab_file_path, "r") as vocab_file:
            vocab = json.load(vocab_file)
        self.vocab = vocab
        self.pad_token_id = vocab["[PAD]"]
        self.unk_token_id = vocab["[UNK]"]
        self.ctc_token_id = vocab["_"]

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        example = self.dataset[idx]
        audio = example[0].flatten()
        x = torch.tensor(self.feature_extractor(audio, sampling_rate=self.resampling_rate).input_values[0])
        x_len = torch.tensor(len(x))
        y = torch.tensor(self.tokenizer(example[2]).input_ids)
        y_len = torch.tensor(len(y))

        return (
            idx,
            x,
            x_len,
            y,
            y_len,
        )

    def collate_fn(self, batch):
        bidx, bx, bx_len, by, by_len = zip(*batch)

        bx = torch.nn.utils.rnn.pad_sequence(bx, batch_first=True, padding_value=0)
        bx_len = torch.tensor(bx_len)
        by = torch.nn.utils.rnn.pad_sequence(by, batch_first=True, padding_value=0)
        by_len = torch.tensor(by_len)

        return bidx, bx, bx_len, by, by_len

    def extract_vocab(self, vocab_file_path: str) -> None:
        print("create vocab...")
        all_texts = [example[2] for example in self.dataset]
        all_text = " ".join(all_texts)
        vocab_list = list(set(all_text))

        vocab = {v: k for k, v in enumerate(vocab_list)}
        # use | as delimeter in stead of " "
        vocab["|"] = vocab[" "]
        # delete unused char
        del vocab[" "]
        # add unk and pad token
        vocab["[UNK]"] = len(vocab)
        vocab["[PAD]"] = len(vocab)
        vocab["_"] = len(vocab)

        with open(vocab_file_path, "w") as vocab_file:
            json.dump(vocab, vocab_file)
