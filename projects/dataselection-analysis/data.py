import json
import os

import pandas as pd
import torch
import torchaudio
from torch.utils.data import Dataset
from transformers import Wav2Vec2CTCTokenizer, Wav2Vec2FeatureExtractor


class LibriAdaptUS(Dataset):
    def __init__(self, split: str):
        """
        split:
            matrix-train, matrix-test,
            nexus6-train, nexus6-test,
            pseye-train, pseye-test,
            respeaker-train, respeaker-test,
            shure-train, shure-test,
            usb-train, usb-test
        """
        self.mic = split.split('-')[0]
        self.type = split.split('-')[1]
        self.resampling_rate = 16000
        self.dataset_path_root = "../../datasets"
        self.metadata = pd.read_csv(
            os.path.join(self.dataset_path_root, f"libriadapt/en-us/clean/{self.type}_files_{self.mic}.csv")
        )
        self.wav_file_paths = self.metadata["wav_filename"].tolist()
        self.wav_file_names = [os.path.basename(wav_file_path) for wav_file_path in self.wav_file_paths]
        for i in range(len(self.wav_file_paths)):
            self.wav_file_paths[i] = os.path.join(self.dataset_path_root, self.wav_file_paths[i])
        self.transcripts = self.metadata["transcript"].tolist()

        assert len(self.wav_file_paths) == len(self.transcripts), (
            "wav_file_paths and transcripts should have the same length"
        )
        # only for normalize and resampling
        self.feature_extractor = Wav2Vec2FeatureExtractor.from_pretrained("facebook/wav2vec2-conformer-rel-pos-large")

        vocab_file_path = f"vocabs/vocab_{split}.json"
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
        return len(self.wav_file_paths)

    def __getitem__(self, idx):
        wav_file_path = self.wav_file_paths[idx]
        waveform, sample_rate = torchaudio.load(wav_file_path)
        waveform = waveform[0].flatten()
        # only for normalize and resampling cuz each wav file is not batched
        x = self.feature_extractor(waveform, sampling_rate=self.resampling_rate, return_tensors="pt").input_values[0]
        x_len = torch.tensor(x.shape[0])
        transcript = self.transcripts[idx]
        y = self.tokenizer(transcript, return_tensors="pt").input_ids[0]
        y_len = torch.tensor(y.shape[0])

        file_name = self.wav_file_names[idx]

        return file_name, x, x_len, y, y_len

    def collate_fn(self, batch):
        bfile_name, bx, bx_len, by, by_len = zip(*batch)
        bx = torch.nn.utils.rnn.pad_sequence(bx, batch_first=True, padding_value=0)
        by = torch.nn.utils.rnn.pad_sequence(by, batch_first=True, padding_value=self.pad_token_id)

        return bfile_name, bx, bx_len, by, by_len

    def extract_vocab(self, vocab_file_path: str) -> None:
        all_texts = self.transcripts
        all_texts_joined = " ".join(all_texts)
        vocab = list(set(all_texts_joined))

        vocab = {v: k for k, v in enumerate(vocab)}
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


class LibriAdaptIn(Dataset):
    def __init__(self, split: str):
        """
        split:
            matrix-train, matrix-test,
            nexus6-train, nexus6-test,
            pseye-train, pseye-test,
            respeaker-train, respeaker-test,
            shure-train, shure-test,
            usb-train, usb-test
        """
        self.mic = split.split('-')[0]
        self.type = split.split('-')[1]
        self.resampling_rate = 16000
        self.dataset_path_root = "../../datasets"
        self.metadata = pd.read_csv(
            os.path.join(self.dataset_path_root, f"libriadapt/en-in/clean/{self.type}_files_{self.mic}.csv")
        )
        self.wav_file_paths = self.metadata["wav_filename"].tolist()
        self.wav_file_names = [os.path.basename(wav_file_path) for wav_file_path in self.wav_file_paths]
        for i in range(len(self.wav_file_paths)):
            self.wav_file_paths[i] = os.path.join(self.dataset_path_root, self.wav_file_paths[i])
        self.transcripts = self.metadata["transcript"].tolist()

        assert len(self.wav_file_paths) == len(self.transcripts), (
            "wav_file_paths and transcripts should have the same length"
        )
        # only for normalize and resampling
        self.feature_extractor = Wav2Vec2FeatureExtractor.from_pretrained("facebook/wav2vec2-conformer-rel-pos-large")

        vocab_file_path = f"vocabs/vocab_{split}.json"
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
        return len(self.wav_file_paths)

    def __getitem__(self, idx):
        wav_file_path = self.wav_file_paths[idx]
        waveform, sample_rate = torchaudio.load(wav_file_path)
        waveform = waveform[0].flatten()
        # only for normalize and resampling cuz each wav file is not batched
        x = self.feature_extractor(waveform, sampling_rate=self.resampling_rate, return_tensors="pt").input_values[0]
        x_len = torch.tensor(x.shape[0])
        transcript = self.transcripts[idx]
        y = self.tokenizer(transcript, return_tensors="pt").input_ids[0]
        y_len = torch.tensor(y.shape[0])

        file_name = self.wav_file_names[idx]

        return file_name, x, x_len, y, y_len

    def collate_fn(self, batch):
        bfile_name, bx, bx_len, by, by_len = zip(*batch)
        bx = torch.nn.utils.rnn.pad_sequence(bx, batch_first=True, padding_value=0)
        by = torch.nn.utils.rnn.pad_sequence(by, batch_first=True, padding_value=self.pad_token_id)

        return bfile_name, bx, bx_len, by, by_len

    def extract_vocab(self, vocab_file_path: str) -> None:
        all_texts = self.transcripts
        all_texts_joined = " ".join(all_texts)
        vocab = list(set(all_texts_joined))

        vocab = {v: k for k, v in enumerate(vocab)}
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


class LibriAdaptGb(Dataset):
    def __init__(self, split: str):
        """
        split:
            matrix-train, matrix-test,
            nexus6-train, nexus6-test,
            pseye-train, pseye-test,
            respeaker-train, respeaker-test,
            shure-train, shure-test,
            usb-train, usb-test
        """
        self.mic = split.split('-')[0]
        self.type = split.split('-')[1]
        self.resampling_rate = 16000
        self.dataset_path_root = "../../datasets"
        self.metadata = pd.read_csv(
            os.path.join(self.dataset_path_root, f"libriadapt/en-gb/clean/{self.type}_files_{self.mic}.csv")
        )
        self.wav_file_paths = self.metadata["wav_filename"].tolist()
        self.wav_file_names = [os.path.basename(wav_file_path) for wav_file_path in self.wav_file_paths]
        for i in range(len(self.wav_file_paths)):
            self.wav_file_paths[i] = os.path.join(self.dataset_path_root, self.wav_file_paths[i])
        self.transcripts = self.metadata["transcript"].tolist()

        assert len(self.wav_file_paths) == len(self.transcripts), (
            "wav_file_paths and transcripts should have the same length"
        )
        # only for normalize and resampling
        self.feature_extractor = Wav2Vec2FeatureExtractor.from_pretrained("facebook/wav2vec2-conformer-rel-pos-large")

        vocab_file_path = f"vocabs/vocab_{split}.json"
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
        return len(self.wav_file_paths)

    def __getitem__(self, idx):
        wav_file_path = self.wav_file_paths[idx]
        waveform, sample_rate = torchaudio.load(wav_file_path)
        waveform = waveform[0].flatten()
        # only for normalize and resampling cuz each wav file is not batched
        x = self.feature_extractor(waveform, sampling_rate=self.resampling_rate, return_tensors="pt").input_values[0]
        x_len = torch.tensor(x.shape[0])
        transcript = self.transcripts[idx]
        y = self.tokenizer(transcript, return_tensors="pt").input_ids[0]
        y_len = torch.tensor(y.shape[0])

        file_name = self.wav_file_names[idx]

        return file_name, x, x_len, y, y_len

    def collate_fn(self, batch):
        bfile_name, bx, bx_len, by, by_len = zip(*batch)
        bx = torch.nn.utils.rnn.pad_sequence(bx, batch_first=True, padding_value=0)
        by = torch.nn.utils.rnn.pad_sequence(by, batch_first=True, padding_value=self.pad_token_id)

        return bfile_name, bx, bx_len, by, by_len

    def extract_vocab(self, vocab_file_path: str) -> None:
        all_texts = self.transcripts
        all_texts_joined = " ".join(all_texts)
        vocab = list(set(all_texts_joined))

        vocab = {v: k for k, v in enumerate(vocab)}
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


def get_dataloader(
    dataset: Dataset,
    batch_size: int = 8,
    num_workers: int = 4,
    shuffle: bool = True,
    drop_last: bool = True,
    pin_memory: bool = True,
) -> torch.utils.data.DataLoader:
    dataloader = torch.utils.data.DataLoader(
        dataset,
        batch_size=batch_size,
        num_workers=num_workers,
        shuffle=shuffle,
        pin_memory=pin_memory,
        collate_fn=dataset.collate_fn,
    )

    return dataloader
