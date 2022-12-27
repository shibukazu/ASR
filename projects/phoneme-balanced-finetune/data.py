import glob
import json
import os
from typing import Callable, List

import torch
import torchaudio
from transformers import Wav2Vec2CTCTokenizer, Wav2Vec2FeatureExtractor


class LibriLightBase(torch.utils.data.Dataset):
    _EXT_AUDIO = ".flac"
    _EXT_TRANSCRIPT = ".trans.txt"
    _SUBSET_MAP = {"10m": ["1h/0"], "1h": ["1h/*"], "10h": ["1h/*", "9h"]}

    class Example:
        def __init__(
            self,
            audio_file_path: str,
            transcript: str,
            sample_rate: int,
            speaker_id: int,
            chapter_id: int,
            utterance_id: int,
        ):
            self.audio_file_path = audio_file_path
            self.transcript = transcript
            self.sample_rate = sample_rate
            self.speaker_id = speaker_id
            self.chapter_id = chapter_id
            self.utterance_id = utterance_id

    def __init__(self, subset: str, root: str = "datasets/librispeech_finetuning"):
        self.root = root
        self.subset = subset
        if subset not in ["10m", "1h", "10h"]:
            raise ValueError(f"subset must be one of '10m', '1h', '10h', but got {subset}")
        self.folders = self._SUBSET_MAP[subset]
        self.examples = []

        # get all examples from specific subset
        for folder in self.folders:
            folder_path = os.path.join(root, folder)
            audio_file_paths = glob.glob(f"{folder_path}/*/*/*/*{self._EXT_AUDIO}")
            for audio_file_path in audio_file_paths:
                if audio_file_path.endswith(self._EXT_AUDIO):
                    audio_file_name = os.path.basename(audio_file_path)
                    audio_file_dir_path = os.path.dirname(audio_file_path)
                    speaker_id, chapter_id, utterance_id = audio_file_name[: -len(self._EXT_AUDIO)].split("-")
                    transcript_file_path = os.path.join(
                        audio_file_dir_path, f"{speaker_id}-{chapter_id}{self._EXT_TRANSCRIPT}"
                    )
                    with open(transcript_file_path, "r") as f:
                        for line in f:
                            if line.startswith(f"{speaker_id}-{chapter_id}-{utterance_id}"):
                                fileid_text, transcript = line.strip().split(" ", 1)
                                if fileid_text == f"{speaker_id}-{chapter_id}-{utterance_id}":
                                    break
                        else:
                            raise ValueError(f"transcript not found for {audio_file_path}")
                    self.examples.append(
                        self.Example(
                            audio_file_path=audio_file_path,
                            transcript=transcript,
                            sample_rate=16000,
                            speaker_id=int(speaker_id),
                            chapter_id=int(chapter_id),
                            utterance_id=int(utterance_id),
                        )
                    )


class LibriLightDataset(torch.utils.data.Dataset):
    def __init__(
        self,
        subset: str,
        root: str = "datasets/librispeech_finetuning",
        vocab_file_path: str = "vocabs/librilight.json",
    ):
        """
        subset: 10m, 1h, 10h
        """
        self.base = LibriLightBase(subset=subset, root=root)
        self.dataset = self.base.examples
        # only for normalization of input
        self.feature_extractor = Wav2Vec2FeatureExtractor.from_pretrained("facebook/wav2vec2-base")

        if not os.path.exists(vocab_file_path):
            # extract vocab from train dataset
            texts = []
            for example in self.dataset:
                texts.append(example.transcript)
            self.extract_vocab(texts, vocab_file_path)
        else:
            if not os.path.exists(vocab_file_path):
                raise ValueError(f"vocab file not found at {vocab_file_path}")

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
        audio, _ = torchaudio.load(example.audio_file_path)
        audio = audio.flatten()
        x = self.feature_extractor(audio, sampling_rate=example.sample_rate).input_values[0]
        x_len = len(x)
        y = self.tokenizer(example.transcript).input_ids
        y_len = len(y)

        transcript = example.transcript
        # audio file related info
        speaker_id = example.speaker_id
        chapter_id = example.chapter_id
        utterance_id = example.utterance_id

        return (
            idx,
            torch.tensor(x),
            torch.tensor(x_len),
            torch.tensor(y),
            torch.tensor(y_len),
            transcript,
            speaker_id,
            chapter_id,
            utterance_id,
        )

    def collate_fn(self, batch):
        bidx, bx, bx_len, by, by_len, btranscript, bspeaker_id, bchapter_id, butterance_id = zip(*batch)
        bx = torch.nn.utils.rnn.pad_sequence(bx, batch_first=True, padding_value=0)
        bx_len = torch.tensor(bx_len)
        by = torch.nn.utils.rnn.pad_sequence(by, batch_first=True, padding_value=self.pad_token_id)
        by_len = torch.tensor(by_len)

        return (
            bidx,
            bx,
            bx_len,
            by,
            by_len,
            btranscript,
            bspeaker_id,
            bchapter_id,
            butterance_id,
        )

    def extract_vocab(self, all_text: List, vocab_file_path: str) -> None:

        all_text = " ".join(all_text)
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


class LibriSpeechDataset:
    def __init__(
        self,
        split: str,
        root: str = "./datasets/librispeech",
        vocab_file_path: str = "vocabs/librispeech.json",
    ):
        """
        split:
        train, dev, test
        """
        if split == "train":
            dataset_train_clean_100 = torchaudio.datasets.LIBRISPEECH(root, url="train-clean-100")
            self.dataset = dataset_train_clean_100
        elif split == "dev":
            dataset_dev_clean = torchaudio.datasets.LIBRISPEECH(root, url="dev-clean")
            self.dataset = dataset_dev_clean
        elif split == "test":
            dataset_test_clean = torchaudio.datasets.LIBRISPEECH(root, url="test-clean")
            self.dataset = dataset_test_clean
        else:
            raise ValueError("invalid split")

        # trainかつvocabがない場合は作成
        if not os.path.exists(vocab_file_path):
            self.extract_vocab(vocab_file_path=vocab_file_path)
        else:
            if not os.path.exists(vocab_file_path):
                raise ValueError("vocab file not found")
        # only for normalization of input on quantization
        self.feature_extractor = Wav2Vec2FeatureExtractor.from_pretrained("facebook/wav2vec2-base")
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
        x = self.feature_extractor(audio, sampling_rate=example[1]).input_values[0]
        x_len = len(x)
        y = self.tokenizer(example[2]).input_ids
        y_len = len(y)

        transcript = example[2]

        return (
            idx,
            torch.tensor(x),
            torch.tensor(x_len),
            torch.tensor(y),
            torch.tensor(y_len),
            transcript,
        )

    def collate_fn(self, batch):
        bidx, bx, bx_len, by, by_len, btranscript = zip(*batch)

        bx = torch.nn.utils.rnn.pad_sequence(bx, batch_first=True, padding_value=0)
        bx_len = torch.tensor(bx_len)
        by = torch.nn.utils.rnn.pad_sequence(by, batch_first=True, padding_value=self.pad_token_id)
        by_len = torch.tensor(by_len)

        return bidx, bx, bx_len, by, by_len, btranscript

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


def get_dataloader(
    dataset: torch.utils.data.Dataset,
    batch_size: int = 8,
    num_workers: int = 4,
    shuffle: bool = True,
    drop_last: bool = True,
    pin_memory: bool = True,
    collate_fn: Callable = None,
) -> torch.utils.data.DataLoader:
    dataloader = torch.utils.data.DataLoader(
        dataset,
        batch_size=batch_size,
        num_workers=num_workers,
        shuffle=shuffle,
        drop_last=drop_last,
        pin_memory=pin_memory,
        collate_fn=collate_fn,
    )

    return dataloader
