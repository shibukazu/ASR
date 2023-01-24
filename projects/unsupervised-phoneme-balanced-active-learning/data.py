import glob
import json
import os
from typing import Callable, List

import torch
import torchaudio
from transformers import Wav2Vec2CTCTokenizer, Wav2Vec2FeatureExtractor

import datasets


class TIMITDatasetWav(torch.utils.data.Dataset):
    def __init__(self, split: str, resampling_rate: int = 16000):
        """
        split: train, test
        """
        self.resampling_rate = resampling_rate
        # dataset実体はhuggingfaceのdatasetsで取得
        self.dataset = datasets.load_dataset(
            "../../datasets/loading_scripts/timit.py",
            data_dir="../../datasets/TIMIT/",
            cache_dir="/home/shibutani/fs/.cache/huggingface/datasets",
        )
        self.dataset = self.dataset[split]
        # only for normalization of input
        self.feature_extractor = Wav2Vec2FeatureExtractor.from_pretrained("facebook/wav2vec2-large-lv60")
        vocab_file_path = "vocabs/timit.json"
        if split == "train" and not os.path.exists(vocab_file_path):
            # extract vocab from train dataset
            self.extract_vocab(self.dataset["text"], vocab_file_path)
        assert os.path.exists(vocab_file_path), "vocab file not found"
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

        self.phoneme_converter = {
            "iy": "iy",
            "ih": "ih",
            "eh": "eh",
            "ae": "ae",
            "ix": "ih",
            "ax": "ah",
            "ah": "ah",
            "uw": "uw",
            "ux": "uw",
            "uh": "uh",
            "ao": "aa",
            "aa": "aa",
            "ey": "ey",
            "ay": "ay",
            "oy": "oy",
            "aw": "aw",
            "ow": "ow",
            "l": "el",
            "el": "el",
            "r": "r",
            "y": "y",
            "w": "w",
            "er": "axr",
            "axr": "axr",
            "m": "em",
            "em": "em",
            "n": "en",
            "nx": "en",
            "en": "en",
            "ng": "eng",
            "eng": "eng",
            "ch": "ch",
            "jh": "jh",
            "dh": "dh",
            "b": "b",
            "d": "d",
            "dx": "dx",
            "g": "g",
            "p": "p",
            "t": "t",
            "k": "k",
            "z": "z",
            "zh": "sh",
            "v": "v",
            "f": "f",
            "th": "th",
            "s": "s",
            "sh": "sh",
            "hh": "hh",
            "hv": "hh",
            "cl": "bcl",
            "pcl": "bcl",
            "tcl": "bcl",
            "kcl": "bcl",
            "qcl": "bcl",
            "vcl": "bcl",
            "bcl": "bcl",
            "dcl": "bcl",
            "gcl": "bcl",
            "epi": "bcl",
            "sil": "bcl",
            "h#": "bcl",
            "#h": "bcl",
            "pau": "bcl",
        }

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        example = self.dataset[idx]
        # covert utternce based on converter
        utterance = example["phonetic_detail"]["utterance"]
        converted_utterance = []
        for phoneme in utterance:
            if phoneme not in self.phoneme_converter:
                converted_utterance.append(phoneme)
            else:
                converted_utterance.append(self.phoneme_converter[phoneme])
        example["phonetic_detail"]["utterance"] = converted_utterance

        # calculate start and stop second
        example["phonetic_detail"]["start_sec"] = []
        for i in range(len(example["phonetic_detail"]["start"])):
            example["phonetic_detail"]["start_sec"].append(example["phonetic_detail"]["start"][i] / 16000)
        example["phonetic_detail"]["stop_sec"] = []
        for i in range(len(example["phonetic_detail"]["stop"])):
            example["phonetic_detail"]["stop_sec"].append(example["phonetic_detail"]["stop"][i] / 16000)

        return (
            idx,
            torch.tensor(example["x"]),
            torch.tensor(example["x_len"]),
            torch.tensor(example["y"]),
            torch.tensor(example["y_len"]),
            example["text"],
            example["phonetic_detail"],
        )

    def collate_fn(self, batch):
        bidx, bx, bx_len, by, by_len, texts, phonetic_details = zip(*batch)

        bx = torch.nn.utils.rnn.pad_sequence(bx, batch_first=True, padding_value=0)
        bx_len = torch.tensor(bx_len)
        by = torch.nn.utils.rnn.pad_sequence(by, batch_first=True, padding_value=0)
        by_len = torch.tensor(by_len)

        return bidx, bx, bx_len, by, by_len, texts, phonetic_details

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


class TIMITDatasetMelSpecdb(torch.utils.data.Dataset):
    """
    要素としてメルスペクトログラム(db)を返すDataset
    """

    def __init__(self, split: str, resampling_rate: int = 16000):
        """
        split: train, test
        """
        self.resampling_rate = resampling_rate
        # dataset実体はhuggingfaceのdatasetsで取得
        self.dataset = datasets.load_dataset(
            "../../datasets/loading_scripts/timit.py",
            data_dir="../../datasets/TIMIT/",
            cache_dir="/home/shibutani/fs/.cache/huggingface/datasets",
        )
        self.dataset = self.dataset[split]
        vocab_file_path = "vocabs/timit.json"
        if split == "train" and not os.path.exists(vocab_file_path):
            # extract vocab from train dataset
            self.extract_vocab(self.dataset["text"], vocab_file_path)
        assert os.path.exists(vocab_file_path), "vocab file not found"
        self.tokenizer = Wav2Vec2CTCTokenizer(
            vocab_file_path, unk_token="[UNK]", pad_token="[PAD]", word_delimiter_token="|"
        )
        self.mel_spec_converter = torchaudio.transforms.MelSpectrogram(
            sample_rate=16000,
            n_fft=400,
            win_length=400,  # 25ms
            hop_length=160,  # 10ms
            n_mels=80,
            window_fn=torch.hann_window,
        )
        self.amplitude_to_db = torchaudio.transforms.AmplitudeToDB(stype="power")

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
        audio = example["audio"]["array"].flatten()
        audio = torch.tensor(audio)
        mel_spec = self.mel_spec_converter(audio)
        mel_spec_db = self.amplitude_to_db(mel_spec)
        x = mel_spec_db.transpose(0, 1)
        x_len = len(x)
        y = self.tokenizer(example["text"]).input_ids
        y_len = len(y)
        example["x"] = x
        example["x_len"] = x_len
        example["y"] = y
        example["y_len"] = y_len
        return (
            idx,
            example["x"],
            torch.tensor(example["x_len"]),
            torch.tensor(example["y"]),
            torch.tensor(example["y_len"]),
            example["text"],
            example["phonetic_detail"],
            audio,
        )

    def collate_fn(self, batch):
        bidx, bx, bx_len, by, by_len, texts, phonetic_details, audios = zip(*batch)

        bx = torch.nn.utils.rnn.pad_sequence(bx, batch_first=True, padding_value=0)
        bx_len = torch.tensor(bx_len)
        by = torch.nn.utils.rnn.pad_sequence(by, batch_first=True, padding_value=self.pad_token_id)
        by_len = torch.tensor(by_len)

        return bidx, bx, bx_len, by, by_len, texts, phonetic_details, audios

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
    def __init__(self, root: str, split: str, resampling_rate: int = 16000):
        """
        split:
        train, dev, test
        """
        if split == "train":
            dataset_train_clean_100 = torchaudio.datasets.LIBRISPEECH(
                os.path.join(root, "librispeech"), url="train-clean-100"
            )
            self.dataset = dataset_train_clean_100
        elif split == "dev":
            dataset_dev_clean = torchaudio.datasets.LIBRISPEECH(os.path.join(root, "librispeech"), url="dev-clean")
            self.dataset = dataset_dev_clean
        elif split == "test":
            dataset_test_clean = torchaudio.datasets.LIBRISPEECH(os.path.join(root, "librispeech"), url="test-clean")
            self.dataset = dataset_test_clean
        else:
            raise ValueError("invalid split")
        self.resampling_rate = resampling_rate

        # trainかつvocabがない場合は作成
        vocab_file_path = "vocabs/librispeech.json"
        if split == "train":
            if not os.path.exists(vocab_file_path):
                self.extract_vocab(vocab_file_path=vocab_file_path)
        # only for normalization of input on quantization
        self.feature_extractor = Wav2Vec2FeatureExtractor.from_pretrained("facebook/wav2vec2-large-lv60")
        self.tokenizer = Wav2Vec2CTCTokenizer(
            vocab_file_path, unk_token="[UNK]", pad_token="[PAD]", word_delimiter_token="|"
        )
        self.mel_spec_converter = torchaudio.transforms.MelSpectrogram(
            sample_rate=16000,
            n_fft=400,
            win_length=400,  # 25ms
            hop_length=160,  # 10ms
            n_mels=80,
            window_fn=torch.hann_window,
        )
        self.amplitude_to_db = torchaudio.transforms.AmplitudeToDB(stype="power")

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
        # "normalized_audio" is only for quantization
        normalized_audio = torch.tensor(
            self.feature_extractor(audio, sampling_rate=self.resampling_rate).input_values[0]
        )
        mel_spec = self.mel_spec_converter(audio)
        mel_spec_db = self.amplitude_to_db(mel_spec)
        x = mel_spec_db.transpose(0, 1)
        x_len = torch.tensor(len(x))
        y = torch.tensor(self.tokenizer(example[2]).input_ids)
        y_len = torch.tensor(len(y))
        # "text" is only for check
        text = example[2]
        return (idx, x, x_len, y, y_len, normalized_audio, text)

    def collate_fn(self, batch):
        bidx, bx, bx_len, by, by_len, baudio, btext = zip(*batch)

        bx = torch.nn.utils.rnn.pad_sequence(bx, batch_first=True, padding_value=0)
        bx_len = torch.tensor(bx_len)
        by = torch.nn.utils.rnn.pad_sequence(by, batch_first=True, padding_value=self.pad_token_id)
        by_len = torch.tensor(by_len)

        return bidx, bx, bx_len, by, by_len, baudio, btext

    def extract_vocab(self, vocab_file_path: str) -> None:
        print("create vocab...")
        all_texts = [example[2] for example in self.dataset]
        all_text = " ".join(all_texts)
        vocab_list = list(set(all_text))

        vocab = {v: k for k, v in enumerate(vocab_list)}
        # use | as delimeter in stead of " "
        print(self.dataset[0])
        vocab["|"] = vocab[" "]
        # delete unused char
        del vocab[" "]
        # add unk and pad token
        vocab["[UNK]"] = len(vocab)
        vocab["[PAD]"] = len(vocab)
        vocab["_"] = len(vocab)

        with open(vocab_file_path, "w") as vocab_file:
            json.dump(vocab, vocab_file)


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
