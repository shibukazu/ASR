import glob
import json
import os
from typing import List

import librosa
import numpy as np
import pandas as pd
import torch
import torchaudio
from torchaudio.transforms import Resample
from modules.spec_aug import SpecAug


class YesNoDataset(torch.utils.data.Dataset):
    def __init__(self, wav_dir_path, model_sample_rate):
        super().__init__()

        dataset = []
        columns = ["path", "text_idx"]
        self.tokens = ["y", "e", "s", "n", "o", " ", "<pad>", "<blank>"]
        self.token_to_idx = {label: i for i, label in enumerate(self.tokens)}
        self.idx_to_token = {i: label for i, label in enumerate(self.tokens)}
        self.blank_idx = self.token_to_idx["<blank>"]
        self.pad_idx = self.token_to_idx["<pad>"]

        for wav_file_path in glob.glob(wav_dir_path + "*.wav"):
            file_name = os.path.splitext(os.path.basename(wav_file_path))[0]
            text_idx = []
            for c in file_name:
                if c == "1":
                    text_idx += [self.token_to_idx[ic] for ic in "yes"]
                elif c == "0":
                    text_idx += [self.token_to_idx[ic] for ic in "no"]
                elif c == "_":
                    text_idx.append(self.token_to_idx[" "])
                else:
                    raise ValueError("Invalid Dir Path")
            dataset.append([wav_file_path, text_idx])

        self.dataset = pd.DataFrame(dataset, columns=columns)
        self.model_sample_rate = model_sample_rate
        self.spectrogram_transformer = torchaudio.transforms.MelSpectrogram(
            # スペクトル設定
            sample_rate=self.model_sample_rate,
            n_fft=400,
            # スペクトログラム設定
            win_length=400,
            hop_length=160,
            window_fn=torch.hann_window,
            # メルスペクトログラム設定
            n_mels=40,
        )

    def __len__(self):
        return self.dataset.shape[0]

    def __getitem__(self, idx):
        wav_file_path = self.dataset.iloc[idx, 0]
        text_idx = self.dataset.iloc[idx, 1]
        wav_data, sample_rate = torchaudio.load(wav_file_path)
        if sample_rate != self.model_sample_rate:
            wav_data = torchaudio.functional.resample(wav_data, sample_rate, self.model_sample_rate)
            sample_rate = self.model_sample_rate
        spectrogram = self.spectrogram_transformer(wav_data)
        spectrogram_db = librosa.amplitude_to_db(spectrogram)

        return spectrogram_db[0].transpose(1, 0), torch.tensor(text_idx, dtype=torch.int32)

    def collate_fn(self, batch):
        # spectrogram_db: tensor[Time, Melbins]
        # text_idx: tensor[text_len]
        spectrogram_dbs, text_idxs = zip(*batch)

        original_spectrogram_db_lens = torch.tensor(
            np.array([len(spectrogram_db) for spectrogram_db in spectrogram_dbs]),
            dtype=torch.int32,
        )
        original_text_idx_lens = torch.tensor(np.array([len(text_idx) for text_idx in text_idxs]), dtype=torch.int32)

        # padding and packing for spectrogram_db
        padded_spectrogram_dbs = []
        for spectrogram_db in spectrogram_dbs:
            padded_spectrogram_db = np.pad(
                spectrogram_db,
                ((0, max(original_spectrogram_db_lens) - spectrogram_db.shape[0]), (0, 0)),
                "constant",
                constant_values=0,
            )
            padded_spectrogram_dbs.append(padded_spectrogram_db)

        padded_spectrogram_dbs = torch.tensor(np.array(padded_spectrogram_dbs))

        # padding and packing for text_idx
        padded_text_idxs = torch.nn.utils.rnn.pad_sequence(text_idxs, batch_first=True, padding_value=self.pad_idx)
        # packed_padded_texts = pack_padded_sequence(
        # padded_texts, original_text_idx_lens, batch_first=True, enforce_sorted=False)

        # テキストはCTCロス計算でしか使わず、RNNに入力しないのでpackingによるマスクは不要
        return padded_spectrogram_dbs, padded_text_idxs, original_spectrogram_db_lens, original_text_idx_lens


class LibriLightBase(torch.utils.data.Dataset):
    _EXT_AUDIO = ".flac"
    _EXT_TRANSCRIPT = ".trans.txt"
    _SUBSET_MAP = {"10m": ["1h/0"], "1h": ["1h/*"], "10h": ["1h/*", "9h"], "9h": ["9h"]}

    class Example:
        def __init__(
            self,
            audio_file_path: str,
            transcript: str,
            speaker_id: int,
            chapter_id: int,
            utterance_id: int,
        ):
            self.audio_file_path = audio_file_path
            self.transcript = transcript
            self.speaker_id = speaker_id
            self.chapter_id = chapter_id
            self.utterance_id = utterance_id

    def __init__(self, subset: str, root: str = "datasets/librispeech_finetuning"):
        self.root = root
        self.subset = subset
        if subset not in ["10m", "1h", "10h", "9h"]:
            raise ValueError(f"subset must be one of '10m', '1h', '10h', '9h' but got {subset}")
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

                    # transcript normalization
                    transcript = transcript.lower()

                    self.examples.append(
                        self.Example(
                            audio_file_path=audio_file_path,
                            transcript=transcript,
                            speaker_id=int(speaker_id),
                            chapter_id=int(chapter_id),
                            utterance_id=int(utterance_id),
                        )
                    )


class LibriLightDataset(torch.utils.data.Dataset):
    def __init__(
        self,
        subset: str,
        spec_aug: SpecAug = None,
        root: str = "datasets/librispeech_finetuning",
        vocab_file_path: str = "vocabs/librilight.json",
    ):
        """
        subset: 10m, 1h, 10h, 9h
        """
        self.base = LibriLightBase(subset=subset, root=root)
        self.dataset = self.base.examples
        self.mel_spec_converter = torchaudio.transforms.MelSpectrogram(
            sample_rate=16000,
            n_fft=400,
            win_length=400,  # 25ms
            hop_length=160,  # 10ms
            n_mels=80,
            window_fn=torch.hann_window,
        )
        self.amplitude_to_db = torchaudio.transforms.AmplitudeToDB(stype="power")
        self.spec_aug = spec_aug

        if not os.path.exists(vocab_file_path):
            # extract vocab from train dataset
            texts = []
            for example in self.dataset:
                texts.append(example.transcript)
            self.extract_vocab(texts, vocab_file_path)
        else:
            if not os.path.exists(vocab_file_path):
                raise ValueError(f"vocab file not found at {vocab_file_path}")

        with open(vocab_file_path, "r") as f:
            token_to_idx = json.load(f)
        self.token_to_idx = token_to_idx
        self.idx_to_token = {v: k for k, v in self.token_to_idx.items()}
        self.pad_idx = self.token_to_idx["<pad>"]
        self.unk_idx = self.token_to_idx["<unk>"]
        self.blank_idx = self.token_to_idx["<blank>"]

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        example = self.dataset[idx]
        audio, sampling_rate = torchaudio.load(example.audio_file_path)
        resampler = Resample(sampling_rate, 16000)
        audio = resampler(audio).flatten()
        mel_spec = self.mel_spec_converter(audio)
        mel_spec_db = self.amplitude_to_db(mel_spec)

        x = mel_spec_db.transpose(0, 1)
        if self.spec_aug is not None:
            x = self.spec_aug(x)
        x_len = len(x)
        y = self.convert_text_to_token_indices(example.transcript)
        y_len = len(y)

        transcript = example.transcript
        # audio file related info
        speaker_id = example.speaker_id
        chapter_id = example.chapter_id
        utterance_id = example.utterance_id

        return (
            idx,
            x,
            torch.tensor(x_len, dtype=torch.int32),
            torch.tensor(y, dtype=torch.int32),
            torch.tensor(y_len, dtype=torch.int32),
            transcript,
            speaker_id,
            chapter_id,
            utterance_id,
        )

    def collate_fn(self, batch):
        bidx, bx, bx_len, by, by_len, btranscript, bspeaker_id, bchapter_id, butterance_id = zip(*batch)
        bx = torch.nn.utils.rnn.pad_sequence(bx, batch_first=True, padding_value=0)
        bx_len = torch.tensor(bx_len)
        by = torch.nn.utils.rnn.pad_sequence(by, batch_first=True, padding_value=self.pad_idx)
        by_len = torch.tensor(by_len)

        return (
            bx,
            by,
            bx_len,
            by_len,
        )

    def extract_vocab(self, all_text: List, vocab_file_path: str) -> None:
        print("extracting vocab...")
        all_text = " ".join(all_text)
        token_list = list(set(all_text))

        token_to_idx = {v: k for k, v in enumerate(token_list)}
        # add unk, pad, blank token
        token_to_idx["<unk>"] = len(token_to_idx)
        token_to_idx["<pad>"] = len(token_to_idx)
        token_to_idx["<blank>"] = len(token_to_idx)

        with open(vocab_file_path, "w") as f:
            json.dump(token_to_idx, f)

    def convert_text_to_token_indices(self, text: str) -> List[int]:
        token_indices = []
        for char in text:
            if char in self.token_to_idx:
                token_indices.append(self.token_to_idx[char])
            else:
                token_indices.append(self.unk_idx)
        return token_indices
