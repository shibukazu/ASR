import glob
import json
import os
import random
from typing import List

import pandas as pd
import torch
import torchaudio
from modules.spec_aug import SpecAug
from torchaudio.transforms import Resample
from tqdm import tqdm


class YesNoDataset(torch.utils.data.Dataset):
    def __init__(self, wav_dir_path, model_sample_rate, split):
        super().__init__()

        dataset = []
        columns = ["path", "text_idx"]
        self.tokens = ["y", "e", "s", "n", "o", " ", "<pad>", "<blank>"]
        self.token_to_idx = {label: i for i, label in enumerate(self.tokens)}
        self.idx_to_token = {i: label for i, label in enumerate(self.tokens)}
        self.blank_idx = self.token_to_idx["<blank>"]
        self.pad_idx = self.token_to_idx["<pad>"]
        all_wav_file_paths = glob.glob(wav_dir_path + "*.wav")
        sorted(all_wav_file_paths)
        if split == "train":
            wav_file_paths = all_wav_file_paths[: int(len(all_wav_file_paths) * 0.8)]
        elif split == "dev":
            wav_file_paths = all_wav_file_paths[int(len(all_wav_file_paths) * 0.8) :]
        else:
            raise ValueError("Invalid Split")
        for wav_file_path in wav_file_paths:
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
        self.mel_spec_converter = torchaudio.transforms.MelSpectrogram(
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
        self.amplitude_to_db = torchaudio.transforms.AmplitudeToDB(stype="power")

    def __len__(self):
        return self.dataset.shape[0]

    def __getitem__(self, idx):
        wav_file_path = self.dataset.iloc[idx, 0]
        audio, sample_rate = torchaudio.load(wav_file_path)
        audio = audio.flatten()
        resampler = Resample(sample_rate, self.model_sample_rate)
        resampled_audio = resampler(audio)
        mel_spec = self.mel_spec_converter(resampled_audio)
        mel_spec_db = self.amplitude_to_db(mel_spec)
        x = mel_spec_db.transpose(0, 1)
        x_len = len(x)

        token_indices = self.dataset.iloc[idx, 1]
        y = token_indices
        y_len = len(y)

        audio_sec = len(resampled_audio) / self.model_sample_rate

        return (
            idx,
            x,
            torch.tensor(y, dtype=torch.int32),
            torch.tensor(x_len, dtype=torch.int32),
            torch.tensor(y_len, dtype=torch.int32),
            audio_sec,
        )


class LibriSpeechDataset(torch.utils.data.Dataset):
    def __init__(
        self,
        root: str,
        split: str,
        audio_sec_file_path: str,
        resampling_rate: int = 16000,
        vocab_file_path: str = "vocab.json",
        spec_aug: SpecAug = None,
    ):
        """
        split:
        train, dev, test, dev-test
        """
        if split == "train-100h":
            self.dataset = torchaudio.datasets.LIBRISPEECH(os.path.join(root, "librispeech"), url="train-clean-100")
        elif split == "train-360h":
            self.dataset = torchaudio.datasets.LIBRISPEECH(os.path.join(root, "librispeech"), url="train-clean-360")
        elif split == "train-500h":
            self.dataset = torchaudio.datasets.LIBRISPEECH(os.path.join(root, "librispeech"), url="train-other-500")
        elif split == "train-960h":
            dataset_train_clean_100 = torchaudio.datasets.LIBRISPEECH(
                os.path.join(root, "librispeech"), url="train-clean-100"
            )
            dataset_train_clean_360 = torchaudio.datasets.LIBRISPEECH(
                os.path.join(root, "librispeech"), url="train-clean-360"
            )
            dataset_train_other_500 = torchaudio.datasets.LIBRISPEECH(
                os.path.join(root, "librispeech"), url="train-other-500"
            )
            self.dataset = dataset_train_clean_100 + dataset_train_clean_360 + dataset_train_other_500
        elif split == "dev-clean":
            self.dataset = torchaudio.datasets.LIBRISPEECH(os.path.join(root, "librispeech"), url="dev-clean")
        elif split == "dev-other":
            self.dataset = torchaudio.datasets.LIBRISPEECH(os.path.join(root, "librispeech"), url="dev-other")
        elif split == "dev":
            dataset_dev_clean = torchaudio.datasets.LIBRISPEECH(os.path.join(root, "librispeech"), url="dev-clean")
            dataset_dev_other = torchaudio.datasets.LIBRISPEECH(os.path.join(root, "librispeech"), url="dev-other")
            self.dataset = dataset_dev_clean + dataset_dev_other
        elif split == "test-clean":
            self.dataset = torchaudio.datasets.LIBRISPEECH(os.path.join(root, "librispeech"), url="test-clean")
        elif split == "test-other":
            self.dataset = torchaudio.datasets.LIBRISPEECH(os.path.join(root, "librispeech"), url="test-other")
        elif split == "test":
            dataset_test_clean = torchaudio.datasets.LIBRISPEECH(os.path.join(root, "librispeech"), url="test-clean")
            dataset_test_other = torchaudio.datasets.LIBRISPEECH(os.path.join(root, "librispeech"), url="test-other")
            self.dataset = dataset_test_clean + dataset_test_other
        else:
            raise ValueError("Invalid split")
        self.resampling_rate = resampling_rate
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
            transcripts = []
            for example in self.dataset:
                transcript = example[2]
                transcript = transcript.lower()
                transcripts.append(transcript)
            self.extract_vocab(transcripts, vocab_file_path)
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

        with open(audio_sec_file_path, "r") as f:
            self.audio_sec_dict = json.load(f)

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        audio, sampling_rate, transcript, speaker_id, chapter_id, utterance_id = self.dataset[idx]
        audio = audio.flatten()
        resampler = Resample(sampling_rate, self.resampling_rate)
        resampled_audio = resampler(audio)
        mel_spec = self.mel_spec_converter(resampled_audio)
        mel_spec_db = self.amplitude_to_db(mel_spec)
        x = mel_spec_db.transpose(0, 1)
        if self.spec_aug is not None:
            x = self.spec_aug(x)
        x_len = len(x)
        transcript = transcript.lower()
        y = self.convert_transcript_to_token_indices(transcript)
        y_len = len(y)

        audio_sec = len(resampled_audio) / self.resampling_rate
        assert audio_sec == self.audio_sec_dict[str(idx)]

        return (
            idx,
            x,
            torch.tensor(y, dtype=torch.int32),
            torch.tensor(x_len, dtype=torch.int32),
            torch.tensor(y_len, dtype=torch.int32),
            audio_sec,
        )

    def extract_vocab(self, all_transcripts: List, vocab_file_path: str) -> None:
        print("Extracting vocab...")
        all_transcripts = " ".join(all_transcripts)
        token_list = list(set(all_transcripts))

        token_to_idx = {v: k for k, v in enumerate(token_list)}
        # add unk, pad, blank token
        token_to_idx["<unk>"] = len(token_to_idx)
        token_to_idx["<pad>"] = len(token_to_idx)
        token_to_idx["<blank>"] = len(token_to_idx)

        with open(vocab_file_path, "w") as f:
            json.dump(token_to_idx, f)

    def convert_transcript_to_token_indices(self, transcript: str) -> List[int]:
        token_indices = []
        for token in transcript:
            if token in self.token_to_idx:
                token_indices.append(self.token_to_idx[token])
            else:
                token_indices.append(self.unk_idx)
        return token_indices


class RandomTimeBatchSampler(torch.utils.data.Sampler):
    def __init__(self, dataset, batch_sec, audio_sec_file_path):
        self.dataset = dataset
        self.batch_sec = batch_sec
        self.num_batches = 0
        with open(audio_sec_file_path, "r") as f:
            self.audio_sec_dict = json.load(f)

    def __iter__(self):
        bar = tqdm(total=len(self.dataset))
        bar.set_description("Batch Prepare")
        batches = []
        indices = list(range(len(self.dataset)))
        random.shuffle(indices)
        sampled_size = 0
        while sampled_size < len(self.dataset):
            batch = []
            sampled_sec = 0
            while sampled_sec < self.batch_sec and sampled_size < len(self.dataset):
                batch.append(indices[sampled_size])
                audio_sec = self.audio_sec_dict[str(indices[sampled_size])]
                sampled_sec += audio_sec
                sampled_size += 1
                bar.update(1)
            batches.append(batch)
        self.num_batches = len(batches)
        return iter(batches)

    def __len__(self):
        # NOTE: This is not trustworthy
        return self.num_batches


class Collate:
    def __init__(self, pad_idx):
        self.pad_idx = pad_idx

    def fn(self, batch):
        bidx, bx, by, bx_len, by_len, baudio_sec = zip(*batch)

        bx = torch.nn.utils.rnn.pad_sequence(bx, batch_first=True, padding_value=0)
        bx_len = torch.tensor(bx_len)
        by = torch.nn.utils.rnn.pad_sequence(by, batch_first=True, padding_value=self.pad_idx)
        by_len = torch.tensor(by_len)

        return bidx, bx, by, bx_len, by_len, baudio_sec


def get_dataloader(
    dataset,
    batch_sec,
    num_workers,
    pad_idx,
    pin_memory,
    audio_sec_file_path,
):
    collate_fn = Collate(pad_idx).fn

    dataloader = torch.utils.data.DataLoader(
        dataset,
        batch_sampler=RandomTimeBatchSampler(dataset, batch_sec, audio_sec_file_path),
        num_workers=num_workers,
        collate_fn=collate_fn,
        pin_memory=pin_memory,
    )

    return dataloader
