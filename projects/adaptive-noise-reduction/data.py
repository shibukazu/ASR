import json
import random

import torch
import torchaudio
from modules.spec_aug import SpecAug
from tokenizer import SentencePieceTokenizer
from torchaudio.transforms import Resample
from tqdm import tqdm


class YesNoDataset(torch.utils.data.Dataset):
    def __init__(
        self,
        json_file_path: str,
        tokenizer: SentencePieceTokenizer,
        resampling_rate: int = 16000,
        spec_aug: SpecAug = None,
    ):
        super().__init__()

        self.json = json.load(open(json_file_path))
        self.spec_aug = spec_aug
        self.tokenizer = tokenizer
        self.resampling_rate = resampling_rate
        self.mel_spec_converter = torchaudio.transforms.MelSpectrogram(
            # スペクトル設定
            sample_rate=self.resampling_rate,
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
        return len(self.json)

    def __getitem__(self, idx):
        data = self.json[str(idx)]
        wav_file_path, sampling_rate, audio_sec, raw_transcript = (
            data["wav_file_path"],
            data["sampling_rate"],
            data["audio_sec"],
            data["raw_transcript"],
        )
        audio, sampling_rate = torchaudio.load(wav_file_path)
        audio = audio.flatten()
        resampler = Resample(sampling_rate, self.resampling_rate)
        resampled_audio = resampler(audio)
        mel_spec = self.mel_spec_converter(resampled_audio)
        mel_spec_db = self.amplitude_to_db(mel_spec)

        x = mel_spec_db.transpose(0, 1)
        if self.spec_aug is not None:
            x = self.spec_aug(x)
        x_len = len(x)
        transcript = raw_transcript.lower()
        y = self.tokenizer.text_to_token_ids(transcript)
        y_len = len(y)

        return (
            idx,
            x,
            torch.tensor(y, dtype=torch.int32),
            torch.tensor(x_len, dtype=torch.int32),
            torch.tensor(y_len, dtype=torch.int32),
            audio_sec,
        )

    def get_audio_sec(self, idx):
        data = self.json[str(idx)]
        _, _, audio_sec, _ = (
            data["wav_file_path"],
            data["sampling_rate"],
            data["audio_sec"],
            data["raw_transcript"],
        )
        return audio_sec


class CSJDataset(torch.utils.data.Dataset):
    def __init__(
        self,
        json_file_path: str,
        tokenizer: SentencePieceTokenizer,
        resampling_rate: int = 16000,
        spec_aug: SpecAug = None,
    ):
        super().__init__()

        self.json = json.load(open(json_file_path))
        self.keys = list(self.json.keys())
        self.spec_aug = spec_aug
        self.tokenizer = tokenizer
        self.resampling_rate = resampling_rate
        self.mel_spec_converter = torchaudio.transforms.MelSpectrogram(
            # スペクトル設定
            sample_rate=self.resampling_rate,
            n_fft=400,
            # スペクトログラム設定
            win_length=400,
            hop_length=160,
            window_fn=torch.hann_window,
            # メルスペクトログラム設定
            n_mels=80,
        )
        self.amplitude_to_db = torchaudio.transforms.AmplitudeToDB(stype="power")

    def __len__(self):
        return len(self.json)

    def __getitem__(self, idx):
        key = self.keys[idx]
        data = self.json[key]
        wav_file_path, sampling_rate, audio_sec, raw_transcript = (
            data["wav_file_path"],
            data["sampling_rate"],
            data["audio_sec"],
            data["raw_transcript"],
        )
        audio, sampling_rate = torchaudio.load(wav_file_path)
        audio = audio.flatten()
        resampler = Resample(sampling_rate, self.resampling_rate)
        resampled_audio = resampler(audio)
        mel_spec = self.mel_spec_converter(resampled_audio)
        mel_spec_db = self.amplitude_to_db(mel_spec)

        x = mel_spec_db.transpose(0, 1)
        if self.spec_aug is not None:
            x = self.spec_aug(x)
        x_len = len(x)
        y = self.tokenizer.text_to_token_ids(raw_transcript)
        y_len = len(y)

        return (
            idx,
            x,
            torch.tensor(y, dtype=torch.int32),
            torch.tensor(x_len, dtype=torch.int32),
            torch.tensor(y_len, dtype=torch.int32),
            audio_sec,
        )

    def get_audio_sec(self, idx):
        key = self.keys[idx]
        data = self.json[key]
        audio_sec = data["audio_sec"]
        return audio_sec

    def get_text_len(self, idx):
        key = self.keys[idx]
        data = self.json[key]
        text_len = len(data["raw_transcript"])
        return text_len


class RandomTimeTextLenFixedBatchSampler(torch.utils.data.Sampler):
    def __init__(self, dataset, batch_sec, batch_text_len):
        self.dataset = dataset
        self.batch_sec = batch_sec
        self.batch_text_len = batch_text_len
        self.batches = []

        bar = tqdm(total=len(self.dataset))
        bar.set_description("Batch Prepare")
        indices = list(range(len(self.dataset)))
        random.shuffle(indices)
        sampled_size = 0
        while sampled_size < len(self.dataset):
            batch = []
            sampled_sec = 0
            sampled_text_len = 0
            while sampled_size < len(self.dataset):
                audio_sec = self.dataset.get_audio_sec(indices[sampled_size])
                text_len = self.dataset.get_text_len(indices[sampled_size])
                if audio_sec + sampled_sec > self.batch_sec or text_len + sampled_text_len > self.batch_text_len:
                    break
                batch.append(indices[sampled_size])
                sampled_sec += audio_sec
                sampled_text_len += text_len
                sampled_size += 1
                bar.update(1)
            self.batches.append(batch)

    def __iter__(self):
        random.shuffle(self.batches)
        return iter(self.batches)

    def __len__(self):
        return len(self.batches)


class RandomTimeFixedBatchSampler(torch.utils.data.Sampler):
    def __init__(self, dataset, batch_sec):
        self.dataset = dataset
        self.batch_sec = batch_sec
        self.batches = []

        bar = tqdm(total=len(self.dataset))
        bar.set_description("Batch Prepare")
        indices = list(range(len(self.dataset)))
        random.shuffle(indices)
        sampled_size = 0
        while sampled_size < len(self.dataset):
            batch = []
            sampled_sec = 0
            while sampled_size < len(self.dataset):
                audio_sec = self.dataset.get_audio_sec(indices[sampled_size])
                if audio_sec + sampled_sec > self.batch_sec:
                    break
                batch.append(indices[sampled_size])
                sampled_sec += audio_sec
                sampled_size += 1
                bar.update(1)
            self.batches.append(batch)

    def __iter__(self):
        random.shuffle(self.batches)
        return iter(self.batches)

    def __len__(self):
        return len(self.batches)


def get_dataloader(
    dataset,
    batch_sec,
    batch_text_len,
    num_workers,
    pin_memory,
    pad_idx,
):
    def collate_fn(batch):
        bidx, bx, by, bx_len, by_len, baudio_sec = zip(*batch)

        bx = torch.nn.utils.rnn.pad_sequence(bx, batch_first=True, padding_value=0)
        bx_len = torch.tensor(bx_len)
        by = torch.nn.utils.rnn.pad_sequence(by, batch_first=True, padding_value=pad_idx)
        by_len = torch.tensor(by_len)

        return bidx, bx, by, bx_len, by_len, baudio_sec

    dataloader = torch.utils.data.DataLoader(
        dataset,
        batch_sampler=RandomTimeTextLenFixedBatchSampler(dataset, batch_sec, batch_text_len),
        num_workers=num_workers,
        collate_fn=collate_fn,
        pin_memory=pin_memory,
    )

    return dataloader
