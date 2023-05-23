import json
import random

import torch
import torchaudio
from modules.spec_aug import SpecAug
from tokenizer import SentencePieceTokenizer
from torchaudio.transforms import Resample
from tqdm import tqdm


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


class CSJVADPretrainDataset(torch.utils.data.Dataset):
    def __init__(
        self,
        json_file_path: str,
        tokenizer: SentencePieceTokenizer,
        resampling_rate: int = 16000,
        spec_aug: SpecAug = None,
    ):
        super().__init__()
        with open(json_file_path) as f:
            self.json = json.load(f)
        speakers = list(self.json.keys())
        self.inner_json = {}
        for speaker in speakers:
            for key in self.json[speaker]:
                self.inner_json[key] = self.json[speaker][key]
        self.inner_keys = list(self.inner_json.keys())
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
        return len(self.inner_json)

    def __getitem__(self, idx):
        key = self.inner_keys[idx]
        data = self.inner_json[key]
        wav_file_path, sampling_rate, audio_sec, raw_transcript, vad, subsampled_vad = (
            data["wav_file_path"],
            data["sampling_rate"],
            data["audio_sec"],
            data["raw_transcript"],
            data["vad"],
            data["subsampled_vad"],
        )
        audio, sampling_rate = torchaudio.load(wav_file_path)
        audio = audio.flatten()
        resampler = Resample(sampling_rate, self.resampling_rate)
        resampled_audio = resampler(audio)
        mel_spec = self.mel_spec_converter(resampled_audio)
        mel_spec_db = self.amplitude_to_db(mel_spec)

        x = mel_spec_db.transpose(0, 1)
        # assert len(x) > len(vad)
        # vadと同じ長さにする
        x = x[: len(vad), :]
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
            torch.tensor(subsampled_vad, dtype=torch.float32),
            torch.tensor(len(subsampled_vad), dtype=torch.int32),
        )

    def get_audio_sec(self, idx):
        key = self.inner_keys[idx]
        data = self.inner_json[key]
        audio_sec = data["audio_sec"]
        return audio_sec

    def get_text_len(self, idx):
        key = self.inner_keys[idx]
        data = self.inner_json[key]
        text_len = len(data["raw_transcript"])
        return text_len


class CSJVADAdaptationDataset(torch.utils.data.Dataset):
    def __init__(
        self,
        json_file_path: str,
        tokenizer: SentencePieceTokenizer,
        resampling_rate: int = 16000,
        spec_aug: SpecAug = None,
    ):
        super().__init__()
        with open(json_file_path) as f:
            self.json = json.load(f)
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
        wav_file_path, sampling_rate, audio_sec, raw_transcript, vad = (
            data["wav_file_path"],
            data["sampling_rate"],
            data["audio_sec"],
            data["raw_transcript"],
            data["vad"],
        )
        audio, sampling_rate = torchaudio.load(wav_file_path)
        audio = audio.flatten()
        resampler = Resample(sampling_rate, self.resampling_rate)
        resampled_audio = resampler(audio)
        mel_spec = self.mel_spec_converter(resampled_audio)
        mel_spec_db = self.amplitude_to_db(mel_spec)

        x = mel_spec_db.transpose(0, 1)
        # assert len(x) > len(vad)
        # vadと同じ長さにする
        x = x[: len(vad), :]
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
            torch.tensor(vad, dtype=torch.float32),
            torch.tensor(len(vad), dtype=torch.int32),
        )

    def get_audio_sec(self, idx):
        key = self.inner_keys[idx]
        data = self.inner_json[key]
        audio_sec = data["audio_sec"]
        return audio_sec

    def get_text_len(self, idx):
        key = self.inner_keys[idx]
        data = self.inner_json[key]
        text_len = len(data["raw_transcript"])
        return text_len


class CSJVAD1PathAdaptationDataset(torch.utils.data.Dataset):
    def __init__(
        self,
        json_file_path: str,
        tokenizer: SentencePieceTokenizer,
        resampling_rate: int = 16000,
        spec_aug: SpecAug = None,
    ):
        super().__init__()
        with open(json_file_path) as f:
            self.json = json.load(f)
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
        _wav_file_paths, _audio_sec, _raw_transcripts, _vads, _subsampled_vads = (
            data["wav_file_paths"],
            data["audio_sec"],
            data["raw_transcripts"],
            data["vads"],
            data["subsampled_vads"],
        )
        xs, x_lens = [], []
        ys, y_lens = [], []
        subsampled_vads, subsampled_vad_lens = [], []
        vads, vad_lens = [], []
        audio_sec = _audio_sec
        num_wav_files = len(_wav_file_paths)
        for i in range(num_wav_files):
            vad = _vads[i]
            vad = torch.tensor(vad, dtype=torch.float32)
            vads.append(vad)
            vad_len = len(vad)
            vad_len = torch.tensor(len(vad), dtype=torch.int32)
            vad_lens.append(vad_len)
            subsampled_vad = _subsampled_vads[i]
            subsampled_vad = torch.tensor(subsampled_vad, dtype=torch.float32)
            subsampled_vads.append(subsampled_vad)
            subsampled_vad_len = len(subsampled_vad)
            subsampled_vad_len = torch.tensor(len(subsampled_vad), dtype=torch.int32)
            subsampled_vad_lens.append(subsampled_vad_len)

            wav_file_path = _wav_file_paths[i]
            audio, sampling_rate = torchaudio.load(wav_file_path)
            audio = audio.flatten()
            resampler = Resample(sampling_rate, self.resampling_rate)
            resampled_audio = resampler(audio)

            mel_spec = self.mel_spec_converter(resampled_audio)
            mel_spec_db = self.amplitude_to_db(mel_spec)

            x = mel_spec_db.transpose(0, 1)
            x = x[: len(vad), :]
            if self.spec_aug is not None:
                x = self.spec_aug(x)
            x_len = len(x)
            x_len = torch.tensor(x_len, dtype=torch.int32)

            xs.append(x)
            x_lens.append(x_len)

            raw_transcript = _raw_transcripts[i]
            y = self.tokenizer.text_to_token_ids(raw_transcript)
            y = torch.tensor(y, dtype=torch.int32)
            y_len = len(y)
            y_len = torch.tensor(y_len, dtype=torch.int32)

            ys.append(y)
            y_lens.append(y_len)

        return (idx, xs, ys, x_lens, y_lens, audio_sec, vads, subsampled_vads, subsampled_vads, subsampled_vad_lens)


class CSJVADUtteranceAdaptationDataset(torch.utils.data.Dataset):
    """
    発話単位でのオフラインAdaptation実験
    """
    def __init__(
        self,
        json_file_path: str,
        tokenizer: SentencePieceTokenizer,
        resampling_rate: int = 16000,
        spec_aug: SpecAug = None,
    ):
        super().__init__()
        with open(json_file_path) as f:
            self.json = json.load(f)
        speakers = list(self.json.keys())
        self.inner_json = {}
        for speaker in speakers:
            for key in self.json[speaker]:
                self.inner_json[key] = self.json[speaker][key]
        self.inner_keys = list(self.inner_json.keys())
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
        return len(self.inner_json)

    def __getitem__(self, idx):
        key = self.inner_keys[idx]
        data = self.inner_json[key]
        wav_file_path, sampling_rate, audio_sec, raw_transcript, vad = (
            data["wav_file_path"],
            data["sampling_rate"],
            data["audio_sec"],
            data["raw_transcript"],
            data["vad"],
        )
        audio, sampling_rate = torchaudio.load(wav_file_path)
        audio = audio.flatten()
        resampler = Resample(sampling_rate, self.resampling_rate)
        resampled_audio = resampler(audio)
        mel_spec = self.mel_spec_converter(resampled_audio)
        mel_spec_db = self.amplitude_to_db(mel_spec)

        x = mel_spec_db.transpose(0, 1)
        # assert len(x) > len(vad)
        # vadと同じ長さにする
        x = x[: len(vad), :]
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
            torch.tensor(vad, dtype=torch.float32),
            torch.tensor(len(vad), dtype=torch.int32),
        )

    def get_audio_sec(self, idx):
        key = self.inner_keys[idx]
        data = self.inner_json[key]
        audio_sec = data["audio_sec"]
        return audio_sec

    def get_text_len(self, idx):
        key = self.inner_keys[idx]
        data = self.inner_json[key]
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


def get_vad_pretrain_dataloader(
    dataset,
    batch_sec,
    batch_text_len,
    num_workers,
    pin_memory,
    pad_idx,
):
    def collate_fn(batch):
        bidx, bx, by, bx_len, by_len, baudio_sec, bsubsampled_vad, bsubsampled_vad_len = zip(*batch)

        bx = torch.nn.utils.rnn.pad_sequence(bx, batch_first=True, padding_value=0)
        bx_len = torch.tensor(bx_len)
        by = torch.nn.utils.rnn.pad_sequence(by, batch_first=True, padding_value=pad_idx)
        by_len = torch.tensor(by_len)
        bsubsampled_vad = torch.nn.utils.rnn.pad_sequence(bsubsampled_vad, batch_first=True, padding_value=0)
        bsubsampled_vad_len = torch.tensor(bsubsampled_vad_len)

        return bidx, bx, by, bx_len, by_len, baudio_sec, bsubsampled_vad, bsubsampled_vad_len

    dataloader = torch.utils.data.DataLoader(
        dataset,
        batch_sampler=RandomTimeTextLenFixedBatchSampler(dataset, batch_sec, batch_text_len),
        num_workers=num_workers,
        collate_fn=collate_fn,
        pin_memory=pin_memory,
    )

    return dataloader
