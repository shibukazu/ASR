import glob
import json
import os
from typing import Callable, List, Tuple

import torch
import torchaudio
from torch import Tensor
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

                    # transcript normalization
                    transcript = transcript.lower()
                    transcript = transcript + "\n"

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
        sampling_rate: int = 16000,
    ):
        """
        subset: 10m, 1h, 10h
        """
        self.base = LibriLightBase(subset=subset, root=root)
        self.dataset = self.base.examples
        self.sampling_rate = sampling_rate
        # only for normalization of input
        self.feature_extractor = Wav2Vec2FeatureExtractor.from_pretrained("facebook/wav2vec2-base")

        if not os.path.exists(vocab_file_path):
            # extract vocab from train dataset
            texts = []
            for example in self.dataset:
                texts.append(example.transcript)
            print("create vocab")
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
        x = self.feature_extractor(audio, sampling_rate=self.sampling_rate).input_values[0]
        x_len = len(x)
        y = self.tokenizer(example.transcript).input_ids
        y_len = len(y)

        transcript = example.transcript

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

        return (
            bidx,
            bx,
            bx_len,
            by,
            by_len,
            btranscript,
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


class TEDLIUMRelease2Base(torch.utils.data.Dataset):
    def __init__(
        self,
        talk_id: str,
        root: str = "datasets/TEDLIUM_release2",
        subset: str = "train",
    ) -> None:

        self._path = os.path.join(root, subset)
        # Create list for all samples
        self._lines = None
        self._talk_id = None
        stm_path = os.path.join(self._path, "stm")

        files = os.listdir(stm_path)
        if talk_id + ".stm" in files:
            stm_path = os.path.join(self._path, "stm", talk_id + ".stm")
            self._talk_id = talk_id
            with open(stm_path) as f:
                lines = len(f.readlines())
                self._lines = list(range(lines))
        else:
            raise ValueError("talk_id is not valid")
        # Create dict path for later read
        self._dict_path = os.path.join(root, "TEDLIUM.152k.dic")
        self._phoneme_dict = None

    def _load_tedlium_item(self, line: int) -> Tuple[Tensor, int, str, int, int, int]:
        """Loads a TEDLIUM dataset sample given a file name and corresponding sentence name.

        Args:
            line (int): Line identifier for the sample inside the text file

        Returns:
            (Tensor, int, str, int, int, int):
            ``(waveform, sample_rate, transcript, talk_id, speaker_id, identifier)``
        """
        transcript_path = os.path.join(self._path, "stm", self._talk_id) + ".stm"
        with open(transcript_path) as f:
            transcript = f.readlines()[line]
            talk_id, _, speaker_id, start_time, end_time, identifier, transcript = transcript.split(" ", 6)

        wave_path = os.path.join(self._path, "sph", self._talk_id) + ".sph"
        waveform, sample_rate = self._load_audio(wave_path, start_time=start_time, end_time=end_time)
        return (waveform, sample_rate, transcript, talk_id, speaker_id, identifier)

    def _load_audio(
        self, path: str, start_time: float, end_time: float, sample_rate: int = 16000
    ) -> Tuple[Tensor, int]:
        """Default load function used in TEDLIUM dataset, you can overwrite this function to customize functionality
        and load individual sentences from a full ted audio talk file.

        Args:
            path (str): Path to audio file
            start_time (int): Time in seconds where the sample sentence stars
            end_time (int): Time in seconds where the sample sentence finishes
            sample_rate (float, optional): Sampling rate

        Returns:
            [Tensor, int]: Audio tensor representation and sample rate
        """
        start_time = int(float(start_time) * sample_rate)
        end_time = int(float(end_time) * sample_rate)

        kwargs = {"frame_offset": start_time, "num_frames": end_time - start_time}

        return torchaudio.load(path, **kwargs)

    def __getitem__(self, n: int) -> Tuple[Tensor, int, str, int, int, int]:
        """Load the n-th sample from the dataset.

        Args:
            n (int): The index of the sample to be loaded

        Returns:
            Tuple of the following items;

            Tensor:
                Waveform
            int:
                Sample rate
            str:
                Transcript
            int:
                Talk ID
            int:
                Speaker ID
            int:
                Identifier
        """
        line = self._lines[n]
        return self._load_tedlium_item(line)

    def __len__(self) -> int:
        """TEDLIUM dataset custom function overwritting len default behaviour.

        Returns:
            int: TEDLIUM dataset length
        """
        return len(self._lines)

    @property
    def phoneme_dict(self):
        """dict[str, tuple[str]]: Phonemes. Mapping from word to tuple of phonemes.
        Note that some words have empty phonemes.
        """
        # Read phoneme dictionary
        if not self._phoneme_dict:
            self._phoneme_dict = {}
            with open(self._dict_path, "r", encoding="utf-8") as f:
                for line in f.readlines():
                    content = line.strip().split()
                    self._phoneme_dict[content[0]] = tuple(content[1:])  # content[1:] can be empty list
        return self._phoneme_dict.copy()


class TEDLIUMRelease2Dataset(torch.utils.data.Dataset):
    def __init__(
        self,
        talk_id: str,
        subset: str = "train",
        root: str = "datasets/TEDLIUM_release2",
        vocab_file_path: str = "vocabs/librilight.json",
        sampling_rate: int = 16000,
    ):
        """
        subset: 10m, 1h, 10h
        """
        self.dataset = TEDLIUMRelease2Base(subset=subset, root=root, talk_id=talk_id)
        self.sampling_rate = sampling_rate
        # only for normalization of input
        self.feature_extractor = Wav2Vec2FeatureExtractor.from_pretrained("facebook/wav2vec2-base")

        if not os.path.exists(vocab_file_path):
            # extract vocab from train dataset
            texts = []
            for example in self.dataset:
                texts.append(example[2])
            print("create vocab")
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
        audio = example[0]
        audio = audio.flatten()
        x = self.feature_extractor(audio, sampling_rate=self.sampling_rate).input_values[0]
        x_len = len(x)
        transcript = example[2]
        y = self.tokenizer(transcript).input_ids
        y_len = len(y)

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

        return (
            bidx,
            bx,
            bx_len,
            by,
            by_len,
            btranscript,
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
