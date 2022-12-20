import torch
from whisper.audio import N_FRAMES, log_mel_spectrogram, pad_or_trim
from whisper.tokenizer import get_tokenizer

import datasets


class TIMITDataset(torch.utils.data.Dataset):
    def __init__(self, device: str, is_train: bool):
        self.type = "train" if is_train else "test"
        # dataset実体はhuggingfaceのdatasetsで取得
        self.dataset = datasets.load_dataset(
            "../../datasets/loading_scripts/timit.py",
            data_dir="../../datasets/TIMIT/",
            cache_dir="/home/shibutani/fs/.cache/huggingface/datasets")
        self.dataset = self.dataset[self.type]
        self.language = "en"
        self.tokenizer = get_tokenizer(multilingual=False, language=self.language)

        def preprocess_function(example):
            mel = log_mel_spectrogram(example["file"])
            x = pad_or_trim(mel, N_FRAMES)
            x_len = len(x)
            label = self.tokenizer.encode(example["text"])
            y_input = [
                self.tokenizer._get_single_token_id("<|startoftranscript|>"),
                self.tokenizer._get_single_token_id("<|en|>"),
                self.tokenizer._get_single_token_id("<|transcribe|>"),
                self.tokenizer._get_single_token_id("<|notimestamps|>")
            ] + label
            y_input_len = len(y_input)
            y_target = label + [self.tokenizer._get_single_token_id("<|endoftext|>")]
            y_target_len = len(y_target)
            example["x"] = x
            example["x_len"] = x_len
            example["y_input"] = y_input
            example["y_input_len"] = y_input_len
            example["y_target"] = y_target
            example["y_target_len"] = y_target_len

            return example

        self.dataset = self.dataset.map(preprocess_function, num_proc=8)

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        example = self.dataset[idx]

        return (
            idx,
            torch.tensor(example["x"]),
            torch.tensor(example["x_len"]),
            torch.tensor(example["y_input"]),
            torch.tensor(example["y_input_len"]),
            torch.tensor(example["y_target"]),
            torch.tensor(example["y_target_len"]),
        )

    def collate_fn(self, batch):
        bidx, bx, bx_len, by_input, by_input_len, by_target, by_target_len = zip(*batch)

        bx = torch.nn.utils.rnn.pad_sequence(bx, batch_first=True, padding_value=0)
        bx_len = torch.tensor(bx_len)
        by_input = torch.nn.utils.rnn.pad_sequence(by_input, batch_first=True, padding_value=0)
        by_input_len = torch.tensor(by_input_len)
        by_target = torch.nn.utils.rnn.pad_sequence(by_target, batch_first=True, padding_value=0)
        by_target_len = torch.tensor(by_target_len)

        return bidx, bx, bx_len, by_input, by_input_len, by_target, by_target_len
