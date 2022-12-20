import random
import sys
import time

sys.path.append("..")
import json
from typing import List

import numpy as np
import torch
from torch import nn, optim
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import DataLoader, Dataset
from torchaudio.functional import resample
from torchmetrics.functional import char_error_rate, word_error_rate
from transformers import Wav2Vec2CTCTokenizer

from datasets import load_dataset

tkwargs_int = {
    "dtype": torch.int32,
    "device": "cuda",
}
tkwargs_float = {
    "dtype": torch.float32,
    "device": "cuda",
}


class TIMITDatasetWav(Dataset):
    def __init__(self, vocab_file_path: str, resample_rate: int = 16000, is_train: bool = False):

        self.type = "train" if is_train else "test"
        self.resample_rate = resample_rate

        dataset = load_dataset(
            "../../datasets/loading_scripts/timit.py",
            data_dir="../../datasets/TIMIT/",
            cache_dir="/home/shibutani/fs/.cache/huggingface/datasets")
        dataset = dataset.remove_columns(["id"])

        self.extract_vocab(all_text=dataset["train"]["text"], vocab_file_path=vocab_file_path)
    
        self.tokenizer = Wav2Vec2CTCTokenizer(
            vocab_file_path, unk_token="[UNK]", pad_token="[PAD]", word_delimiter_token="|"
        )
        
        def prepare_dataset(example):
            audio = example["audio"]["array"].flatten()
            orig_freq = example["audio"]["sampling_rate"]
            x = resample(audio, orig_freq=orig_freq, new_freq=self.resample_rate)
            mean = np.mean(x)
            std = np.std(x)
            z = ((x - mean) / std)
            assert z.shape == x.shape, f"標準化後の形が異なります。"
            assert z.ndim == 1, f"標準化後の次元が不正です。"
            assert abs(np.mean(z) - 0) < 1, f"標準化後の平均値が不正です。"
            assert abs(np.std(z) - 1) < 1, f"標準化後の標準偏差が不正です。"
            example["input_values"] = z
            example["input_length"] = len(example["input_values"])
            example["labels"] = self.tokenizer(example["text"]).input_ids
            return example

        self.dataset = dataset.map(
            prepare_dataset, num_proc=4
        )

        with open(vocab_file_path, "r") as vocab_file:
            vocab = json.load(vocab_file)
        self.vocab = vocab
        self.pad_token_id = vocab["[PAD]"]
        self.unk_token_id = vocab["[UNK]"]
        self.ctc_token_id = vocab["_"]

        self.input_to_text = {}
        for idx in range(len(self.dataset[self.type])):
            self.input_to_text[idx] = self.dataset[self.type][idx]["text"]
        

    def __len__(self):
        return len(self.dataset[self.type])
    
    def __getitem__(self, idx):

        return  idx, torch.tensor(self.dataset[self.type][idx]["input_values"]), torch.tensor(self.dataset[self.type][idx]["labels"]) 
    
    def collate_fn(self, batch):
        idxs, wavs, text_idxs = zip(*batch)
        original_wav_lens = torch.tensor(np.array([len(wav) for wav in wavs]))
        original_text_idx_lens = torch.tensor(np.array([len(text_idx) for text_idx in text_idxs]))
        # padding for spectrogram_db
        padded_wavs = []
        for wav in wavs:
            padded_wav = np.pad(wav, ((0, max(original_wav_lens)-wav.shape[0])), "constant", constant_values=0)
            padded_wavs.append(padded_wav)
        
        padded_wavs = torch.tensor(np.array(padded_wavs))

        # padding and packing for text_idx
        padded_text_idxs = pad_sequence(text_idxs, batch_first=True, padding_value=self.pad_token_id)

        return idxs, padded_wavs, padded_text_idxs, original_wav_lens, original_text_idx_lens

    def extract_vocab(
        self,
        all_text: List = None, 
        vocab_file_path: str = "./full_vocab.json",
        ) -> None:
        

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
            

dataset = TIMITDatasetWav(vocab_file_path="./timit_full_vocab.json", resample_rate=16000, is_train=True)
print(f"dataset size: {len(dataset)}")

BATCH_SIZE = 8
train_dataloader = DataLoader(
    dataset,
    batch_size=BATCH_SIZE,
    shuffle=True,
    num_workers=8,
    # 不完全なバッチの無視
    drop_last=True,
    # 高速化?
    pin_memory=True,
    collate_fn=dataset.collate_fn
)


class Model(nn.Module):
    def __init__(self, nlabel):
        super(Model, self).__init__()
        self.in_size = bundle._params["encoder_embed_dim"]
        self.nlabel = nlabel
        self.wav2vec_encoder = bundle.get_model()
        self.fc = nn.Linear(self.in_size, self.nlabel, bias=True)
        self.log_softmax = nn.functional.log_softmax
    
    def forward(self, x, x_lengths):
        # args:
        #   x: [B, T]
        #   x_lengths: [B]
        #       padding前のシーケンス長
        # return:
        #   log_prob: [B, T, nlabel]
        #   y_lengths: [B]
        #       非パディング部分のシーケンス長
        encoded, y_lengths = self.wav2vec_encoder.extract_features(x, x_lengths) # encoded: [L, B, T, in_size]

        y = self.fc(encoded[-1]) # [B, T', nlabel]
        
        log_probs = self.log_softmax(y, dim=2) # [B, T', nlabel]
        return log_probs, y_lengths
        

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"This learning will be running on {device}.")

num_labels = len(dataset.vocab)
num_epochs = 30

def ctc_simple_decode(hypotheses_idxs, vocab):
    # hypothesis_idxs: tensor(batch, time)
    # labels: np.array(num_labels)
    index_to_vocab = {v: k for k, v in vocab.items()}
    hypotheses_idxs = hypotheses_idxs.cpu().numpy()
    hypotheses = []
    padding_idx = vocab["[PAD]"]
    blank_idx = vocab["_"]
    separator_idx = vocab["|"]
    
    for hypothesis_idxs in hypotheses_idxs:
        hypothesis = []
        prev_idx = -1
        for idx in hypothesis_idxs:
            if idx == blank_idx:
                continue
            elif idx == prev_idx:
                continue
            elif idx == padding_idx:
                continue
            elif idx == separator_idx:
                hypothesis.append(" ")
                prev_idx = idx
            else:
                hypothesis.append(index_to_vocab[idx])
                prev_idx = idx
        hypotheses.append("".join(hypothesis))
    return hypotheses

from torch.optim.lr_scheduler import _LRScheduler


class TransformerLR(_LRScheduler):
    """TransformerLR class for adjustment of learning rate.

    The scheduling is based on the method proposed in 'Attention is All You Need'.
    """

    def __init__(self, optimizer, warmup_epochs=1000, last_epoch=-1, verbose=False):
        """Initialize class."""
        self.warmup_epochs = warmup_epochs
        self.normalize = self.warmup_epochs**0.5
        super().__init__(optimizer, last_epoch, verbose)

    def get_lr(self):
        """Return adjusted learning rate."""
        step = self.last_epoch + 1
        scale = self.normalize * min(step**-0.5, step * self.warmup_epochs**-1.5)
        return [base_lr * scale for base_lr in self.base_lrs]

for run in range(10):
    print(f"{run} th run")
    model = Model(num_labels).to(device)

    ctc_loss = nn.CTCLoss(reduction="sum", blank=dataset.ctc_token_id)
    optimizer = optim.Adam(model.parameters(), lr=0.0001, weight_decay=0.005)
    scheduler = TransformerLR(optimizer, warmup_epochs=1000)

    print("model initialized")

    for i in range(num_epochs):
        print(f"{i} th epoch")
        t0 = time.time()
        model.train()
        epoch_loss = 0
        cnt = 0
        for _, (idxs, padded_wavs, padded_text_idxs, original_wav_lens, original_text_idx_lens) in enumerate(train_dataloader):
            cnt += 1
            padded_wavs = padded_wavs.to(device)
            original_wav_lens = original_wav_lens.to(device)
            padded_text_idxs = padded_text_idxs.to(device)
            original_text_idx_lens = original_text_idx_lens.to(device)
            
            optimizer.zero_grad()
            
            log_probs, y_lengths  = model(x=padded_wavs, x_lengths=original_wav_lens)

            loss = ctc_loss(log_probs.transpose(1, 0), padded_text_idxs, y_lengths, original_text_idx_lens)
            loss.backward()
            optimizer.step()
            # lossはバッチ内平均ロス
            epoch_loss += (loss.item() / BATCH_SIZE)
        scheduler.step()
        # バッチ内平均ロスの和をイテレーション数で割ることで、一つのデータあたりの平均ロスを求める

        model.eval()
        input_to_wer = {}
        # idxが固定されるか確認する用
        input_to_teacher = {}
        with torch.no_grad():
            cnt = 0
            total_cer = 0
            total_wer = 0
            for _, (idxs, padded_wavs, padded_text_idxs, original_wav_lens, original_text_idx_lens) in enumerate(train_dataloader):
                cnt += 1
                padded_wavs = padded_wavs.to(device)
                original_wav_lens = original_wav_lens.to(device)
                padded_text_idxs = padded_text_idxs.to(device)
                original_text_idx_lens = original_text_idx_lens.to(device)
                
                log_probs, y_lengths  = model(x=padded_wavs, x_lengths=original_wav_lens)
                # for CER calculation
                hypotheses_idxs = log_probs.argmax(dim=2) 
                hypotheses = ctc_simple_decode(hypotheses_idxs, dataset.vocab)
                teachers = ctc_simple_decode(padded_text_idxs, dataset.vocab)
                total_cer += char_error_rate(hypotheses, teachers)
                batch_wer = 0
                for idx, hypothesis, teacher in zip(idxs, hypotheses, teachers):
                    input_to_wer[idx] = word_error_rate(hypothesis, teacher)
                    batch_wer += input_to_wer[idx]
                total_wer += batch_wer / len(idxs)


        t1 = time.time()
        print(f"{i} epoch: {epoch_loss / cnt} loss,  CER: {total_cer / cnt}, WER: {total_wer / cnt}, {t1 - t0} sec")

        checkpoint = {
            "model": model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "scheduler": scheduler.state_dict(),
            "random": random.getstate(),
            "np_random": np.random.get_state(), # numpy.randomを使用する場合は必要
            "torch": torch.get_rng_state(),
            "torch_random": torch.random.get_rng_state(),
            "cuda_random": torch.cuda.get_rng_state(), # gpuを使用する場合は必要
            "cuda_random_all": torch.cuda.get_rng_state_all(), # 複数gpuを使用する場合は必要
            "input_to_wer": input_to_wer,
            "input_to_text": dataset.input_to_text,
        }
        if (i + 1) % 2 == 0:
            torch.save(checkpoint, f"cpts/timit_finetune_checkpoint_{run}_{i}_{total_wer / cnt:.3f}.pt")
            print(f"checkpoint saved to cpts/timit_finetune_checkpoint_{run}_{i}_{total_wer / cnt:.3f}.pt")