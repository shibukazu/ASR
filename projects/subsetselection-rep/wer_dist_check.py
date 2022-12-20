from torch.utils.data import Dataset

import torch
from torch import nn, optim
import pandas as pd
import torchaudio
import librosa
import numpy as np
import math

from torchmetrics.functional import char_error_rate, word_error_rate

import glob
import os
import re
import copy
import pickle
import math
tkwargs_int = {
    "dtype": torch.int32,
    "device": "cuda",
}
tkwargs_float = {
    "dtype": torch.float32,
    "device": "cuda",
}
# 事前に行ったFineTuningにおけるWERからサブセットセレクションに用いる平均WERを計算する
from collections import defaultdict
average_wers = defaultdict(list)
scores = []
WER_CALCULATION_EPOCH = 30
TOTAL_RUN = 10
for run in range(TOTAL_RUN):
    cpt = torch.load(f"cpts/timit_finetune_checkpoint_{run}_{WER_CALCULATION_EPOCH-1}.pt")
    wers = cpt["input_to_wer"]
    teachers = cpt["input_to_teacher"]
    for idx, wer in wers.items():
        average_wers[idx].append(wer.item())
for idx, v in average_wers.items():
    scores.append((idx, np.mean(v)))
with open("timit_wer_scores.bin", "wb") as f:
    pickle.dump(scores, f)

import copy
import random
def cowerage_sampler(dataset, scores, retain, other_size):
    asr_dataset = copy.deepcopy(dataset)
    sorted_scores = sorted(scores, key=lambda x: x[1], reverse=True)
    highest_wer = sorted_scores[0][1]
    lowest_wer = sorted_scores[-1][1]
    num_buckets = int(len(sorted_scores) / 10)

    buckets = []
    for i in range(num_buckets):
        bucket = []
        # WERの範囲をnum_buckets等分する
        # 等分であるため、一つもデータが含まれないbucketも存在する
        lower_limit = lowest_wer + (
            (i - 1) * (highest_wer - lowest_wer) / num_buckets
        )
        upper_limit = lowest_wer + ((i) * (highest_wer - lowest_wer) / num_buckets)
        for score in sorted_scores:
            if score[1] >= lower_limit and score[1] < upper_limit:
                bucket.append(score)
        buckets.append(bucket)
    selected_scores_tmp = []
    counter = 0
    for bucket in buckets:
        sampled = random.sample(bucket, math.ceil(retain * len(bucket)))
        selected_scores_tmp.append(sampled)
        counter += len(sampled)
    # 他のサンプリング方法以下にする
    while counter > other_size:
        for selected_score_tmp in selected_scores_tmp:
            if counter == other_size:
                break
            else:
                if len(selected_score_tmp) > 0:
                    selected_score_tmp.pop()
                    counter -= 1
    selected_scores = []
    for selected_score_tmp in selected_scores_tmp:
        selected_scores.extend(selected_score_tmp)
    selected_idxs = [selected_score[0] for selected_score in selected_scores]
    selected_wers = [selected_score[1] for selected_score in selected_scores]
    asr_dataset["train"] = asr_dataset["train"].filter(
            lambda example, index: index in selected_idxs,
            with_indices=True,
            num_proc=16,
    )
    return asr_dataset, selected_wers

def random_sampler(dataset, scores, retain):
    asr_dataset = copy.deepcopy(dataset)
    select_size = int(retain * len(asr_dataset["train"]))
    asr_dataset["train"] = asr_dataset["train"].shuffle()
    asr_dataset["train"] = asr_dataset["train"].filter(
            lambda example, index: index < select_size, with_indices=True, num_proc=16,
    )
    return asr_dataset

def top_k_sampler(dataset, scores, retain):
    asr_dataset = copy.deepcopy(dataset)
    sorted_scores = sorted(scores, key=lambda x: x[1], reverse=True)
    boundary = int(len(sorted_scores) * retain)
    selected_scores = sorted_scores[0:boundary]
    selected_idxs = [selected_score[0] for selected_score in selected_scores]
    selected_wers = [selected_score[1] for selected_score in selected_scores]
    asr_dataset["train"] = asr_dataset["train"].filter(
            lambda example, index: index in selected_idxs,
            with_indices=True,
            num_proc=16,
    )
    return asr_dataset, selected_wers

def bottom_k_sampler(dataset, scores, retain):
    asr_dataset = copy.deepcopy(dataset)
    sorted_scores = sorted(scores, key=lambda x: x[1], reverse=False)
    boundary = int(len(sorted_scores) * retain)
    selected_scores = sorted_scores[0:boundary]
    selected_idxs = [selected_score[0] for selected_score in selected_scores]
    selected_wers = [selected_score[1] for selected_score in selected_scores]
    asr_dataset["train"] = asr_dataset["train"].filter(
            lambda example, index: index in selected_idxs,
            with_indices=True,
            num_proc=16,
    )
    return asr_dataset, selected_wers

from datasets import load_dataset
base_dataset = load_dataset("../../datasets/loading_scripts/timit.py", data_dir="../../datasets/TIMIT/")
print(f"base_dataset size: {len(base_dataset['train'])}")
RETAIN = 0.6
random_selected_dataset = random_sampler(base_dataset, scores, RETAIN)
top_k_selected_dataset = top_k_sampler(base_dataset, scores, RETAIN)
bottom_k_selected_dataset = bottom_k_sampler(base_dataset, scores, RETAIN)
cowerage_selected_dataset = cowerage_sampler(base_dataset, scores, RETAIN, other_size=len(bottom_k_selected_dataset["train"]))

