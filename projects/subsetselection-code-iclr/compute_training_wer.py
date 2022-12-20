import re
import copy
import numpy as np

from transformers import Wav2Vec2Processor
from datasets import load_dataset, load_metric
from collections import defaultdict
from transformers import Wav2Vec2ForCTC
import json
from utilities import *


inputs_to_wer = defaultdict(list)


class Datasets:
    TIMIT = "timit"
    LJSPEECH = "ljspeech"
    LIBRISPEECH = "librispeech"


# Training parameters
selected_dataset = Datasets.TIMIT


if selected_dataset == Datasets.TIMIT:
    full_dataset = load_dataset("timit_asr")
elif selected_dataset == Datasets.LJSPEECH:
    full_dataset = load_dataset("lj_speech")
    full_dataset = full_dataset.cast_column("audio", Audio(sampling_rate=16_000))
elif selected_dataset == Datasets.LIBRISPEECH:
    full_dataset = load_dataset("librispeech_asr", "clean")
    ten_hours = int(0.1 * len(full_dataset["train.100"]))
    full_dataset["train"] = full_dataset["train.100"].filter(
        lambda example, index: index < ten_hours, with_indices=True, num_proc=4
    )
    del full_dataset["train.360"]
    del full_dataset["train.100"]


def train_and_record_wer(run_id):
    asr_dataset = copy.deepcopy(full_dataset)
    asr_dataset = asr_dataset.remove_columns(["id"])
    asr_dataset = asr_dataset.map(remove_special_characters)
    vocabs = asr_dataset.map(
        extract_all_chars,
        batched=True,
        batch_size=-1,
        keep_in_memory=True,
        remove_columns=asr_dataset.column_names["train"],
    )
    vocab_list = list(
        set(vocabs["train"]["vocab"][0]) | set(vocabs["test"]["vocab"][0])
    )
    vocab_dict = {v: k for k, v in enumerate(vocab_list)}
    vocab_dict["|"] = vocab_dict[" "]
    del vocab_dict[" "]
    vocab_dict["[UNK]"] = len(vocab_dict)
    vocab_dict["[PAD]"] = len(vocab_dict)

    with open("vocab.json", "w") as vocab_file:
        json.dump(vocab_dict, vocab_file)

    from transformers import Wav2Vec2CTCTokenizer

    tokenizer = Wav2Vec2CTCTokenizer(
        "./vocab.json", unk_token="[UNK]", pad_token="[PAD]", word_delimiter_token="|"
    )
    from transformers import Wav2Vec2FeatureExtractor

    feature_extractor = Wav2Vec2FeatureExtractor(
        feature_size=1,
        sampling_rate=16000,
        padding_value=0.0,
        do_normalize=True,
        return_attention_mask=False,
    )
    processor = Wav2Vec2Processor(
        feature_extractor=feature_extractor, tokenizer=tokenizer
    )

    def compute_metrics_and_store_wer(pred):
        pred_logits = pred.predictions
        pred_ids = np.argmax(pred_logits, axis=-1)

        pred.label_ids[pred.label_ids == -100] = processor.tokenizer.pad_token_id

        pred_str = processor.batch_decode(pred_ids)
        label_str = processor.batch_decode(pred.label_ids, group_tokens=False)

        wer = wer_metric.compute(predictions=pred_str, references=label_str)

        i = 0
        for sentence, prediction in zip(label_str, pred_str):
            try:
                temp_wer = wer_metric.compute(
                    predictions=[prediction], references=[sentence]
                )
                inputs_to_wer[i].append(temp_wer)
            except:
                inputs_to_wer[i].append(1)
            i += 1

        return {"wer": wer}

    def prepare_dataset(batch):
        audio = batch["audio"]
        batch["input_values"] = processor(
            audio["array"], sampling_rate=audio["sampling_rate"]
        ).input_values[0]
        batch["input_length"] = len(batch["input_values"])

        with processor.as_target_processor():
            batch["labels"] = processor(batch["text"]).input_ids
        return batch

    asr_dataset = asr_dataset.map(
        prepare_dataset, remove_columns=asr_dataset.column_names["train"], num_proc=4
    )
    data_collator = DataCollatorCTCWithPadding(processor=processor, padding=True)

    model = Wav2Vec2ForCTC.from_pretrained(
        "facebook/wav2vec2-base",
        ctc_loss_reduction="mean",
        pad_token_id=processor.tokenizer.pad_token_id,
    )

    from transformers import Trainer
    from transformers import TrainingArguments

    directory = "train-" + str(run_id) + "-full"
    training_args = TrainingArguments(
        output_dir=directory,
        group_by_length=False,
        per_device_train_batch_size=8,
        per_device_eval_batch_size=8,
        evaluation_strategy="epoch",
        num_train_epochs=20,
        fp16=True,
        gradient_checkpointing=True,
        save_steps=1000,
        logging_steps=500,
        learning_rate=1e-4,
        weight_decay=0.005,
        warmup_steps=1000,
        save_total_limit=1,
        push_to_hub=False,
        dataloader_num_workers=16,
    )

    trainer = Trainer(
        model=model,
        data_collator=data_collator,
        args=training_args,
        compute_metrics=compute_metrics_and_store_wer,
        train_dataset=asr_dataset["train"],
        eval_dataset=asr_dataset["train"],
        tokenizer=processor.feature_extractor,
    )
    trainer.train()

    import pickle

    with open("training_scores_wer_run_{0}.pickle".format(run_id), "wb") as handle:
        pickle.dump(inputs_to_wer, handle, protocol=4)


TOTAL_RUNS = 10
for run_id in range(1, TOTAL_RUNS + 1):
    train_and_record_wer(run_id)


"""Compute average of all the runs"""
all_scores = []
for i in range(1, TOTAL_RUNS):
    with open("training_scores_wer_run_{0}.pickle".format(i), "rb") as handle:
        scores = pickle.load(handle)
        all_scores.append(scores)


from collections import defaultdict
from operator import add

sum_of_scores = all_scores[0]
for score in all_scores[1:]:
    for k, v in score.items():
        sum_of_scores[k] = list(map(add, sum_of_scores[k], score[k]))

for k, v in sum_of_scores.items():
    for i in range(len(sum_of_scores[k])):
        sum_of_scores[k][i] /= TOTAL_RUNS

averaged_scores = sum_of_scores

import pickle

with open(
    "averaged_training_scores_{}.pickle", format(selected_dataset), "wb"
) as handle:
    pickle.dump(averaged_scores, handle, protocol=4)

