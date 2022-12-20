import re
import copy
import numpy as np

from transformers import Wav2Vec2Processor
from datasets import load_dataset, load_metric
from collections import defaultdict
from transformers import Wav2Vec2ForCTC
import json


import copy
import numpy as np
import glob
from collections import defaultdict
import random
import pickle

from utilities import *


class PruningStrategies:
    RANDOM = "RANDOM"
    TOP_K = "TOP_K"
    BOTTOM_K = "BOTTOM_K"
    COWERAGE = "COWERAGE"


class Datasets:
    TIMIT = "timit"
    LJSPEECH = "ljspeech"
    LIBRISPEECH = "librispeech"


# Training parameters
selected_dataset = Datasets.TIMIT
EPOCH = 8
SELECTED_STRATEGY = PruningStrategies.TOP_K
WITHIN_BUCKET_STRATEGY = PruningStrategies.RANDOM
record_test_wer_for_individual_examples = False


with open(
    "averaged_training_scores_{}.pickle".format(selected_dataset), "rb"
) as handle:
    inputs_to_wer = pickle.load(handle)

if selected_dataset == Datasets.TIMIT:
    full_dataset = load_dataset("timit_asr")
elif selected_dataset == Dataset.LJSPEECH:
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


test_sentences = []
for x in full_dataset["test"]:
    test_sentences.append(x["text"])

# 全データによるFine-Tuningの任意のエポックにおける平均WERをもとにサンプリングする
epoch_scores = []
for k, v in inputs_to_wer.items():
    epoch_scores.append((k, v[EPOCH - 1]))


def chunks(lst, n):
    """Yield successive n-sized chunks from lst."""
    for i in range(0, len(lst), n):
        yield lst[i : i + n]


test_inputs_to_wer = defaultdict(list)

group_by_length = False if record_test_wer_for_individual_examples else True

wers = []
for RETAIN_PERCENTAGE in [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]:

    asr_dataset = copy.deepcopy(full_dataset)

    if SELECTED_STRATEGY == PruningStrategies.COWERAGE:
        epoch_scores_sorted = sorted(epoch_scores, key=lambda x: x[1], reverse=True)
        highest_wer = epoch_scores_sorted[0][1]
        lowest_wer = epoch_scores_sorted[-1][1]
        num_buckets = int(len(epoch_scores_sorted) / 10)

        buckets = []
        for i in range(num_buckets):
            bucket = []
            lower_limit = lowest_wer + (
                (i - 1) * (highest_wer - lowest_wer) / num_buckets
            )
            upper_limit = lowest_wer + ((i) * (highest_wer - lowest_wer) / num_buckets)
            for score in epoch_scores_sorted:
                if score[1] >= lower_limit and score[1] < upper_limit:
                    bucket.append(score)
            buckets.append(bucket)

        filtered_indices = []
        WITHIN_BUCKET_STRATEGY = PruningStrategies.RANDOM
        for bucket in buckets:
            if WITHIN_BUCKET_STRATEGY == PruningStrategies.RANDOM:
                elements = random.sample(
                    bucket, math.ceil(RETAIN_PERCENTAGE * len(bucket))
                )
            elif WITHIN_BUCKET_STRATEGY == PruningStrategies.BOTTOM_K:
                elements = bucket[math.ceil((1 - RETAIN_PERCENTAGE) * len(bucket)) :]
            elif WITHIN_BUCKET_STRATEGY == PruningStrategies.TOP_K:
                elements = bucket[: math.ceil(RETAIN_PERCENTAGE * len(bucket))]
            filtered_indices.extend(elements)

        retain_indices = {x[0]: x[1] for x in filtered_indices}
        asr_dataset["train"] = asr_dataset["train"].filter(
            lambda example, index: index in retain_indices,
            with_indices=True,
            num_proc=16,
        )

    elif SELECTED_STRATEGY == PruningStrategies.RANDOM:
        pruned_size = int(RETAIN_PERCENTAGE * len(asr_dataset["train"]))
        asr_dataset["train"] = asr_dataset["train"].shuffle()
        asr_dataset["train"] = asr_dataset["train"].filter(
            lambda example, index: index < pruned_size, with_indices=True
        )

    elif SELECTED_STRATEGY == PruningStrategies.TOP_K:
        epoch_scores_sorted = sorted(epoch_scores, key=lambda x: x[1], reverse=True)
        OFFSET = 0
        index = int(len(epoch_scores_sorted) * RETAIN_PERCENTAGE) + OFFSET
        filtered_indices = epoch_scores_sorted[OFFSET:index]
        retain_indices = {x[0]: x[1] for x in filtered_indices}
        asr_dataset["train"] = asr_dataset["train"].filter(
            lambda example, index: index in retain_indices, with_indices=True
        )

    elif SELECTED_STRATEGY == PruningStrategies.BOTTOM_K:
        epoch_scores_sorted = sorted(epoch_scores, key=lambda x: x[1], reverse=False)
        OFFSET = 0
        index = int(len(epoch_scores_sorted) * RETAIN_PERCENTAGE) + OFFSET
        filtered_indices = epoch_scores_sorted[OFFSET:index]
        retain_indices = {x[0]: x[1] for x in filtered_indices}
        asr_dataset["train"] = asr_dataset["train"].filter(
            lambda example, index: index in retain_indices, with_indices=True
        )

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

    import json

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
        # we do not want to group tokens when computing the metrics
        label_str = processor.batch_decode(pred.label_ids, group_tokens=False)
        wer = wer_metric.compute(predictions=pred_str, references=label_str)

        if record_test_wer_for_individual_examples:
            print("Recording")
            i = 0
            for sentence, prediction in zip(label_str, pred_str):
                try:
                    temp_wer = wer_metric.compute(
                        predictions=[prediction], references=[sentence]
                    )
                    test_inputs_to_wer[i].append(temp_wer)
                except:
                    test_inputs_to_wer[i].append(1)
                i += 1

        return {"wer": wer}

    def prepare_dataset(batch):
        audio = batch["audio"]

        # batched output is "un-batched" to ensure mapping is correct
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

    directory = "pruning_experiments"
    training_args = TrainingArguments(
        output_dir=directory,
        group_by_length=group_by_length,
        per_device_train_batch_size=32,
        evaluation_strategy="epoch",
        num_train_epochs=20,
        fp16=True,
        gradient_checkpointing=True,
        save_steps=1000,
        eval_steps=3000,
        logging_steps=500,
        learning_rate=1e-4,
        weight_decay=0.005,
        warmup_steps=1000,
        save_total_limit=1,
        push_to_hub=False,
        dataloader_num_workers=16,
    )

    trainer2 = Trainer(
        model=model,
        data_collator=data_collator,
        args=training_args,
        compute_metrics=compute_metrics_and_store_wer,
        train_dataset=asr_dataset["train"],
        eval_dataset=asr_dataset["test"],
        tokenizer=processor.feature_extractor,
    )

    trainer2.train()

    import copy
    import torch
    import numpy as np

    from transformers import Wav2Vec2Processor
    from datasets import load_dataset, load_metric
    from collections import defaultdict
    from transformers import Wav2Vec2ForCTC

    def map_to_result(batch):
        with torch.no_grad():
            input_values = torch.tensor(batch["input_values"]).cuda().unsqueeze(0)
            logits = model(input_values).logits

            pred_ids = torch.argmax(logits, dim=-1)
            batch["pred_str"] = processor.batch_decode(pred_ids)[0]
            batch["text"] = processor.decode(batch["labels"], group_tokens=False)

            return batch

    model = model.to("cuda")
    results = asr_dataset["test"].map(
        map_to_result, remove_columns=asr_dataset["test"].column_names
    )
    wer = wer_metric.compute(predictions=results["pred_str"], references=test_sentences)
    wers.append(wer)
    print("Percentage: ", RETAIN_PERCENTAGE)
    print("Test WER: {:.3f}".format(wer))
    with open("{0}_{1}.txt".format(SELECTED_STRATEGY, RETAIN_PERCENTAGE), "w") as f:
        f.write("%f" % wer)


if record_test_wer_for_individual_examples:
    with open("test_scores.pickle", "wb") as handle:
        pickle.dump(test_inputs_to_wer, handle, protocol=4)

