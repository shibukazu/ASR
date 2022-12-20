import pickle
from logging import Logger, config, getLogger

import hydra
import matplotlib.pyplot as plt
import mlflow
import numpy as np
import pandas as pd
import torch
from conf import logging_conf
from data import LibriAdaptUS
from model import MyWav2Vec2ConformerForPreTraining
from omegaconf import DictConfig
from rich.logging import RichHandler
from transformers import BertModel, BertTokenizer
from util.mlflow import log_params_from_omegaconf_dict


def calculate_kl_divergence(indices1: np.ndarray, indices2: np.ndarray, max_index: int, logger: Logger) -> float:
    """
    indices: (num_codebooks * seq_len)
    """
    eps = 1e-8
    bins = np.linspace(0, max_index, max_index + 1)

    hist1, bin_edges1 = np.histogram(indices1, bins=bins, density=True)
    hist1 += eps
    hist1 = hist1 / (np.diff(bin_edges1) * hist1.sum())
    hist2, bin_edges2 = np.histogram(indices2, bins=bins, density=True)
    hist2 += eps
    hist2 = hist2 / (np.diff(bin_edges2) * hist2.sum())

    return np.sum(hist1 * np.log(hist1 / hist2))


CONF_NAME = "sentence-kl-relation"


@hydra.main(version_base=None, config_path="conf", config_name=CONF_NAME)
def main(cfg: DictConfig):
    """
    BERT Tokenizerによる文章類似度とKLダイバージェンスの関係を調べる
    """
    EXPERIMENT_NAME = CONF_NAME
    ARTIFACT_LOCATION = "./artifacts"
    experiment = mlflow.get_experiment_by_name(EXPERIMENT_NAME)
    if experiment is None:
        experiment_id = mlflow.create_experiment(name=EXPERIMENT_NAME, artifact_location=ARTIFACT_LOCATION)
    else:
        experiment_id = experiment.experiment_id

    with mlflow.start_run(experiment_id=experiment_id) as mlflow_run:
        LOG_DIR = mlflow_run.info.artifact_uri
    config.dictConfig(logging_conf.config_generator(LOG_DIR))
    logger = getLogger()
    logger.handlers[0] = RichHandler(markup=True)  # pretty formatting
    # save parameters from hydra to mlflow
    log_params_from_omegaconf_dict(cfg)

    DEVICE = torch.device(f"cuda:{cfg.train.cuda}" if torch.cuda.is_available() else "cpu")
    logger.info(f"Training on {DEVICE}.")

    model = MyWav2Vec2ConformerForPreTraining.from_pretrained(
        "facebook/wav2vec2-conformer-rel-pos-large",
        cache_dir="/home/shibutani/fs/.cache/huggingface/transformers"
        ).to(DEVICE)
    G = model.config.num_codevector_groups
    V = model.config.num_codevectors_per_group
    max_index = G * V

    tokenizer = BertTokenizer.from_pretrained(
        "bert-base-uncased",
        cache_dir="/home/shibutani/fs/.cache/huggingface/transformers")
    bert_model = BertModel.from_pretrained(
        "bert-base-uncased",
        cache_dir="/home/shibutani/fs/.cache/huggingface/transformers"
        ).to(DEVICE)

    metadata: pd.DataFrame = LibriAdaptUS("shure-train").metadata
    wav_filename_transcript_map = dict(zip(metadata["wav_filename"], metadata["transcript"]))
    # modify filename
    wav_filename_transcript_map = {k.split("/")[-1]: v for k, v in wav_filename_transcript_map.items()}

    with open("pickles/shure_quantized_indices.pkl", "rb") as f:
        wav_filename_quantized_indices_map = pickle.load(f)
    sampled_wav_filenames = np.random.choice(
        list(wav_filename_quantized_indices_map.keys()),
        size=len(wav_filename_quantized_indices_map) // 10,
        replace=False)

    kl_divergences = []
    cos_similarities = []
    for i in range(len(sampled_wav_filenames)):
        wav_filename1 = sampled_wav_filenames[i]
        quantized_indices1 = wav_filename_quantized_indices_map[wav_filename1]
        transcript1 = wav_filename_transcript_map[wav_filename1]

        for j in range(i + 1, len(sampled_wav_filenames)):
            wav_filename2 = sampled_wav_filenames[j]
            quantized_indices2 = wav_filename_quantized_indices_map[wav_filename2]
            transcript2 = wav_filename_transcript_map[wav_filename2]
            # calculate KL divergence
            kl_divergence = calculate_kl_divergence(quantized_indices1, quantized_indices2, max_index, logger)
            kl_divergences.append(kl_divergence)

            # calculate cosine similarity
            sentences = [transcript1, transcript2]
            encoded_dict = tokenizer(sentences, padding=True, return_tensors="pt")
            input_ids = encoded_dict["input_ids"].to(DEVICE)
            attention_mask = encoded_dict["attention_mask"].to(DEVICE)
            with torch.no_grad():
                outputs = bert_model(input_ids, attention_mask=attention_mask)
            # last_hidden_states: (2, seq_len, hidden_size)
            last_hidden_states = outputs[0]
            # 各文章内で分散表現の和を計算 (文章の長さで正規化)
            # (2, hidden_size)
            sentence_embed_vecs = (
                last_hidden_states * attention_mask.unsqueeze(-1)).sum(dim=1) / attention_mask.unsqueeze(-1).sum(dim=1)

            cos_similarity = torch.nn.functional.cosine_similarity(
                sentence_embed_vecs[0], sentence_embed_vecs[1], dim=0)

            cos_similarities.append(cos_similarity.item())

            if j % 1000 == 0:
                logger.info(f"KL divergence: {kl_divergence}, Cosine similarity: {cos_similarity}")

    # save results
    with open("pickles/kl_divergences.pkl", "wb") as f:
        pickle.dump(kl_divergences, f)
        mlflow.log_artifact(f.name)
    with open("pickles/cos_similarities.pkl", "wb") as f:
        pickle.dump(cos_similarities, f)
        mlflow.log_artifact(f.name)

    # calculate correlation
    correlation = np.corrcoef(kl_divergences, cos_similarities)[0, 1]
    logger.info(f"Correlation: {correlation}")
    mlflow.log_metric("correlation", correlation)

    # plot scatter plot
    plt.scatter(kl_divergences, cos_similarities)
    plt.xlabel("KL divergence")
    plt.ylabel("Cosine similarity")
    plt.savefig("images/sentence-kl-relation.png")
    mlflow.log_artifact("images/sentence-kl-relation.png")


if __name__ == "__main__":
    main()
