import pickle
from logging import Logger, config, getLogger
from typing import List

import hydra
import mlflow
import numpy as np
import torch
from conf import logging_conf
from model import MyWav2Vec2ConformerForPreTraining
from omegaconf import DictConfig
from rich.logging import RichHandler
from util.mlflow import log_params_from_omegaconf_dict


def calculate_average_kl_divergence(indices: List[np.ndarray], max_index: int, logger: Logger) -> float:
    """
    indices: (all_data_size, num_codebooks * seq_len)
            all_data_size: 同一の発話（マイク）のデータ数
            同一の発話内容（マイク）におけるすべてのインデックス系列
    同一の発話内容(マイク)におけるすべてのインデックス系列の平均KLダイバージェンスを計算する
    """
    total_number = len(indices) * len(indices)
    counter = 0
    eps = 1e-8
    bins = np.linspace(0, max_index, max_index + 1)
    average_kl_divergence = 0
    for i in range(len(indices)):
        for j in range(len(indices)):
            counter += 1
            if counter % 10000 == 0:
                logger.info(f"progress: {counter}, {counter / total_number * 100:.2f}%")
            hist1, bin_edges1 = np.histogram(indices[i], bins=bins, density=True)
            hist1 += eps
            hist1 = hist1 / (np.diff(bin_edges1) * hist1.sum())
            hist2, bin_edges2 = np.histogram(indices[j], bins=bins, density=True)
            hist2 += eps
            hist2 = hist2 / (np.diff(bin_edges2) * hist2.sum())

            average_kl_divergence += np.sum(hist1 * np.log(hist1 / hist2))
            if i == j:
                assert np.abs(np.sum(hist1 * np.log(hist1 / hist2)) - 0) < eps

    average_kl_divergence /= total_number
    return average_kl_divergence


CONF_NAME = "speaker"


@hydra.main(version_base=None, config_path="conf", config_name=CONF_NAME)
def main(cfg: DictConfig):
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

    model = MyWav2Vec2ConformerForPreTraining.from_pretrained("facebook/wav2vec2-conformer-rel-pos-large").to(DEVICE)

    G = model.config.num_codevector_groups
    V = model.config.num_codevectors_per_group
    max_index = G * V

    f_name = "pickles/nexus6_quantized_indices.pkl"
    with open(f_name, "rb") as f:
        matrix_quantized_indices = pickle.load(f)
    sampled_keys = np.random.choice(
        list(matrix_quantized_indices.keys()), size=len(matrix_quantized_indices) // 10, replace=False)
    # get keys for each spaker
    speaker_keys_map = {}
    for key in sampled_keys:
        speaker_id = key.split("-")[0]
        if speaker_id not in speaker_keys_map:
            speaker_keys_map[speaker_id] = []
        speaker_keys_map[speaker_id].append(key)
    logger.info(f"number of speakers: {len(speaker_keys_map)}")

    # 同一の話者内での平均KLダイバージェンスを計算する
    mic_names = ["matrix", "nexus6", "pseye", "respeaker", "shure", "usb"]
    average_kl_divergences = {}
    for i, (speaker_id, keys) in enumerate(speaker_keys_map.items()):
        logger.info(f"progress: {(i + 1) / len(speaker_keys_map) * 100}%")
        speaker_quantized_indices = []
        for mic in mic_names:
            with open(f"pickles/{mic}_quantized_indices.pkl", "rb") as f:
                mic_quantized_indices = pickle.load(f)
            speaker_quantized_indices.extend([mic_quantized_indices[key] for key in keys])
        logger.info(f"number of utterances: {len(speaker_quantized_indices)}")
        average_kl_divergence = calculate_average_kl_divergence(speaker_quantized_indices, max_index, logger)
        average_kl_divergences[speaker_id] = average_kl_divergence
        logger.info(f"speaker_id: {speaker_id}, average_kl_divergence: {average_kl_divergence}")

    with open("pickles/speaker_average_kl_divergences.pkl", "wb") as f:
        pickle.dump(average_kl_divergences, f)
    mlflow.log_artifact("pickles/speaker_average_kl_divergences.pkl")

    # calculate average kl divergence for all speakers
    average_kl_divergence = np.mean(list(average_kl_divergences.values()))
    logger.info(f"average_kl_divergence: {average_kl_divergence}")
    mlflow.log_metric("average_kl_divergence", average_kl_divergence)


if __name__ == "__main__":
    main()
