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


CONF_NAME = "template"


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

    # 同一のマイク内での平均KL距離
    mic_names = ["matrix", "nexus6", "pseye", "respeaker", "shure", "usb"]
    mic_kl_divergences = {}
    for mic_name in mic_names:
        logger.info(f"mic: {mic_name}")
        f_name = f"pickles/{mic_name}_quantized_indices.pkl"
        with open(f_name, "rb") as f:
            quantized_indices = pickle.load(f)

        selected_quantized_indices = []
        for key in sampled_keys:
            selected_quantized_indices.append(quantized_indices[key])
        kl_divergence = calculate_average_kl_divergence(selected_quantized_indices, max_index, logger)
        mic_kl_divergences[mic_name] = kl_divergence
        logger.info(f"kl_divergence: {kl_divergence}")

    # 全体の平均KL距離
    logger.info(f"average kl_divergence: {np.mean(list(mic_kl_divergences.values()))}")
    mlflow.log_metric(key="Average KL Divergence_mic", value=np.mean(list(mic_kl_divergences.values())))
    mic_kl_divergences_file_name = "outputs/mic_kl_divergences.pkl"
    with open(mic_kl_divergences_file_name, "wb") as f:
        pickle.dump(mic_kl_divergences, f)
    mlflow.log_artifact(mic_kl_divergences_file_name)

    # 同一の発話内での平均KL距離
    utterance_kl_divergences = {}
    for idx, utterance_key in enumerate(sampled_keys):
        logger.info(f"utterance_key: {utterance_key}")
        logger.info(f"progress: {idx}, {idx / len(sampled_keys) * 100:.2f}%")
        selected_quantized_indices = []
        for mic_name in mic_names:
            f_name = f"pickles/{mic_name}_quantized_indices.pkl"
            with open(f_name, "rb") as f:
                quantized_indices = pickle.load(f)
            selected_quantized_indices.append(quantized_indices[utterance_key])
        kl_divergence = calculate_average_kl_divergence(selected_quantized_indices, max_index, logger)
        utterance_kl_divergences[utterance_key] = kl_divergence
        logger.info(f"kl_divergence: {kl_divergence}")

    # 全体の平均KL距離
    logger.info(f"average kl_divergence: {np.mean(list(utterance_kl_divergences.values()))}")
    mlflow.log_metric(key="Average KL Divergence_utterance", value=np.mean(list(utterance_kl_divergences.values())))
    utterance_kl_divergences_file_name = "outputs/utterance_kl_divergences.pkl"
    with open(utterance_kl_divergences_file_name, "wb") as f:
        pickle.dump(utterance_kl_divergences, f)
    mlflow.log_artifact(utterance_kl_divergences_file_name)


if __name__ == "__main__":
    main()
