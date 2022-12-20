from logging import config, getLogger

import hydra
import mlflow
import torch
from conf import logging_conf
from omegaconf import DictConfig
from rich.logging import RichHandler
from transformers import AutoFeatureExtractor, Wav2Vec2ConformerForPreTraining
from util.mlflow import log_params_from_omegaconf_dict
from data import TIMITDataset
CONF_NAME = ""


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

    device = torch.device(f"cuda:{cfg.train.cuda}" if torch.cuda.is_available() else "cpu")
    logger.info(f"Training on {device}.")

    dataset = TIMITDataset(is_train=True, vocab_file_path="vocab.json")
    # train_dataloader = None
    # test_dataloader = None

    # model = None
    # optimizer = None
    # scheduler = None

    NUM_EPOCH = cfg.train.num_epoch
    for i in range(1, NUM_EPOCH + 1):
        logger.info(f"Epoch {i} has started.")
        # train

        # test


if __name__ == "__main__":
    main()
