import os
import pickle
import random
from logging import config, getLogger

import hydra
import mlflow
import numpy as np
import torch
import whisper
from conf import logging_conf
from loss import CrossEntropyLoss
from omegaconf import DictConfig
from optimizer import Optimizer
from rich.logging import RichHandler
from util.mlflow import log_params_from_omegaconf_dict

CONF_NAME = "conf_1"


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

    with open("train_dataset.bin", "rb") as f:
        train_dataset = pickle.load(f)
    with open("test_dataset.bin", "rb") as f:
        test_dataset = pickle.load(f)

    train_dataloader = torch.utils.data.DataLoader(
        train_dataset, batch_size=1, num_workers=2, collate_fn=train_dataset.collate_fn, shuffle=True)
    test_dataloader = torch.utils.data.DataLoader(
        test_dataset, batch_size=1, num_workers=2, collate_fn=test_dataset.collate_fn, shuffle=True)

    model = whisper.load_model('medium').to(DEVICE)

    optimizer = Optimizer(optimizer=torch.optim.AdamW(model.parameters()), max_lr=1e-4, t1=500, t2=5000)
    criterion = CrossEntropyLoss()
    NUM_EPOCH = cfg.train.num_epoch
    NUM_CLASSES = 51865
    OPTIMIZER_UPDATE_FREQ = 32
    SAVE_FREQ = 500
    for i in range(1, NUM_EPOCH + 1):
        logger.info(f"Epoch {i} has started.")
        # train
        model.train()
        for cnt, (bidx, bx, bx_len, by_input, by_input_len, by_target, by_target_len) in enumerate(train_dataloader):
            bx = bx.to(DEVICE)

            with torch.no_grad():
                benc_out = model.encoder(bx)
            by_input = by_input.to(DEVICE)
            bdec_out = model.decoder(by_input, benc_out)

            by_target = by_target.to(DEVICE)
            by_target_len = by_target_len.to(DEVICE)
            loss = criterion(bdec_out, by_target, by_target_len, NUM_CLASSES) / OPTIMIZER_UPDATE_FREQ
            loss.backward()
            if (cnt + 1) % OPTIMIZER_UPDATE_FREQ == 0:
                # 見た感じ勾配クリッピングはしていなそう
                logger.info(f"Step: {optimizer.num_step}.")
                optimizer.step()
                optimizer.zero_grad()
                logger.info(f"Loss: {loss.item()}.")
                mlflow.log_metric("train loss", loss.item(), step=optimizer.num_step)

                if (optimizer.num_step + 1) % SAVE_FREQ == 0:
                    checkpoint = {
                        "model": model.state_dict(),
                        "optimizer": optimizer.optimizer.state_dict(),
                        "optimizer_step": optimizer.num_step,
                        "optimizer_lr": optimizer.lr,
                        "epoch": i,
                    }
                    CPT_SAVE_DIR = f"cpts/{EXPERIMENT_NAME}/{mlflow_run.info.run_id}"
                    CPT_FILE_NAME = f"{CPT_SAVE_DIR}/{optimizer.num_step + 1}.pt"
                    os.makedirs(CPT_SAVE_DIR, exist_ok=True)
                    torch.save(checkpoint, CPT_FILE_NAME)
                    logger.info(f"Checkpoint has been saved to {CPT_FILE_NAME}.")
                    mlflow.log_artifact(CPT_FILE_NAME)


if __name__ == "__main__":
    main()
