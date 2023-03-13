import math
import os
from logging import config, getLogger

import hydra
import mlflow
import torch
from conf import logging_conf
from data import LibriSpeechTextDataset, get_text_dataloader
from hydra.core.hydra_config import HydraConfig
from lm_model import LSTMLM
from omegaconf import DictConfig
from rich.logging import RichHandler
from tokenizer import SentencePieceTokenizer
from torchmetrics.functional.text.perplexity import perplexity
from tqdm import tqdm
from util.mlflow import log_params_from_omegaconf_dict


class DataParallel(torch.nn.DataParallel):
    def __getattr__(self, name):
        try:
            return super().__getattr__(name)
        except AttributeError:
            return getattr(self.module, name)


DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

"""
Memo:
RNN-Tに合わせてBOSトークンのみ追加する
"""


@hydra.main(version_base=None, config_path="conf/lm", config_name=None)
def main(cfg: DictConfig):
    CONF_NAME = HydraConfig.get().job.config_name
    EXPERIMENT_NAME = CONF_NAME
    ARTIFACT_LOCATION = f"./artifacts/{EXPERIMENT_NAME}"
    experiment = mlflow.get_experiment_by_name(EXPERIMENT_NAME)
    if experiment is None:
        experiment_id = mlflow.create_experiment(name=EXPERIMENT_NAME, artifact_location=ARTIFACT_LOCATION)
    else:
        experiment_id = experiment.experiment_id

    torch.backends.cudnn.benchmark = False
    with mlflow.start_run(experiment_id=experiment_id) as mlflow_run:
        LOG_DIR = mlflow_run.info.artifact_uri
        config.dictConfig(logging_conf.config_generator(LOG_DIR))
        logger = getLogger()
        logger.handlers[0] = RichHandler(markup=True)  # pretty formatting
        # save parameters from hydra to mlflow
        log_params_from_omegaconf_dict(cfg)

        tokenizer = SentencePieceTokenizer(
            model_file_path=cfg.tokenizer.model_file_path,
        )
        vocab_size = tokenizer.num_tokens

        if cfg.dataset.name == "Librispeech":
            train_dataset = LibriSpeechTextDataset(
                json_file_path=cfg.dataset.train.json_file_path,
                tokenizer=tokenizer,
            )
            dev_dataset = LibriSpeechTextDataset(
                json_file_path=cfg.dataset.dev.json_file_path,
                tokenizer=tokenizer,
            )
        else:
            raise NotImplementedError

        train_dataloader = get_text_dataloader(
            dataset=train_dataset,
            batch_text_len=cfg.train.batch_text_len,
            num_workers=12,
            pin_memory=True,
            pad_idx=tokenizer.pad_token_id,
        )

        dev_dataloader = get_text_dataloader(
            dataset=dev_dataset,
            batch_text_len=cfg.train.batch_text_len,
            num_workers=12,
            pin_memory=True,
            pad_idx=tokenizer.pad_token_id,
        )

        model_args = {
            "vocab_size": vocab_size,
            "embed_dim": cfg.model.embed_dim,
            "hidden_dim": cfg.model.hidden_dim,
            "num_layers": cfg.model.num_layers,
            "dropout": cfg.model.dropout,
        }

        language_model = LSTMLM(**model_args)
        language_model = DataParallel(language_model).to(DEVICE)
        optimizer = torch.optim.Adam(
            language_model.parameters(),
            lr=cfg.train.optimize.lr,
            weight_decay=cfg.train.optimize.weight_decay,
            betas=(cfg.train.optimize.beta1, cfg.train.optimize.beta2),
            eps=cfg.train.optimize.eps,
        )

        criterion = torch.nn.CrossEntropyLoss(ignore_index=tokenizer.pad_token_id, reduction="mean")

        NUM_EPOCH = cfg.train.num_epoch

        for i in range(1, NUM_EPOCH + 1):
            bar = tqdm(total=len(train_dataset))
            bar.set_description(f"Train Epoch {i}  ")

            torch.cuda.empty_cache()
            language_model.train()
            epoch_train_loss = 0
            epoch_train_perplexity = 0

            for _, btext, btext_len in train_dataloader:
                btext = btext.to(DEVICE)
                btext_len = btext_len

                # RNN-Tと同様にBOSトークンを追加する
                binput = torch.nn.functional.pad(btext, (1, 0), value=tokenizer.bos_token_id)[:, :-1]
                binput_len = btext_len
                boutput = language_model(binput, binput_len)

                bteacher = btext
                loss = criterion(boutput.transpose(1, 2), bteacher)
                loss.backward()

                grad_norm = torch.nn.utils.clip_grad_norm_(
                    language_model.parameters(), cfg.train.optimize.max_grad_norm
                )
                if math.isnan(grad_norm):
                    logger.error("grad norm is nan. Do not update model.")
                    logger.error(f"loss value: {loss.item()}")
                else:
                    optimizer.step()
                    optimizer.zero_grad()

                perp = torch.exp(loss)
                # bprob = torch.nn.functional.softmax(boutput, dim=-1)
                # perp = perplexity(bprob, bteacher, ignore_index=tokenizer.pad_token_id)

                epoch_train_loss += loss.item() * btext.shape[0]
                epoch_train_perplexity += perp.item() * btext.shape[0]
                bar.set_postfix(
                    {
                        "loss": loss.item(),
                        "perplexity": perp.item(),
                    }
                )
                bar.update(btext.shape[0])

            logger.info(f"Train Loss: {epoch_train_loss / len(train_dataset)}")
            logger.info(f"Train Perplexity: {math.exp(epoch_train_loss / len(train_dataset))}")
            mlflow.log_metric("train_loss", epoch_train_loss / len(train_dataset))
            mlflow.log_metric("train_perplexity", math.exp(epoch_train_loss / len(train_dataset)))

            bar = tqdm(total=len(dev_dataset))
            bar.set_description(f"Dev Epoch {i}  ")

            torch.cuda.empty_cache()
            language_model.eval()
            epoch_dev_loss = 0
            epoch_dev_perplexity = 0

            with torch.no_grad():
                for _, btext, btext_len in dev_dataloader:
                    btext = btext.to(DEVICE)
                    btext_len = btext_len

                    # RNN-Tと同様にBOSトークンを追加する
                    binput = torch.nn.functional.pad(btext, (1, 0), value=tokenizer.bos_token_id)[:, :-1]
                    binput_len = btext_len
                    boutput = language_model(binput, binput_len)

                    bteacher = btext
                    loss = criterion(boutput.transpose(1, 2), bteacher)

                    perp = torch.exp(loss)
                    # bprob = torch.nn.functional.softmax(boutput, dim=-1)
                    # perp = perplexity(bprob, bteacher, ignore_index=tokenizer.pad_token_id)
                    epoch_dev_loss += loss.item() * btext.shape[0]
                    epoch_dev_perplexity += perp.item() * btext.shape[0]
                    bar.update(btext.shape[0])

            logger.info(f"Dev Loss: {epoch_dev_loss / len(dev_dataset)}")
            logger.info(f"Dev Perplexity: {math.exp(epoch_dev_loss / len(dev_dataset))}")
            mlflow.log_metric("dev_loss", epoch_dev_loss / len(dev_dataset))
            mlflow.log_metric("dev_perplexity", math.exp(epoch_dev_loss / len(dev_dataset)))

            torch.save(
                {
                    "model_args": model_args,
                    "model": language_model.module.state_dict(),
                    "optimizer": optimizer.state_dict(),
                    "dev_loss": epoch_dev_loss / len(dev_dataset),
                    "dev_perplexity": epoch_dev_perplexity / len(dev_dataset),
                },
                os.path.join(mlflow_run.info.artifact_uri, f"model_{i}.pth"),
            )


if __name__ == "__main__":
    main()
