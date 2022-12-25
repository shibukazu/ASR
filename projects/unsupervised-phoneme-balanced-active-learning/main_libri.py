import os
import pickle
from logging import config, getLogger
from typing import Dict

import hydra
import mlflow
import torch
from conf import logging_conf
from data import LibriSpeechDataset, get_dataloader
from model import Model
from modules.decoders.ctc import greedy_decoder
from modules.transformers.scheduler import TransformerLR
from omegaconf import DictConfig
from rich.logging import RichHandler
from torchmetrics.functional import char_error_rate, word_error_rate
from util.mlflow import log_params_from_omegaconf_dict

CONF_NAME = "librispeech_data_selection"


class RandomSampler:
    def __init__(
        self,
        quantized_indices_memory: Dict,
        dataset: torch.utils.data.Dataset,
        ratio: float,
    ):
        self.quantized_indices_memory = quantized_indices_memory
        self.dataset = dataset
        self.ratio = ratio
        self.sampled_quantized_idx_count = torch.zeros(320 * 320, dtype=torch.float32)
        # train data全体の分布
        target_quantized_idx_count = torch.zeros(320 * 320, dtype=torch.float32)
        for idx in range(len(dataset)):
            quantized_indices = torch.tensor(self.quantized_indices_memory[idx])
            for quantized_idx in quantized_indices:
                target_quantized_idx_count[quantized_idx] += 1
        target_quantized_idx_count += 1e-8
        self.target_distribution = target_quantized_idx_count / target_quantized_idx_count.sum()

    def calculate_kl_divergence_between_target_and_empirical_distribution(self):
        """Calculate KL divergence between target and empirical distribution."""
        sample = self.sampled_quantized_idx_count
        sample += 1e-8
        sample = sample / sample.sum()
        kl_divergence = torch.sum(sample * torch.log(sample / self.target_distribution))
        mlflow.log_metric("min_data_kl_divergence", kl_divergence.item())

    def calculate_kl_divergence_between_uniform_and_empirical_distribution(self):
        """Calculate KL divergence between uniform and empirical distribution."""
        sample = self.sampled_quantized_idx_count
        sample += 1e-8
        sample = sample / sample.sum()
        uniform = torch.ones_like(sample) / len(sample)
        kl_divergence = torch.sum(sample * torch.log(sample / uniform))
        mlflow.log_metric("min_kl_divergence", kl_divergence.item())

    def sample(self):
        """Randomly sample a subset of the dataset"""
        num_samples = int(len(self.dataset) * self.ratio)
        indices = torch.randperm(len(self.dataset))[:num_samples].tolist()
        for idx in indices:
            quantized_indices = self.quantized_indices_memory[idx]
            for quantized_idx in quantized_indices:
                self.sampled_quantized_idx_count[quantized_idx] += 1
        self.calculate_kl_divergence_between_uniform_and_empirical_distribution()
        self.calculate_kl_divergence_between_target_and_empirical_distribution()

        return torch.utils.data.Subset(self.dataset, indices)


class UniformKLBasedSampler:
    def __init__(
        self,
        quantized_indices_memory: Dict,
        dataset: torch.utils.data.Dataset,
        ratio: float,
        device: torch.device,
    ):
        self.device = device
        self.dataset = dataset
        self.ratio = ratio
        self.target_num_samples = int(len(dataset) * ratio)
        self.sampled_indices = set()
        self.not_sampled_indices = set(range(len(dataset)))
        self.sampled_quantized_idx_count = torch.zeros(320 * 320, dtype=torch.float32)
        self.quantized_indices_memory = quantized_indices_memory

    def calculate_kl_divergence_between_uniform_and_empirical_distribution(self, sample):
        """Calculate KL divergence between uniform and empirical distribution.
        Args:
            sample (1D torch.Tensor): quantized indices tensor
        Returns:
            kl_divergence (float): KL divergence between uniform and empirical distribution.
        """
        assert sample.ndim == 1
        # update sampled_quantized_idx_count
        sampled_quantized_idx_count_copy = self.sampled_quantized_idx_count.clone()
        with torch.no_grad():
            for quantized_idx in sample:
                sampled_quantized_idx_count_copy[quantized_idx] += 1
        # calculate KL divergence
        sampled_quantized_idx_count_copy += 1e-8
        empirical_distribution = sampled_quantized_idx_count_copy / sampled_quantized_idx_count_copy.sum()
        uniform_distribution = torch.ones_like(empirical_distribution) / len(empirical_distribution)
        kl_divergence = torch.sum(empirical_distribution * torch.log(empirical_distribution / uniform_distribution))
        if kl_divergence < 0:
            raise ValueError("KL divergence must be positive.")
        return kl_divergence

    def sample(self):
        # sample new samples based on KL divergence until the number of samples reaches the target number of samples
        # return subset of dataset
        while len(self.sampled_indices) < self.target_num_samples:
            # show progress
            if len(self.sampled_indices) % 10 == 0:
                print(f"sampled {len(self.sampled_indices) / self.target_num_samples * 100:.2f} %")
            # calculate KL divergence for each not sampled indices
            kl_divergences = {}
            not_sampled_indices = list(self.not_sampled_indices)
            for idx in not_sampled_indices:
                quantized_indices = torch.tensor(self.quantized_indices_memory[idx])
                kl_divergence = self.calculate_kl_divergence_between_uniform_and_empirical_distribution(
                    quantized_indices
                )
                kl_divergences[idx] = kl_divergence
            # select the index with the minimum KL divergence
            min_kl_divergence_idx = min(kl_divergences.keys(), key=kl_divergences.get)
            mlflow.log_metric(
                "min_kl_divergence", kl_divergences[min_kl_divergence_idx], step=len(self.sampled_indices)
            )
            # update sampled_indices and not_sampled_indices
            self.sampled_indices.add(min_kl_divergence_idx)
            self.not_sampled_indices.remove(min_kl_divergence_idx)
            # update sampled_quantized_idx_count
            quantized_indices = self.quantized_indices_memory[min_kl_divergence_idx]
            for quantized_idx in quantized_indices:
                self.sampled_quantized_idx_count[quantized_idx] += 1

        # return subset of self.dataset
        return torch.utils.data.Subset(self.dataset, list(self.sampled_indices))


class DataKLBasedSampler:
    """
    Train Dataの分布とのKL divergenceを計算して、KL divergenceが最小になるようにサンプリングする
    """

    def __init__(
        self,
        quantized_indices_memory: Dict,
        dataset: torch.utils.data.Dataset,
        ratio: float,
        device: torch.device,
    ):
        self.device = device
        self.dataset = dataset
        self.quantized_indices_memory = quantized_indices_memory
        self.ratio = ratio
        self.target_num_samples = int(len(dataset) * ratio)
        self.sampled_indices = set()
        self.not_sampled_indices = set(range(len(dataset)))
        self.sampled_quantized_idx_count = torch.zeros(320 * 320, dtype=torch.float32)
        # train data全体の分布
        target_quantized_idx_count = torch.zeros(320 * 320, dtype=torch.float32)
        for idx in range(len(dataset)):
            quantized_indices = torch.tensor(self.quantized_indices_memory[idx])
            for quantized_idx in quantized_indices:
                target_quantized_idx_count[quantized_idx] += 1
        target_quantized_idx_count += 1e-8
        self.target_distribution = target_quantized_idx_count / target_quantized_idx_count.sum()

    def calculate_kl_divergence_between_target_and_empirical_distribution(self, sample):
        """Calculate KL divergence between self.target_quantized_idx_count and empirical distribution.
        Args:
            sample (1D torch.Tensor): quantized indices tensor
        Returns:
            kl_divergence (float): KL divergence between self.target_quantized_idx_count and empirical distribution.
        """
        assert sample.ndim == 1
        # update sampled_quantized_idx_count
        sampled_quantized_idx_count_copy = self.sampled_quantized_idx_count.clone()
        with torch.no_grad():
            for quantized_idx in sample:
                sampled_quantized_idx_count_copy[quantized_idx] += 1
        # calculate KL divergence
        sampled_quantized_idx_count_copy += 1e-8
        empirical_distribution = sampled_quantized_idx_count_copy / sampled_quantized_idx_count_copy.sum()
        kl_divergence = torch.sum(
            empirical_distribution * torch.log(empirical_distribution / self.target_distribution)
        )

        if kl_divergence < 0:
            raise ValueError("KL divergence must be positive.")
        return kl_divergence

    def sample(self):
        # sample new samples based on KL divergence until the number of samples reaches the target number of samples
        # return subset of dataset
        while len(self.sampled_indices) < self.target_num_samples:
            # show progress
            if len(self.sampled_indices) % 10 == 0:
                print(f"sampled {len(self.sampled_indices) / self.target_num_samples * 100:.2f} %")
            # calculate KL divergence for each not sampled indices
            kl_divergences = {}
            not_sampled_indices = list(self.not_sampled_indices)
            for idx in not_sampled_indices:
                quantized_indices = torch.tensor(self.quantized_indices_memory[idx])
                kl_divergence = self.calculate_kl_divergence_between_target_and_empirical_distribution(
                    quantized_indices
                )
                kl_divergences[idx] = kl_divergence
            # select the index with the minimum KL divergence
            min_kl_divergence_idx = min(kl_divergences.keys(), key=kl_divergences.get)
            mlflow.log_metric(
                "min_data_kl_divergence", kl_divergences[min_kl_divergence_idx], step=len(self.sampled_indices)
            )
            # update sampled_indices and not_sampled_indices
            self.sampled_indices.add(min_kl_divergence_idx)
            self.not_sampled_indices.remove(min_kl_divergence_idx)
            # update sampled_quantized_idx_count
            quantized_indices = self.quantized_indices_memory[min_kl_divergence_idx]
            for quantized_idx in quantized_indices:
                self.sampled_quantized_idx_count[quantized_idx] += 1

        # return subset of self.dataset
        return torch.utils.data.Subset(self.dataset, list(self.sampled_indices))


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

        train_dataset = LibriSpeechDataset(root="./datasets/", split="train")
        test_dataset = LibriSpeechDataset(root="./datasets/", split="test")
        test_dataloader = get_dataloader(
            test_dataset, cfg.train.num_batch, shuffle=False, drop_last=False, collate_fn=test_dataset.collate_fn
        )
        # quantizeのときのidxと現在のidxの対応が変化していないことを確認する
        assert train_dataset[0][-1] == (
            "CHAPTER ONE MISSUS RACHEL LYNDE IS SURPRISED MISSUS RACHEL"
            + " LYNDE LIVED JUST WHERE THE AVONLEA MAIN ROAD DIPPED DOWN INTO A "
            + "LITTLE HOLLOW FRINGED WITH ALDERS AND LADIES EARDROPS AND TRAVERSED BY A BROOK"
        )
        assert (
            train_dataset[11][-1]
            == "AND HIS BEST SUIT OF CLOTHES WHICH WAS PLAIN PROOF THAT "
            + "HE WAS GOING OUT OF AVONLEA AND HE HAD THE BUGGY AND THE SORREL MARE "
            + "WHICH BETOKENED THAT HE WAS GOING A CONSIDERABLE DISTANCE NOW "
            + "WHERE WAS MATTHEW CUTHBERT GOING AND WHY WAS HE GOING THERE"
        )
        with open("librispeech_quantized_indices_memory.pkl", "rb") as f:
            quantized_indices_memory = pickle.load(f)
        assert len(quantized_indices_memory) == len(train_dataset)
        if cfg.selection.type == "random":
            logger.info("Using random sampler.")
            random_sampler = RandomSampler(
                quantized_indices_memory=quantized_indices_memory,
                dataset=train_dataset,
                ratio=cfg.selection.ratio,
            )
            train_subset = random_sampler.sample()
            logger.info(f"Train subset size: {len(train_subset) / len(train_dataset) * 100:.2f}%")
        elif cfg.selection.type == "uniform_kl_divergence":
            logger.info("Using KL divergence sampler.")
            uniform_kl_based_sampler = UniformKLBasedSampler(
                quantized_indices_memory=quantized_indices_memory,
                dataset=train_dataset,
                ratio=cfg.selection.ratio,
                device=DEVICE,
            )
            train_subset = uniform_kl_based_sampler.sample()
            logger.info(f"Train subset size: {len(train_subset) / len(train_dataset) * 100:.2f}%")
        elif cfg.selection.type == "data_kl_divergence":
            logger.info("Using Data KL divergence sampler.")
            data_kl_based_sampler = DataKLBasedSampler(
                quantized_indices_memory=quantized_indices_memory,
                dataset=train_dataset,
                ratio=cfg.selection.ratio,
                device=DEVICE,
            )
            train_subset = data_kl_based_sampler.sample()
            logger.info(f"Train subset size: {len(train_subset) / len(train_dataset) * 100:.2f}%")
        else:
            raise NotImplementedError

        train_dataloader = get_dataloader(
            train_subset, cfg.train.num_batch, shuffle=True, drop_last=True, collate_fn=train_dataset.collate_fn
        )
        num_label = len(train_dataset.vocab.keys())
        model = Model(nlabel=num_label, cfg=cfg).to(DEVICE)
        ctc_loss = torch.nn.CTCLoss(reduction="sum", blank=train_dataset.ctc_token_id)
        optimizer = torch.optim.Adam(
            model.parameters(),
            lr=cfg.train.optimize.lr,
            betas=(cfg.train.optimize.beta1, cfg.train.optimize.beta2),
            eps=cfg.train.optimize.eps,
        )
        scheduler = TransformerLR(
            optimizer, d_model=cfg.model.subsampling.output_feature_size, warmup_steps=cfg.train.optimize.warmup_steps
        )  # Warmup終了時点でおよそ0.0017になっている

        NUM_EPOCH = cfg.train.num_epoch
        NUM_ACCUM_STEP = cfg.train.num_accum_step
        for epoch in range(NUM_EPOCH):
            logger.info(f"Epoch {epoch + 1}/{NUM_EPOCH}")
            model.train()
            train_epoch_loss = 0
            train_epoch_cer = 0
            train_epoch_wer = 0
            train_cnt = 0

            for i, (bidx, bx, bx_len, by, by_len, _, _) in enumerate(train_dataloader):
                bx = bx.to(DEVICE)
                bx_len = bx_len.to(DEVICE)
                by = by.to(DEVICE)
                by_len = by_len.to(DEVICE)
                log_probs, y_lengths = model(bx, bx_len)
                loss = ctc_loss(log_probs.transpose(1, 0), by, y_lengths, by_len)
                loss.backward()
                if (i + 1) % NUM_ACCUM_STEP == 0:
                    optimizer.step()
                    scheduler.step()
                    optimizer.zero_grad()

                train_epoch_loss += loss.item() / bx.size(0)

                # calculate CER
                hypothesis = torch.argmax(log_probs, dim=-1)
                hypotheses = greedy_decoder(hypothesis, train_dataset.vocab, "[PAD]", "|", "_")
                answers = greedy_decoder(by, train_dataset.vocab, "[PAD]", "|", "_")
                train_epoch_cer += char_error_rate(hypotheses, answers)
                train_epoch_wer += word_error_rate(hypotheses, answers)

                train_cnt += 1

            mlflow.log_metric("train_loss", train_epoch_loss / train_cnt, step=epoch)
            mlflow.log_metric("train_cer", train_epoch_cer / train_cnt, step=epoch)
            mlflow.log_metric("train_wer", train_epoch_wer / train_cnt, step=epoch)

            logger.info(f"Train loss: {train_epoch_loss / train_cnt:.4f}")
            logger.info(f"Train CER: {train_epoch_cer / train_cnt:.4f}")
            logger.info(f"Train WER: {train_epoch_wer / train_cnt:.4f}")

            model.eval()
            test_epoch_loss = 0
            test_epoch_cer = 0
            test_epoch_wer = 0
            test_cnt = 0
            with torch.no_grad():
                for i, (bidx, bx, bx_len, by, by_len, _, _) in enumerate(test_dataloader):
                    bx = bx.to(DEVICE)
                    bx_len = bx_len.to(DEVICE)
                    by = by.to(DEVICE)
                    by_len = by_len.to(DEVICE)
                    log_probs, y_lengths = model(bx, bx_len)
                    loss = ctc_loss(log_probs.transpose(1, 0), by, y_lengths, by_len)
                    test_epoch_loss += loss.item() / bx.size(0)

                    # calculate CER
                    hypothesis = torch.argmax(log_probs, dim=-1)
                    hypotheses = greedy_decoder(hypothesis, train_dataset.vocab, "[PAD]", "|", "_")
                    answers = greedy_decoder(by, train_dataset.vocab, "[PAD]", "|", "_")
                    test_epoch_cer += char_error_rate(hypotheses, answers)
                    test_epoch_wer += word_error_rate(hypotheses, answers)

                    test_cnt += 1

            mlflow.log_metric("test_loss", test_epoch_loss / test_cnt, step=epoch)
            mlflow.log_metric("test_cer", test_epoch_cer / test_cnt, step=epoch)
            mlflow.log_metric("test_wer", test_epoch_wer / test_cnt, step=epoch)

            logger.info(f"Test loss: {test_epoch_loss / test_cnt:.4f}")
            logger.info(f"Test CER: {test_epoch_cer / test_cnt:.4f}")
            logger.info(f"Test WER: {test_epoch_wer / test_cnt:.4f}")

            checkpoint_dir = f"cpts/{EXPERIMENT_NAME}/{mlflow_run.info.run_id}"
            os.makedirs(checkpoint_dir, exist_ok=True)
            torch.save(model.state_dict(), f"{checkpoint_dir}/epoch_{epoch}_{test_epoch_cer / test_cnt:.4f}.pth")
            mlflow.log_artifact(f"{checkpoint_dir}/epoch_{epoch}_{test_epoch_cer / test_cnt:.4f}.pth")
        mlflow.log_artifact(LOG_DIR)


if __name__ == "__main__":
    main()

