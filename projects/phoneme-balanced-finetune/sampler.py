from typing import Dict

import mlflow
import torch


class UniformKLSampler:
    def __init__(
        self,
        quantized_indices_memory: Dict,
        dataset: torch.utils.data.Dataset,
        limit: float,
    ):
        self.dataset = dataset
        self.limit = limit
        self.sampled_duration = 0.0
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
        while self.sampled_duration < self.limit:
            # show progress
            print(f"sampled {self.sampled_duration / self.limit * 100:.2f} %")
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
            # calculate sample duration
            duration = self.dataset[min_kl_divergence_idx][2].item() / 16000
            self.sampled_duration += duration

            # update sampled_indices and not_sampled_indices
            self.sampled_indices.add(min_kl_divergence_idx)
            self.not_sampled_indices.remove(min_kl_divergence_idx)
            # update sampled_quantized_idx_count
            quantized_indices = self.quantized_indices_memory[min_kl_divergence_idx]
            for quantized_idx in quantized_indices:
                self.sampled_quantized_idx_count[quantized_idx] += 1

        # return subset of self.dataset
        return torch.utils.data.Subset(self.dataset, list(self.sampled_indices))


class RandomSampler:
    def __init__(
        self,
        quantized_indices_memory: Dict,
        dataset: torch.utils.data.Dataset,
        limit: float,
    ):
        """
        limit: sample size limit (sec)
        """
        self.quantized_indices_memory = quantized_indices_memory
        self.dataset = dataset
        self.limit = limit
        self.sampled_duration = 0.0
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
        randomized_indices = torch.randperm(len(self.dataset)).tolist()
        sampled_indices = []
        for idx in randomized_indices:
            duration = self.dataset[idx][2].item() / 16000
            self.sampled_duration += duration
            sampled_indices.append(idx)
            if self.sampled_duration > self.limit:
                break
        for idx in sampled_indices:
            quantized_indices = self.quantized_indices_memory[idx]
            for quantized_idx in quantized_indices:
                self.sampled_quantized_idx_count[quantized_idx] += 1
        self.calculate_kl_divergence_between_uniform_and_empirical_distribution()
        self.calculate_kl_divergence_between_target_and_empirical_distribution()

        return torch.utils.data.Subset(self.dataset, sampled_indices)
