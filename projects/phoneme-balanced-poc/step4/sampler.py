import mlflow
import numpy as np
import torch
from quantizer import Quantizer


class RandomSampler:
    def __init__(
        self,
        sampling_pool: torch.utils.data.Dataset,
        target_dataset: torch.utils.data.Dataset,
        limit_duration: int,
    ):
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.quantizer = Quantizer("facebook/wav2vec2-large-lv60", device)

        self.sampling_pool = sampling_pool
        self.target_dataset = target_dataset
        self.limit_duration = limit_duration

        self.sampling_pool_quantized_indices = {}
        self.sampled_quantized_idx_counter = torch.zeros(320 * 320, dtype=np.float32)
        target_quantized_idx_counter = torch.zeros(320 * 320, dtype=np.float32)

        with torch.no_grad():
            for idx in range(len(sampling_pool)):
                audio = sampling_pool[idx][1].to(device)
                quantized_indices = self.quantizer.quantize(audio)
                self.sampling_pool_quantized_indices[idx] = quantized_indices

        with torch.no_grad():
            for idx in range(len(target_dataset)):
                audio = target_dataset[idx][1].to(device)
                quantized_indices = self.quantizer.quantize(audio)
                for quantized_idx in quantized_indices:
                    target_quantized_idx_counter[quantized_idx] += 1
            target_quantized_idx_counter += 1e-8
            self.target_quantized_idx_distribution = target_quantized_idx_counter / target_quantized_idx_counter.sum()

    def calculate_kl_divergence(self):
        sampled_quantized_idx_counter_copy = self.sampled_quantized_idx_counter.clone()
        sampled_quantized_idx_counter_copy += 1e-8
        sampled_quantized_idx_distribution = (
            sampled_quantized_idx_counter_copy / sampled_quantized_idx_counter_copy.sum()
        )
        kl_divergence = np.sum(
            sampled_quantized_idx_distribution
            * np.log(sampled_quantized_idx_distribution / self.target_quantized_idx_distribution)
        )
        mlflow.log_metric("kl_divergence", kl_divergence)

    def sample(self):
        randomized_indices = torch.randperm(len(self.sampling_pool)).tolist()
        sampled_indices = set()
        sampled_duration = 0
        for idx in randomized_indices:
            duration = self.sampling_pool[idx][2].item() / 16000
            sampled_duration += duration
            sampled_indices.add(idx)
            if sampled_duration > self.limit_duration:
                break

        for idx in list(sampled_indices):
            quantized_indices = self.sampling_pool_quantized_indices[idx]
            for quantized_idx in quantized_indices:
                self.sampled_quantized_idx_counter[quantized_idx] += 1
        self.calculate_kl_divergence()

        return torch.utils.data.Subset(self.sampling_pool, list(sampled_indices))


class QuantizeTrigramKLSampler:
    def __init__(
        self,
        sampling_pool: torch.utils.data.Dataset,
        target_dataset: torch.utils.data.Dataset,
        limit_duration: int,
        is_max: bool,
    ):
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.quantizer = Quantizer("facebook/wav2vec2-large-lv60", device)
        self.sampling_pool = sampling_pool
        self.target_dataset = target_dataset
        self.limit_duration = limit_duration
        self.is_max = is_max

        self.sampled_indices = set()
        self.not_sampled_indices = set(range(len(sampling_pool)))
        self.sampled_duration = 0

        self.sampling_pool_quantized_indices = {}
        self.sampled_quantized_idx_counter = torch.zeros(320 * 320, dtype=np.float32)
        target_quantized_idx_counter = torch.zeros(320 * 320, dtype=np.float32)

        with torch.no_grad():
            for idx in range(len(sampling_pool)):
                audio = sampling_pool[idx][1].to(device)
                quantized_indices = self.quantizer.quantize(audio)
                self.sampling_pool_quantized_indices[idx] = quantized_indices

        with torch.no_grad():
            for idx in range(len(target_dataset)):
                audio = target_dataset[idx][1].to(device)
                quantized_indices = self.quantizer.quantize(audio)
                for quantized_idx in quantized_indices:
                    target_quantized_idx_counter[quantized_idx] += 1
            target_quantized_idx_counter += 1e-8
            self.target_quantized_idx_distribution = target_quantized_idx_counter / target_quantized_idx_counter.sum()

    def calculate_kl_divergence(self, sample):

        sampled_quantized_idx_counter_copy = self.sampled_quantized_idx_counter.copy()
        for quantized_idx in sample:
            sampled_quantized_idx_counter_copy[quantized_idx] += 1
        sampled_quantized_idx_counter_copy += 1e-8
        sampled_quantized_idx_distribution = (
            sampled_quantized_idx_counter_copy / sampled_quantized_idx_counter_copy.sum()
        )
        kl_divergence = torch.sum(
            sampled_quantized_idx_distribution
            * torch.log(sampled_quantized_idx_distribution / self.target_quantized_idx_distribution)
        )

        if kl_divergence < 0:
            raise ValueError("KL divergence must be positive.")
        return kl_divergence

    def sample(self):
        while self.sampled_duration < self.limit_duration:
            # show progress
            print(f"sampled_duration: {self.sampled_duration / self.limit_duration * 100:.2f}%")
            if len(self.not_sampled_indices) == 0:
                break
            kl_divergences = {}
            for i in list(self.not_sampled_indices):
                quantized_indices = self.sampling_pool_quantized_indices[i]

                kl_divergence = self.calculate_kl_divergence(quantized_indices)
                kl_divergences[i] = kl_divergence

            if self.is_max:
                sampled_idx = max(kl_divergences.keys(), key=kl_divergences.get)
            else:
                sampled_idx = min(kl_divergences.keys(), key=kl_divergences.get)

            self.sampled_indices.add(sampled_idx)
            self.not_sampled_indices.remove(sampled_idx)
            self.sampled_duration += self.sampling_pool[sampled_idx][2].item() / 16000
            print(f"kl_divergence: {kl_divergences[sampled_idx]}")
            mlflow.log_metric("kl_divergence", kl_divergences[sampled_idx], step=len(self.sampled_indices))

            quantized_indices = self.sampling_pool_quantized_indices[sampled_idx]
            for quantized_idx in quantized_indices:
                self.sampled_quantized_idx_counter[quantized_idx] += 1

        return torch.utils.data.Subset(self.sampling_pool, list(self.sampled_indices))
