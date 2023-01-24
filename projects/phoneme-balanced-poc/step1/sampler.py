import mlflow
import numpy as np
import torch


class PhonemeKLSampler:
    def __init__(
        self,
        sampling_pool: torch.utils.data.Dataset,
        target_dataset: torch.utils.data.Dataset,
        limit_duration: int,
        is_max: bool,
        phone_to_idx: dict,
    ):
        self.sampling_pool = sampling_pool
        self.target_dataset = target_dataset
        self.limit_duration = limit_duration
        self.is_max = is_max
        self.phone_to_idx = phone_to_idx

        self.sampled_indices = set()
        self.not_sampled_indices = set(range(len(sampling_pool)))
        self.sampled_duration = 0
        self.sampled_phone_counter = np.zeros(len(self.phone_to_idx), dtype=np.float32)

        print("Calculating target phoneme distribution...")
        print(f"num_phonemes: {len(self.phone_to_idx)}")
        phone_counter = np.array([0] * len(self.phone_to_idx), dtype=np.float32)
        for i in range(len(target_dataset)):
            phones = target_dataset[i][-1]
            phone_indices = [self.phone_to_idx[phone] for phone in phones if phone is not None]
            for phone_idx in phone_indices:
                phone_counter[phone_idx] += 1
        phone_counter += 1e-8
        self.target_phoneme_distribution = phone_counter / sum(phone_counter)

    def calculate_kl_divergence_between_target_and_sampled_phoneme_distribution(self, phones):

        sampled_phoneme_counter_copy = self.sampled_phone_counter.copy()
        phone_indices = [self.phone_to_idx[phone] for phone in phones if phone is not None]
        for phone_idx in phone_indices:
            sampled_phoneme_counter_copy[phone_idx] += 1

        sampled_phoneme_counter_copy += 1e-8
        sampled_phoneme_distribution = sampled_phoneme_counter_copy / sum(sampled_phoneme_counter_copy)

        kl_divergence = np.sum(
            sampled_phoneme_distribution * np.log(sampled_phoneme_distribution / self.target_phoneme_distribution)
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
                phones = self.sampling_pool[i][-1]

                kl_divergence = self.calculate_kl_divergence_between_target_and_sampled_phoneme_distribution(phones)
                kl_divergences[i] = kl_divergence

            if self.is_max:
                sampled_idx = max(kl_divergences.keys(), key=kl_divergences.get)
            else:
                sampled_idx = min(kl_divergences.keys(), key=kl_divergences.get)

            self.sampled_indices.add(sampled_idx)
            self.not_sampled_indices.remove(sampled_idx)
            self.sampled_duration += self.sampling_pool[sampled_idx][2].item() / 16000
            print(f"KL Div: {kl_divergences[sampled_idx]}")
            mlflow.log_metric("KL Div", kl_divergences[sampled_idx])
            # update sampled phoneme counter
            phones = self.sampling_pool[sampled_idx][-1]
            phone_indices = [self.phone_to_idx[phone] for phone in phones if phone is not None]
            for phone_idx in phone_indices:
                self.sampled_phone_counter[phone_idx] += 1

        return torch.utils.data.Subset(self.sampling_pool, list(self.sampled_indices))


class RandomSampler:
    def __init__(
        self,
        sampling_pool: torch.utils.data.Dataset,
        limit_duration: int,
    ):

        self.sampling_pool = sampling_pool
        self.limit_duration = limit_duration

    def sample(self):
        randomized_indices = torch.randperm(len(self.sampling_pool)).tolist()
        sampled_indices = set()
        sampled_duration = 0
        for idx in randomized_indices:
            duration = self.sampling_pool[idx][2] / 16000
            sampled_duration += duration
            sampled_indices.add(idx)
            if sampled_duration > self.limit_duration:
                break

        return torch.utils.data.Subset(self.sampling_pool, list(sampled_indices))
