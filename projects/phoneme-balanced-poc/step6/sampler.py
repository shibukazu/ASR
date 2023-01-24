import torch


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
