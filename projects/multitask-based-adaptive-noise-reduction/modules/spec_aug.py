import numpy as np


class SpecAug:
    """
    マスクのみ行うSpecAug
    """

    def __init__(self, freq_mask_max_length, time_mask_max_length, num_freq_mask, num_time_mask):
        self.freq_mask_max_length = freq_mask_max_length
        self.num_freq_mask = num_freq_mask
        self.time_mask_max_length = time_mask_max_length
        self.num_time_mask = num_time_mask

    def freq_mask(self, spec):
        """
        Args:
            spec (Tensor): Tensor [T, F]
        """
        cloned = spec.clone()  # [T, F]
        num_mel_channels = cloned.shape[1]
        for _ in range(0, self.num_freq_mask):
            f = np.random.randint(0, self.freq_mask_max_length + 1)
            if f >= num_mel_channels:
                continue
            f_zero = np.random.randint(0, num_mel_channels - f)

            cloned[:, f_zero : f_zero + f] = 0
        return cloned

    def time_mask(self, spec):
        """
        Args:
            spec (Tensor): Tensor [T, F]
        """
        cloned = spec.clone()
        num_time_steps = cloned.shape[0]
        for _ in range(0, self.num_time_mask):
            t = np.random.randint(0, self.time_mask_max_length + 1)
            if t >= num_time_steps:
                continue
            t_zero = np.random.randint(0, num_time_steps - t)

            cloned[t_zero : t_zero + t, :] = 0
        return cloned

    def __call__(self, spec):
        """
        Args:
            spec (Tensor): Tensor [T, F]
        """
        spec = self.freq_mask(spec)
        spec = self.time_mask(spec)
        return spec
