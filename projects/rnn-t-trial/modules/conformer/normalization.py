import torch


class CausalBatchNormalization(torch.nn.Module):
    def __init__(
        self,
        eps=1e-8,
        affine=True,
    ):
        super().__init__()
        self.eps = eps
        self.affine = affine
        # NOTE: サイズ1で正しいか？
        if self.affine:
            self.gamma = torch.nn.Parameter(torch.ones(1))
            self.beta = torch.nn.Parameter(torch.zeros(1))

    def forward(self, x):
        # x: [B, D, T]
        cum_sum = x.cumsum(dim=-1).sum(dim=0).repeat(x.shape[0], 1, 1)
        cum_num_element = (
            (torch.arange(1, x.shape[-1] + 1) * x.shape[0]).repeat(x.shape[0], x.shape[1], 1).to(x.device)
        )
        cum_mean = cum_sum / cum_num_element
        cum_var = ((x - cum_mean) ** 2).cumsum(dim=-1).sum(dim=0).repeat(x.shape[0], 1, 1) / cum_num_element
        cum_std = torch.sqrt(cum_var + self.eps)
        cum_std = cum_std + self.eps
        normalized_x = (x - cum_mean) / cum_std
        if self.affine:
            normalized_x = normalized_x * self.gamma + self.beta
        return normalized_x


class TimewiseBatchNormalization(torch.nn.Module):
    def __init__(
        self,
        input_size,
        eps=1e-8,
        affine=True,
    ):
        super().__init__()
        self.eps = eps
        self.affine = affine
        # NOTE: サイズ1で正しいか？
        if self.affine:
            self.gamma = torch.nn.Parameter(torch.ones(input_size))
            self.beta = torch.nn.Parameter(torch.zeros(input_size))

    def forward(self, x):
        # x: [B, D, T]
        time_and_chan_wise_sum = x.sum(dim=0).unsqueeze(0).repeat(x.shape[0], 1, 1)
        time_and_chan_wise_mean = time_and_chan_wise_sum / x.shape[0]
        time_and_chan_wise_var = ((x - time_and_chan_wise_mean) ** 2).sum(dim=0).unsqueeze(0).repeat(
            x.shape[0], 1, 1
        ) / x.shape[0]
        time_and_chan_wise_std = torch.sqrt(time_and_chan_wise_var + self.eps)
        time_and_chan_wise_std = time_and_chan_wise_std + self.eps
        normalized_x = (x - time_and_chan_wise_mean) / time_and_chan_wise_std
        if self.affine:
            normalized_x = normalized_x.transpose(1, 2)
            normalized_x = normalized_x * self.gamma + self.beta
            normalized_x = normalized_x.transpose(1, 2)
        return normalized_x


class CausalLayerNormalization(torch.nn.Module):
    def __init__(
        self,
        eps=1e-8,
        affine=True,
    ):
        super().__init__()
        self.eps = eps
        self.affine = affine
        # NOTE: サイズ1で正しいか？
        if self.affine:
            self.gamma = torch.nn.Parameter(torch.ones(1))
            self.beta = torch.nn.Parameter(torch.zeros(1))

    def forward(self, x):
        # x: [B, D, T]
        cum_sum = x.cumsum(dim=-1).sum(dim=1).unsqueeze(1).repeat(1, x.shape[1], 1)
        cum_num_element = (
            (torch.arange(1, x.shape[-1] + 1) * x.shape[1]).repeat(x.shape[0], x.shape[1], 1).to(x.device)
        )
        cum_mean = cum_sum / cum_num_element
        cum_var = ((x - cum_mean) ** 2).cumsum(dim=-1).sum(dim=1).unsqueeze(1).repeat(
            1, x.shape[1], 1
        ) / cum_num_element
        cum_std = torch.sqrt(cum_var + self.eps)
        cum_std = cum_std + self.eps
        normalized_x = (x - cum_mean) / cum_std
        if self.affine:
            normalized_x = normalized_x * self.gamma + self.beta
        return normalized_x


class TimewiseLayerNormalization(torch.nn.Module):
    def __init__(
        self,
        input_size,
        eps=1e-8,
        affine=True,
    ):
        super().__init__()
        self.eps = eps
        self.affine = affine
        # NOTE: サイズ1で正しいか？
        if self.affine:
            self.gamma = torch.nn.Parameter(torch.ones(input_size))
            self.beta = torch.nn.Parameter(torch.zeros(input_size))

    def forward(self, x):
        # x: [B, D, T]
        time_and_batch_wise_sum = x.sum(dim=1).unsqueeze(1).repeat(1, x.shape[1], 1)
        time_and_batch_wise_mean = time_and_batch_wise_sum / x.shape[1]
        time_and_batch_wise_var = ((x - time_and_batch_wise_mean) ** 2).sum(dim=1).unsqueeze(1).repeat(
            1, x.shape[1], 1
        ) / x.shape[1]
        time_and_batch_wise_std = torch.sqrt(time_and_batch_wise_var + self.eps)
        time_and_batch_wise_std = time_and_batch_wise_std + self.eps
        normalized_x = (x - time_and_batch_wise_mean) / time_and_batch_wise_std
        if self.affine:
            normalized_x = normalized_x.transpose(1, 2)
            normalized_x = normalized_x * self.gamma + self.beta
            normalized_x = normalized_x.transpose(1, 2)
        return normalized_x
