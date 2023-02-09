import torch

from . import convolution, feed_forward, multi_head_attention, normalization


class CausalConformerBlock(torch.nn.Module):
    def __init__(
        self,
        input_size,
        ff_hidden_size,
        conv_hidden_size,
        conv_kernel_size,
        mha_num_heads,
        dropout,
        num_previous_frames,
    ):
        super().__init__()
        self.ff_module1 = feed_forward.CausalFeedForwardModule(input_size, ff_hidden_size, dropout)
        self.conv_module = convolution.CausalConvolutionModule(input_size, conv_hidden_size, conv_kernel_size, dropout)
        self.mha_module = multi_head_attention.CausalMultiHeadAttentionModule(
            input_size, mha_num_heads, dropout, num_previous_frames=num_previous_frames
        )
        self.ff_module2 = feed_forward.CausalFeedForwardModule(input_size, ff_hidden_size, dropout)
        self.layer_norm = normalization.TimewiseLayerNormalization()

    def forward(self, x, x_lengths):
        # x: [B, T, D]
        # output: [B, T, D]
        res = x
        x = self.ff_module1(x)
        x = x * 0.5 + res
        res = x
        x = self.conv_module(x)
        x = x + res
        res = x
        x = self.mha_module(x, x_lengths)
        x = x + res
        res = x
        x = self.ff_module2(x)
        x = x * 0.5 + res
        x = x.transpose(1, 2)  # [B, D, T]
        x = self.layer_norm(x)
        x = x.transpose(1, 2)  # [B, T, D]
        return x


class ConformerBlock(torch.nn.Module):
    def __init__(
        self,
        input_size,
        ff_hidden_size,
        conv_hidden_size,
        conv_kernel_size,
        mha_num_heads,
        dropout,
        num_previous_frames,
    ):
        super().__init__()
        self.ff_module1 = feed_forward.FeedForwardModule(input_size, ff_hidden_size, dropout)
        self.conv_module = convolution.ConvolutionModule(input_size, conv_hidden_size, conv_kernel_size, dropout)
        self.mha_module = multi_head_attention.MultiHeadAttentionModule(
            input_size, mha_num_heads, dropout, num_previous_frames=num_previous_frames
        )
        self.ff_module2 = feed_forward.FeedForwardModule(input_size, ff_hidden_size, dropout)
        self.layer_norm = normalization.TimewiseLayerNormalization()

    def forward(self, x, x_lengths):
        # x: [B, T, D]
        # output: [B, T, D]
        res = x
        x = self.ff_module1(x)
        x = x * 0.5 + res
        res = x
        x = self.conv_module(x)
        x = x + res
        res = x
        x = self.mha_module(x, x_lengths)
        x = x + res
        res = x
        x = self.ff_module2(x)
        x = x * 0.5 + res
        x = x.transpose(1, 2)  # [B, D, T]
        x = self.layer_norm(x)
        x = x.transpose(1, 2)  # [B, T, D]
        return x
