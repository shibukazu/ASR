import torch

from . import adapter, convolution, feed_forward, multi_head_attention, normalization


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
        is_timewise_ln,
    ):
        super().__init__()
        self.ff_module1 = feed_forward.CausalFeedForwardModule(input_size, ff_hidden_size, dropout, is_timewise_ln)
        self.conv_module = convolution.CausalConvolutionModule(
            input_size, conv_hidden_size, conv_kernel_size, dropout, is_timewise_ln
        )
        self.mha_module = multi_head_attention.CausalMultiHeadAttentionModule(
            input_size, mha_num_heads, dropout, is_timewise_ln, num_previous_frames=num_previous_frames
        )
        self.ff_module2 = feed_forward.CausalFeedForwardModule(input_size, ff_hidden_size, dropout, is_timewise_ln)
        if is_timewise_ln:
            self.layer_norm = normalization.TimewiseLayerNormalization(input_size)
        else:
            self.layer_norm = normalization.CausalLayerNormalization(input_size)

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


class CausalConformerAdapterBlock(torch.nn.Module):
    def __init__(
        self,
        input_size,
        ff_hidden_size,
        conv_hidden_size,
        conv_kernel_size,
        mha_num_heads,
        dropout,
        num_previous_frames,
        is_timewise_ln,
        adapter_hidden_size,
    ):
        super().__init__()
        self.ff_module1 = feed_forward.CausalFeedForwardModule(input_size, ff_hidden_size, dropout, is_timewise_ln)
        self.conv_module = convolution.CausalConvolutionModule(
            input_size, conv_hidden_size, conv_kernel_size, dropout, is_timewise_ln
        )
        self.mha_module = multi_head_attention.CausalMultiHeadAttentionModule(
            input_size, mha_num_heads, dropout, is_timewise_ln, num_previous_frames=num_previous_frames
        )
        self.ff_module2 = feed_forward.CausalFeedForwardModule(input_size, ff_hidden_size, dropout, is_timewise_ln)
        if is_timewise_ln:
            self.layer_norm = normalization.TimewiseLayerNormalization(input_size)
        else:
            self.layer_norm = normalization.CausalLayerNormalization(input_size)

        self.adapter = adapter.BottleneckLinearAdapter(input_size, adapter_hidden_size)

    def forward(self, x, x_lengths):
        # x: [B, T, D]
        # output x: [B, T, D]
        # output vad_loss: [B, T]
        res = x
        x = self.ff_module1(x)
        x = x * 0.5 + res
        res = x
        x = self.conv_module(x)
        x = x + res
        res = x
        x = self.mha_module(x, x_lengths)
        _res = x

        x = self.adapter(x)
        # vad_loss = self.vad_loss(vad, y_vad)
        x = x + _res

        x = x + res
        res = x
        x = self.ff_module2(x)
        x = x * 0.5 + res
        x = x.transpose(1, 2)  # [B, D, T]
        x = self.layer_norm(x)
        x = x.transpose(1, 2)  # [B, T, D]
        return x


class CausalConformerAfterLNAdapterBlock(torch.nn.Module):
    def __init__(
        self,
        input_size,
        ff_hidden_size,
        conv_hidden_size,
        conv_kernel_size,
        mha_num_heads,
        dropout,
        num_previous_frames,
        is_timewise_ln,
        adapter_hidden_size,
    ):
        super().__init__()
        self.ff_module1 = feed_forward.CausalFeedForwardModule(input_size, ff_hidden_size, dropout, is_timewise_ln)
        self.conv_module = convolution.CausalConvolutionModule(
            input_size, conv_hidden_size, conv_kernel_size, dropout, is_timewise_ln
        )
        self.mha_module = multi_head_attention.CausalMultiHeadAttentionModule(
            input_size, mha_num_heads, dropout, is_timewise_ln, num_previous_frames=num_previous_frames
        )
        self.ff_module2 = feed_forward.CausalFeedForwardModule(input_size, ff_hidden_size, dropout, is_timewise_ln)
        if is_timewise_ln:
            self.layer_norm = normalization.TimewiseLayerNormalization(input_size)
        else:
            self.layer_norm = normalization.CausalLayerNormalization(input_size)

        self.adapter = adapter.BottleneckLinearAdapter(input_size, adapter_hidden_size)

    def forward(self, x, x_lengths):
        # x: [B, T, D]
        # output x: [B, T, D]
        # output vad_loss: [B, T]
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

        res = x
        x = self.adapter(x)
        x = x + res

        return x
