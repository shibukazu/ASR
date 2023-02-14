import torch

from . import normalization


class CausalMultiHeadAttentionModule(torch.nn.Module):
    def __init__(
        self,
        input_size,
        num_heads,
        dropout,
        num_previous_frames,
    ):
        super().__init__()
        # don't use positional encoding in causal conformer
        self.layer_norm = normalization.TimewiseLayerNormalization(input_size)
        self.multi_head_attention = torch.nn.MultiheadAttention(
            embed_dim=input_size,
            num_heads=num_heads,
            dropout=dropout,
            batch_first=True,
        )
        self.dropout = torch.nn.Dropout(dropout)
        self.num_previous_frames = num_previous_frames

    def _create_causal_attn_mask(self, input_length, num_previous_frames):
        # output: bynary mask [T, T]
        if num_previous_frames == "all":
            previous_mask = torch.zeros(input_length, input_length).bool()
        else:
            previous_mask = torch.tril(
                torch.ones(input_length, input_length), diagonal=-(num_previous_frames + 1)
            ).bool()
        future_mask = torch.triu(torch.ones(input_length, input_length), diagonal=1).bool()
        mask = torch.logical_or(previous_mask, future_mask)
        return mask

    def _create_key_padding_mask(self, input_lengths, max_length):
        # input_lengths: [B]
        # output: bynary mask [B, T]
        mask = torch.arange(max_length)[None, :] >= input_lengths[:, None]
        return mask

    def forward(self, x, x_lengths):
        # x: [B, T, D]
        # output: [B, T, D]
        # the input for layer norm should be [B, D, T]
        x = x.transpose(1, 2)
        x = self.layer_norm(x)
        x = x.transpose(1, 2)
        x, _ = self.multi_head_attention(
            query=x,
            key=x,
            value=x,
            attn_mask=self._create_causal_attn_mask(x.size(1), num_previous_frames=self.num_previous_frames).to(
                x.device
            ),
            # FIXME: key_padding_mask cause NaN loss
            # key_padding_mask=self._create_key_padding_mask(x_lengths, x.size(1)).to(x.device),
        )
        x = self.dropout(x)
        return x


class MultiHeadAttentionModule(torch.nn.Module):
    def __init__(
        self,
        input_size,
        num_heads,
        dropout,
    ):
        super().__init__()
        # don't use positional encoding in causal conformer
        self.layer_norm = torch.nn.LayerNorm(input_size)
        self.multi_head_attention = torch.nn.MultiheadAttention(
            embed_dim=input_size,
            num_heads=num_heads,
            dropout=dropout,
            batch_first=True,
        )
        self.dropout = torch.nn.Dropout(dropout)

    def _create_key_padding_mask(self, input_lengths, max_length):
        # input_lengths: [B]
        # output: bynary mask [B, T]
        mask = torch.arange(max_length)[None, :] >= input_lengths[:, None]
        return mask

    def forward(self, x, x_lengths):
        # x: [B, T, D]
        # output: [B, T, D]
        x = self.layer_norm(x)
        x, _ = self.multi_head_attention(
            query=x,
            key=x,
            value=x,
            key_padding_mask=self._create_key_padding_mask(x_lengths, x.size(1)).to(x.device),
            need_weights=False,
        )
        x = self.dropout(x)
        return x
