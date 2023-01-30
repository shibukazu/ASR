import torch

from . import normalization


class CausalMultiHeadAttentionModule(torch.nn.Module):
    def __init__(
        self,
        input_size,
        num_heads,
        dropout,
    ):
        super().__init__()
        # don't use positional encoding in causal conformer
        self.layer_norm = normalization.CausalLayerNormalization()
        self.multi_head_attention = torch.nn.MultiheadAttention(
            embed_dim=input_size,
            num_heads=num_heads,
            dropout=dropout,
            batch_first=True,
        )
        self.dropout = torch.nn.Dropout(dropout)

    def _create_causal_attn_mask(self, input_length):
        # output: bynary mask [T, T]
        mask = torch.triu(torch.ones(input_length, input_length), diagonal=1).bool()
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
            attn_mask=self._create_causal_attn_mask(x.size(1)).to(x.device),
            key_padding_mask=self._create_key_padding_mask(x_lengths, x.size(1)).to(x.device),
        )
        x = self.dropout(x)
        return x
