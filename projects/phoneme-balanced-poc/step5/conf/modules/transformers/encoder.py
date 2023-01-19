import copy

import torch
from torch import nn

from . import positional_encoder


class TransformerEncoderLayer(nn.Module):
    def __init__(self, in_size=256, in_hidden_size=2048, nhead=4, dropout=0.1, norm_first=True):
        # in_size: 入力の特徴量次元
        # in_hidden_size:
        #   Linearにおける中間層の次元
        #   Attention is all you needにおいてはReLU(x*W_1 + b_1)*W_2 + b_2にて計算される
        super(TransformerEncoderLayer, self).__init__()
        # embed_dimは特徴量次元のサイズであり、vdim, kdimを指定しない場合はembed_dimと同じになる
        # 出力は[B, T, F]となるようにそれぞれのattentionが行われるぽい・・・
        self.norm_first = norm_first
        self.norm1 = nn.LayerNorm(in_size)
        self.norm2 = nn.LayerNorm(in_size)
        self.multi_head_self_attn = nn.MultiheadAttention(
            embed_dim=in_size, num_heads=nhead, dropout=dropout, batch_first=True)
        self.dropout = nn.Dropout(dropout)

        self.linear1 = nn.Linear(in_size, in_hidden_size)
        self.linear2 = nn.Linear(in_hidden_size, in_size)
        self.relu = nn.ReLU()
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)

    def forward(self, x, attn_mask=None, key_padding_mask=None):
        # args:
        #   x: [B, T, F]
        #   attn_msxk: [B*H, T, F](Hはヘッド数) or [T, F]
        #       計算したAttentionWeightに対して加算するマスク
        #       マスクされたAttentionWeightは無視される
        #       オフラインエンコーダーでは一般にシーケンスすべてを見るためNoneでよい
        #   key_padding_mask: [B, T]
        #       パディングされた部分をTrueにすることで無視するようにする
        #       サブサンプリング済みであれば特に指定しなくてよさそう
        # return:
        #  x: [B, T, F]
        if self.norm_first:
            x = self.norm1(x)
            x = x + self._multi_head_self_attn(x, attn_mask, key_padding_mask)
            x = self.norm2(x)
            x = x + self._feed_forward(x)
        else:
            # Attention is all you needにおける実装
            x = x + self._multi_head_self_attn(x, attn_mask, key_padding_mask)
            x = self.norm1(x)
            x = x + self._feed_forward(x)
            x = self.norm2(x)
        return x

    def _multi_head_self_attn(self, x, attn_mask=None, key_padding_mask=None):
        x = self.multi_head_self_attn(
            x, x, x, attn_mask=attn_mask, key_padding_mask=key_padding_mask, need_weights=False)[0]
        x = self.dropout(x)
        return x

    def _feed_forward(self, x):
        # Attention is all you needではReLU(x*W_1 + b_1)*W_2 + b_2で計算される
        # dropout: 各線形層ごとに適用する
        x = self.linear1(x)
        x = self.relu(x)
        x = self.dropout1(x)
        x = self.linear2(x)
        x = self.dropout2(x)

        return x


class TransformerEncoder(nn.Module):
    def __init__(self, in_size=256, nlayer=12, nhead=4, in_hidden_size=2048, dropout=0.1, norm_first=True) -> None:
        # in_size: 入力の特徴量次元(サブサンプリング後の場合もあり)
        # nlayer: TransformerEncoderLayerの数
        # nhead: MultiHeadAttentionのヘッド数
        # in_hidden_size: TransformerEncoderLayerの線形層における中間層の次元
        # dropout: Dropoutの割合
        # norm_first: LayerNormを先に行うかどうか
        super(TransformerEncoder, self).__init__()
        self.nlayer = nlayer
        self.positional_encoding = positional_encoder.PositionalEncoder(in_size=in_size)

        transformer_encoder_layer = TransformerEncoderLayer(
            in_size=in_size, in_hidden_size=in_hidden_size, nhead=nhead, dropout=dropout, norm_first=norm_first)
        self.transformer_encoder = nn.ModuleList(
            # オブジェクトの共有を防ぐためにdeepcopyを使う
            [copy.deepcopy(transformer_encoder_layer) for _ in range(nlayer)]
        )
        self.norm_first = norm_first
        if self.norm_first:
            self.norm = nn.LayerNorm(in_size)

    def forward(self, x, x_lengths):
        # args:
        #   x: [B, T, in_size]
        #   x_lengths: [B]
        # return:
        x = self.positional_encoding(x)  # [B, T', in_size]
        key_padding_mask = self._create_key_padding_mask(x_lengths)  # [B, T']
        for i, layer in enumerate(self.transformer_encoder):
            x = layer(x, attn_mask=None, key_padding_mask=key_padding_mask)
            if i == self.nlayer // 2:
                x_inter = x
        if self.norm_first:
            x = self.norm(x)
            if x_inter is not None:
                x_inter = self.norm(x_inter)

        return x, x_inter

    def _create_key_padding_mask(self, x_lengths):
        # args:
        #   x_lengths: [B]
        # return:
        #   key_padding_mask: [B, T]
        #       パディングされた部分をTrueにすることで無視するようにする
        max_len = x_lengths.max()
        key_padding_mask = torch.arange(max_len, device=x_lengths.device)[None, :] >= x_lengths[:, None]
        return key_padding_mask
