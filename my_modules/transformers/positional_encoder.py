import torch
from torch import nn
import math

class PositionalEncoder(nn.Module):
    # 最大時系列長分のpeを事前に計算しておくことで、毎回の計算を防ぐ実装
    def __init__(self, in_size, dropout=0.1, max_len=5000):
        super(PositionalEncoder, self).__init__()
        # dropoutと入力のスケーリング調整
        self.dropout = nn.Dropout(p=dropout)
        self.xscale = math.sqrt(in_size)
        # pe: [max_len, in_size]
        pe = torch.zeros(max_len, in_size)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1) # [max_len, 1]
        div_term = torch.exp(torch.arange(0, in_size, 2).float() * (-math.log(10000.0) / in_size)) # [in_size//2]
        tiled_div_term = torch.tile(div_term, (max_len, 1)) # [max_len, in_size//2]
        tiled_position = torch.tile(position, (1, tiled_div_term.size(1))) # [max_len, in_size//2]
        # 各posの偶数次元はsin, 奇数次元はcos
        pe[:, 0::2] = torch.sin(tiled_position * tiled_div_term)
        pe[:, 1::2] = torch.cos(tiled_position * tiled_div_term)
        # pe: [1, max_len, in_size]
        pe = pe.unsqueeze(0)
        # 学習しないが、保存しておきたいものはregister_bufferで登録
        self.register_buffer('pe', pe)
    def forward(self, x):
        # x: [B, T, in_size]
        # pe: [1, max_len, in_size] -> [1, T, in_size]は下位次元が一致しているため、ブロードキャストされる
        x = x * self.xscale + self.pe[:, :x.size(1), :]
        return self.dropout(x)