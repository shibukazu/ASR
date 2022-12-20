import torch
from torch import nn

class NaiveConv2DSubSampling(nn.Module):
    # 同一のカーネルによって、特徴量次元および時間次元を縮小する
    # 特徴量次元は線形変換によって任意の長さに変換する
    def __init__(self, in_size, out_size):
        super(NaiveConv2DSubSampling, self).__init__()
        # kernel_sizeとstrideで時間軸のサイズおよび特徴量軸サイズは決定する
        # 以下の設定では、それぞれ1/4になる
        self.kernel_size = 3
        self.stride = 1
        self.conv = nn.Sequential(
            nn.Conv2d(1, 1, self.kernel_size, self.stride),
            nn.ReLU(),
            nn.Conv2d(1, 1, self.kernel_size, self.stride),
            nn.ReLU(),
        )
        # 畳み込み演算後の特徴量軸のサイズ
        conv_out_size = self.output_size(self.output_size(in_size))
        # conv_out_sizeからout_sizeに変換するための線形変換
        self.linear = nn.Linear(conv_out_size, out_size)

    
    def forward(self, x):
        x = x.unsqueeze(1) # (B, 1, T, F)
        x = self.conv(x) # (B, 1, T', F')
        x = x.squeeze(1) # (B, T', F')
        x = self.linear(x) # (B, T', out_size)
        return x
    
    def output_size(self, input_size):
        # floor((input_size - self.kernel_size) / self.stride + 1)
        return torch.div(input_size - self.kernel_size + self.stride, self.stride, rounding_mode="floor")

class Conv2DSubSampling(nn.Module):
    def __init__(self, in_size, out_size, kernel_size1=3, stride1=2, kernel_size2=3, stride2=2):
        # args:
        #   in_size: 元の特徴量次元数
        #   out_size: サブサンプリング後の特徴量次元数
        # note:
        #   この実装では、特徴量次元をチャネル方向に移動させた上で畳み込みを行う
        #   サブサンプリング後の時系列長は以下のように決定される
        #   if kernel_size1 == 3 and stride1 == 2 and kernel_size2 == 3 and stride2 == 2:
        #       then T' = T // 4 (k=4)
        #   if kernel_size1 == 3 and stride1 == 2 and kernel_size2 == 3 and stride2 == 1:
        #       then T' = T // 2 (k=2)
        #   サブサンプリング後の特徴量次元数は以下のように決定される
        #   D' = out_size
        super(Conv2DSubSampling, self).__init__()
        self.kernel_size1 = kernel_size1
        self.stride1 = stride1
        self.kernel_size2 = kernel_size2
        self.stride2 = stride2
        # 以下の演算によって(B,D,T//k,F//k)になる
        self.conv = nn.Sequential(
            nn.Conv2d(1, out_size, self.kernel_size1, self.stride1),
            nn.ReLU(),
            nn.Conv2d(out_size, out_size, self.kernel_size2, self.stride2),
            nn.ReLU(),
        )
        # 畳み込み演算後の特徴量軸のサイズ
        conv_out_size = self.output_size(self.output_size(in_size, self.kernel_size1, self.stride1), self.kernel_size2, self.stride2)
        # (B, T//k, F//k * out_size) -> (B, T//k, out_size)に変換するための線形変換 
        self.linear = nn.Linear(out_size * conv_out_size, out_size)

    
    def forward(self, x, x_lengths):
        # args:
        #   x: (B, T, F)
        #   x_lengths: (B)
        #       xの各サンプルのパディング前時系列長
        # return:
        #   x: (B, T', F')
        #   x_lengths: (B)
        #       パディング前時系列長のサブサンプリング後の値

        x = x.unsqueeze(1) # (B, 1, T, F)
        x = self.conv(x) # (B, D, T', F')
        x = torch.transpose(x, 1, 2).contiguous() # (B, T', D, F')
        x = x.view(x.size(0), x.size(1), -1) # (B, T', D * F')
        x = self.linear(x) # (B, T', D)
        subsampled_x = x
        # 畳み込み演算後の時系列軸のサイズ
        subsampled_x_lengths = self.output_size(self.output_size(x_lengths, self.kernel_size1, self.stride1), self.kernel_size2, self.stride2)
        return subsampled_x, subsampled_x_lengths
    
    def output_size(self, in_size, kernel_size, stride):
        # 畳み込み後のTおよびFのサイズを計算する関数
        # floor((in_size - self.kernel_size) / self.stride + 1)
        return torch.div(in_size - kernel_size + stride, stride, rounding_mode="floor")