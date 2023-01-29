import torch
from . import normalization


class ConvolutionLayer(torch.nn.Module):
    def __init__(self, input_channels, hidden_channels, depthwise_kernel_size, dropout):
        super().__init__()
        self.layer_norm = torch.nn.LayerNorm(input_channels)  # 全チャネルでの正規化
        self.pointwise_conv1 = torch.nn.Conv1d(
            in_channels=input_channels,
            out_channels=2 * hidden_channels,  # for GLU
            kernel_size=1,
            stride=1,
            padding=0,
            dilation=1,
            groups=1,
            bias=True,
        )
        self.glu = torch.nn.GLU(dim=1)  # convの出力は(B, 2D, T)なので、dim=1でチャネル方向を分割
        self.depthwise_conv = torch.nn.Conv1d(
            in_channels=hidden_channels,
            out_channels=hidden_channels,
            kernel_size=depthwise_kernel_size,
            stride=1,
            padding=depthwise_kernel_size // 2,  # 出力サイズを変えないように
            dilation=1,
            groups=hidden_channels,  # 各チャネルごとに畳み込みを行う
            bias=True,
        )
        self.batch_norm = torch.nn.BatchNorm1d(hidden_channels)
        self.swish = torch.nn.SiLU()
        self.pointwise_conv2 = torch.nn.Conv1d(
            in_channels=hidden_channels,
            out_channels=input_channels,
            kernel_size=1,
            stride=1,
            padding=0,
            dilation=1,
            groups=1,
            bias=True,
        )
        self.dropout = torch.nn.Dropout(dropout)

    def forward(self, x):
        # x: [B, T, D]
        x = self.layer_norm(x)  # layer_normは最後の次元に対して行われるため、[B, T, D]のまま
        x = x.transpose(1, 2)  # [B, D, T]
        x = self.pointwise_conv1(x)  # [B, 2D, T]
        x = self.glu(x)  # [B, D, T]
        x = self.depthwise_conv(x)  # [B, D, T]
        x = self.batch_norm(x)  # batch_normは[B, D, T]を期待している
        x = self.swish(x)  # [B, D, T]
        x = self.pointwise_conv2(x)  # [B, D, T]
        x = self.dropout(x)  # [B, D, T]
        x = x.transpose(1, 2)  # [B, T, D]
        return x


class CausalConv1d(torch.nn.Conv1d):
    def __init__(
        self,
        in_channels,
        out_channels,
        kernel_size,
        stride,
        dilation,
        groups,
        bias,
    ):
        self.causal_padding = kernel_size - 1
        super(CausalConv1d, self).__init__(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=kernel_size,
            stride=stride,
            padding=self.causal_padding,
            dilation=dilation,
            groups=groups,
            bias=bias,
        )

    def forward(self, input):
        # input: [B, D, T]
        output = super(CausalConv1d, self).forward(input)
        output = output[:, :, : -self.causal_padding]
        return output


class CausalConvolutionLayer(torch.nn.Module):
    def __init__(self, input_channels, hidden_channels, depthwise_kernel_size, dropout):
        super().__init__()
        self.layer_norm = normalization.CausalLayerNormalization()
        # pointwiseは常にCausal
        self.pointwise_conv1 = torch.nn.Conv1d(
            in_channels=input_channels,
            out_channels=2 * hidden_channels,  # for GLU
            kernel_size=1,
            stride=1,
            padding=0,
            dilation=1,
            groups=1,
            bias=True,
        )
        self.glu = torch.nn.GLU(dim=1)  # convの出力は(B, 2D, T)なので、dim=1でチャネル方向を分割
        # depthwiseはCausalにする必要がある
        # CausalConv1dでは長さは変わらない
        self.depthwise_conv = CausalConv1d(
            in_channels=hidden_channels,
            out_channels=hidden_channels,
            kernel_size=depthwise_kernel_size,
            stride=1,
            dilation=1,
            groups=hidden_channels,  # 各チャネルごとに畳み込みを行う
            bias=True,
        )
        self.batch_norm = normalization.CausalBatchNormalization()
        self.swish = torch.nn.SiLU()
        # pointwiseは常にCausal
        self.pointwise_conv2 = torch.nn.Conv1d(
            in_channels=hidden_channels,
            out_channels=input_channels,
            kernel_size=1,
            stride=1,
            padding=0,
            dilation=1,
            groups=1,
            bias=True,
        )
        self.dropout = torch.nn.Dropout(dropout)

    def forward(self, x):
        # x: [B, T, D]
        x = x.transpose(1, 2)  # [B, D, T]
        x = self.layer_norm(x)  # [B, D, T]
        x = self.pointwise_conv1(x)  # [B, 2D, T]
        x = self.glu(x)  # [B, D, T]
        x = self.depthwise_conv(x)  # [B, D, T]
        x = self.batch_norm(x)  # [B, D, T]
        x = self.swish(x)  # [B, D, T]
        x = self.pointwise_conv2(x)  # [B, D, T]
        x = self.dropout(x)  # [B, D, T]
        x = x.transpose(1, 2)  # [B, T, D]
        return x
