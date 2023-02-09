import torch
from modules import torchaudio_conformer
from modules.conformer.conformer import CausalConformerBlock
from modules.subsampling import Conv2DSubSampling


class LSTMEncoder(torch.nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, dropout):
        super().__init__()
        self.subsampling = Conv2DSubSampling(input_size, hidden_size, 3, 2, 3, 2)
        self.lstm = torch.nn.LSTM(
            hidden_size, hidden_size, num_layers, batch_first=True, bidirectional=True, dropout=dropout
        )

    def forward(self, padded_input, input_lengths):
        subsampled_padded_input, subsampled_input_lengths = self.subsampling(padded_input, input_lengths)

        packed_padded_input = torch.nn.utils.rnn.pack_padded_sequence(
            subsampled_padded_input, subsampled_input_lengths, batch_first=True, enforce_sorted=False
        )
        self.lstm.flatten_parameters()
        packed_padded_output, _ = self.lstm(packed_padded_input)
        padded_output, _ = torch.nn.utils.rnn.pad_packed_sequence(packed_padded_output, batch_first=True)
        padded_output = padded_output[:, :, : self.lstm.hidden_size] + padded_output[:, :, self.lstm.hidden_size :]

        return padded_output, subsampled_input_lengths


class CausalConformerEncoder(torch.nn.Module):
    def __init__(
        self,
        input_size,
        subsampled_input_size,
        num_conformer_blocks,
        ff_hidden_size,
        conv_hidden_size,
        conv_kernel_size,
        mha_num_heads,
        dropout,
        subsampling_kernel_size1,
        subsampling_stride1,
        subsampling_kernel_size2,
        subsampling_stride2,
        num_previous_frames,
    ):
        super().__init__()
        self.subsampling = Conv2DSubSampling(
            input_size,
            subsampled_input_size,
            subsampling_kernel_size1,
            subsampling_stride1,
            subsampling_kernel_size2,
            subsampling_stride2,
        )
        self.fc = torch.nn.Linear(subsampled_input_size, subsampled_input_size)
        self.dropout = torch.nn.Dropout(dropout)
        self.conformer_blocks = torch.nn.ModuleList(
            [
                CausalConformerBlock(
                    input_size=subsampled_input_size,
                    ff_hidden_size=ff_hidden_size,
                    conv_hidden_size=conv_hidden_size,
                    conv_kernel_size=conv_kernel_size,
                    mha_num_heads=mha_num_heads,
                    dropout=dropout,
                    num_previous_frames=num_previous_frames,
                )
                for _ in range(num_conformer_blocks)
            ]
        )

    def forward(self, padded_input, input_lengths):
        subsampled_padded_input, subsampled_input_lengths = self.subsampling(padded_input, input_lengths)
        output = subsampled_padded_input
        output = self.fc(output)
        output = self.dropout(output)
        for conformer_block in self.conformer_blocks:
            output = conformer_block(output, subsampled_input_lengths)

        return output, subsampled_input_lengths


class TorchAudioConformerEncoder(torch.nn.Module):
    def __init__(
        self,
        input_size,
        subsampled_input_size,
        num_conformer_blocks,
        ff_hidden_size,
        conv_hidden_size,
        conv_kernel_size,
        mha_num_heads,
        dropout,
        subsampling_kernel_size1,
        subsampling_stride1,
        subsampling_kernel_size2,
        subsampling_stride2,
        num_previous_frames,
    ):
        super().__init__()
        self.subsampling = Conv2DSubSampling(
            input_size,
            subsampled_input_size,
            subsampling_kernel_size1,
            subsampling_stride1,
            subsampling_kernel_size2,
            subsampling_stride2,
        )
        self.fc = torch.nn.Linear(subsampled_input_size, subsampled_input_size)
        self.dropout = torch.nn.Dropout(dropout)
        self.conformer_blocks = torchaudio_conformer.Conformer(
            input_dim=subsampled_input_size,
            num_heads=mha_num_heads,
            ffn_dim=ff_hidden_size,
            num_layers=num_conformer_blocks,
            depthwise_conv_kernel_size=conv_kernel_size,
            dropout=dropout,
            use_group_norm=False,
            convolution_first=True,
        )

    def forward(self, padded_input, input_lengths):
        subsampled_padded_input, subsampled_input_lengths = self.subsampling(padded_input, input_lengths)
        output = subsampled_padded_input
        output = self.fc(output)
        output = self.dropout(output)
        output, output_lengths = self.conformer_blocks(output, subsampled_input_lengths)

        return output, output_lengths  # ここの長さ同じ？ -> 同じ
