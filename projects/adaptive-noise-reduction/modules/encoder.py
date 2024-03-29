import torch
from modules.conformer.conformer import CausalConformerBlock, CausalConformerVADAdapterBlock
from modules.subsampling import Conv2DSubSampling


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
        is_timewise_ln,
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
                    is_timewise_ln=is_timewise_ln,
                )
                for _ in range(num_conformer_blocks)
            ]
        )

    def normalization(self, padded_input):
        # padded_input: [B, T, D]
        # normalize for each feature dim within each utterance
        mean = padded_input.mean(dim=1, keepdim=True)
        std = padded_input.std(dim=1, keepdim=True)
        normalized_padded_input = (padded_input - mean) / (std + 1e-5)

        return normalized_padded_input

    def forward(self, padded_input, input_lengths):
        normalized_padded_input = self.normalization(padded_input)
        subsampled_padded_input, subsampled_input_lengths = self.subsampling(normalized_padded_input, input_lengths)
        output = subsampled_padded_input
        output = self.fc(output)
        output = self.dropout(output)
        for conformer_block in self.conformer_blocks:
            output = conformer_block(output, subsampled_input_lengths)

        return output, subsampled_input_lengths


class CausalConformerVADAdapterEncoder(torch.nn.Module):
    def __init__(
        self,
        input_size,
        subsampled_input_size,
        num_conformer_blocks,
        ff_hidden_size,
        conv_hidden_size,
        conv_kernel_size,
        mha_num_heads,
        num_adapter_blocks,
        adapter_hidden_size,
        dropout,
        subsampling_kernel_size1,
        subsampling_stride1,
        subsampling_kernel_size2,
        subsampling_stride2,
        num_previous_frames,
        is_timewise_ln,
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
        if (
            subsampling_kernel_size1 == 3
            and subsampling_stride1 == 2
            and subsampling_kernel_size2 == 3
            and subsampling_stride2 == 2
        ):
            self.subsampling_factor = 4
        else:
            raise NotImplementedError
        self.fc = torch.nn.Linear(subsampled_input_size, subsampled_input_size)
        self.dropout = torch.nn.Dropout(dropout)
        self.vad_adapter_conformer_blocks = torch.nn.ModuleList(
            [
                CausalConformerVADAdapterBlock(
                    input_size=subsampled_input_size,
                    ff_hidden_size=ff_hidden_size,
                    conv_hidden_size=conv_hidden_size,
                    conv_kernel_size=conv_kernel_size,
                    mha_num_heads=mha_num_heads,
                    adapter_hidden_size=adapter_hidden_size,
                    dropout=dropout,
                    num_previous_frames=num_previous_frames,
                    is_timewise_ln=is_timewise_ln,
                )
                for _ in range(num_adapter_blocks)
            ]
        )
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
                    is_timewise_ln=is_timewise_ln,
                )
                for _ in range(num_conformer_blocks)
            ]
        )

    def normalization(self, padded_input):
        # padded_input: [B, T, D]
        # normalize for each feature dim within each utterance
        mean = padded_input.mean(dim=1, keepdim=True)
        std = padded_input.std(dim=1, keepdim=True)
        normalized_padded_input = (padded_input - mean) / (std + 1e-5)

        return normalized_padded_input

    def forward(self, padded_input, input_lengths):
        normalized_padded_input = self.normalization(padded_input)
        subsampled_padded_input, subsampled_input_lengths = self.subsampling(normalized_padded_input, input_lengths)
        output = subsampled_padded_input
        output = self.fc(output)
        output = self.dropout(output)
        vad_preds = []
        for conformer_block in self.vad_adapter_conformer_blocks:
            output, vad_pred = conformer_block(output, subsampled_input_lengths)
            vad_preds.append(vad_pred)
        # stack vad_preds to [num_adapter_blocks, B, T]
        vad_preds = torch.stack(vad_preds, dim=0)
        for conformer_block in self.conformer_blocks:
            output = conformer_block(output, subsampled_input_lengths)

        return output, subsampled_input_lengths, vad_preds
