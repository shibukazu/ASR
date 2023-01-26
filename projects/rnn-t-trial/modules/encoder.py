import torch
from modules.subsampling import Conv2DSubSampling


class Encoder(torch.nn.Module):
    def __init__(self, input_size, subsampled_input_size, hidden_size, num_layers, output_size):
        super().__init__()
        self.subsampling = Conv2DSubSampling(input_size, subsampled_input_size, 3, 2, 3, 1)
        self.lstm = torch.nn.LSTM(subsampled_input_size, hidden_size, num_layers, batch_first=True, bidirectional=True)
        self.fc = torch.nn.Linear(hidden_size, output_size)

    def forward(self, padded_input, input_lengths):
        subsampled_padded_input, subsampled_input_lengths = self.subsampling(padded_input, input_lengths)

        packed_padded_input = torch.nn.utils.rnn.pack_padded_sequence(
            subsampled_padded_input, subsampled_input_lengths, batch_first=True, enforce_sorted=False
        )
        packed_padded_output, _ = self.lstm(packed_padded_input)
        padded_output, _ = torch.nn.utils.rnn.pad_packed_sequence(packed_padded_output, batch_first=True)
        padded_output = padded_output[:, :, : self.lstm.hidden_size] + padded_output[:, :, self.lstm.hidden_size :]
        padded_output = self.fc(padded_output)
        return padded_output, subsampled_input_lengths
