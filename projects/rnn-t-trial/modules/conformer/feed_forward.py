import torch

from . import normalization


class CausalFeedForwardModule(torch.nn.Module):
    def __init__(self, input_size, hidden_size, dropout):
        super().__init__()
        self.layer_norm = normalization.CausalLayerNormalization(input_size)
        self.fc1 = torch.nn.Linear(input_size, hidden_size, bias=True)
        self.swish = torch.nn.SiLU()
        self.dropout1 = torch.nn.Dropout(dropout)
        self.fc2 = torch.nn.Linear(hidden_size, input_size, bias=True)
        self.dropout2 = torch.nn.Dropout(dropout)

    def forward(self, x):
        # x: [B, T, D]
        # output: [B, T, D]
        x = x.transpose(1, 2)  # [B, D, T]
        x = self.layer_norm(x)
        x = x.transpose(1, 2)  # [B, T, D]
        x = self.fc1(x)
        x = self.swish(x)
        x = self.dropout1(x)
        x = self.fc2(x)
        x = self.dropout2(x)
        return x


class FeedForwardModule(torch.nn.Module):
    def __init__(self, input_size, hidden_size, dropout):
        super().__init__()
        self.layer_norm = torch.nn.LayerNorm(input_size)
        self.fc1 = torch.nn.Linear(input_size, hidden_size, bias=True)
        self.swish = torch.nn.SiLU()
        self.dropout1 = torch.nn.Dropout(dropout)
        self.fc2 = torch.nn.Linear(hidden_size, input_size, bias=True)
        self.dropout2 = torch.nn.Dropout(dropout)

    def forward(self, x):
        # x: [B, T, D]
        # output: [B, T, D]
        x = self.layer_norm(x)  # layer_normは最後の次元に対して行われるため、[B, T, D]のまま
        x = self.fc1(x)
        x = self.swish(x)
        x = self.dropout1(x)
        x = self.fc2(x)
        x = self.dropout2(x)
        return x
