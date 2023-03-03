import torch


class LSTMLM(torch.nn.Module):
    def __init__(self, vocab_size, embed_dim, hidden_dim, num_layers, dropout):
        super().__init__()
        self.embedding = torch.nn.Embedding(vocab_size, embed_dim)
        self.rnn = torch.nn.LSTM(
            input_size=embed_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            dropout=dropout,
            batch_first=True,
            bidirectional=False,
        )
        self.linear = torch.nn.Linear(hidden_dim, vocab_size)

    def forward(self, x, x_len):
        total_length = x.shape[1]
        x = self.embedding(x)  # [B, T, E]
        self.rnn.flatten_parameters()
        packed_x = torch.nn.utils.rnn.pack_padded_sequence(
            input=x, lengths=x_len.tolist(), batch_first=True, enforce_sorted=False
        )
        packed_x, _ = self.rnn(packed_x)
        x, _ = torch.nn.utils.rnn.pad_packed_sequence(packed_x, total_length=total_length, batch_first=True)
        x = self.linear(x)
        return x

    @torch.no_grad()
    def score(self, x, hidden=None):
        if x.ndim == 0:
            x = x.unsqueeze(0).unsqueeze(0)
        elif x.ndim == 1:
            x = x.unsqueeze(0)
        x = self.embedding(x)
        self.rnn.flatten_parameters()
        x, hidden = self.rnn(x, hidden)
        x = self.linear(x)
        x = torch.nn.functional.log_softmax(x, dim=-1)
        return x, hidden
