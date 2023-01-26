import torch


class Predictor(torch.nn.Module):
    # input: (batch_size, seq_len) token idx sequence
    def __init__(self, vocab_size, embedding_size, hidden_size, num_layers, output_size, blank_idx):
        super().__init__()
        self.embedding = torch.nn.Embedding(vocab_size, embedding_size)
        self.lstm = torch.nn.LSTM(embedding_size, hidden_size, num_layers, batch_first=True)
        self.fc = torch.nn.Linear(hidden_size, output_size)

        self.blank_idx = blank_idx

    def forward(self, padded_input, input_lengths, hidden=None):
        # padded_inputs: [B, U]
        # hidden: inferenceなどで過去の隠れ状態を与える場合に使用
        # sosトークンで一つずらす
        padded_input_prepended = torch.nn.functional.pad(padded_input, (1, 0, 0, 0), value=self.blank_idx)  # [B, U+1]
        input_lengths_prepended = input_lengths + 1
        padded_embedding = self.embedding(padded_input_prepended)  # [B, U+1, D]
        packed_padded_embedding = torch.nn.utils.rnn.pack_padded_sequence(
            padded_embedding, input_lengths_prepended, batch_first=True, enforce_sorted=False
        )
        packed_padded_output, hidden = self.lstm(packed_padded_embedding, hidden)
        padded_output, _ = torch.nn.utils.rnn.pad_packed_sequence(packed_padded_output, batch_first=True)
        padded_output = self.fc(padded_output)  # [B, U+1, D]

        return padded_output, hidden

    def forward_wo_prepend(self, padded_input, input_lengths, hidden=None):
        # padded_inputs: [B, U]
        # hidden: inferenceなどで過去の隠れ状態を与える場合に使用

        padded_embedding = self.embedding(padded_input)  # [B, U+1, D]
        packed_padded_embedding = torch.nn.utils.rnn.pack_padded_sequence(
            padded_embedding, input_lengths, batch_first=True, enforce_sorted=False
        )
        packed_padded_output, hidden = self.lstm(packed_padded_embedding, hidden)
        padded_output, _ = torch.nn.utils.rnn.pad_packed_sequence(packed_padded_output, batch_first=True)
        padded_output = self.fc(padded_output)  # [B, U+1, D]

        return padded_output, hidden
