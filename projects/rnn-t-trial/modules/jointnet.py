import torch


class JointNet(torch.nn.Module):
    def __init__(self, enc_out_size, pred_out_size, hidden_size, vocab_size):
        super().__init__()
        self.fc1 = torch.nn.Linear(enc_out_size, hidden_size)
        self.activation = torch.nn.Tanh()
        self.fc2 = torch.nn.Linear(hidden_size, vocab_size)

    def forward(self, padded_enc_output, padded_pred_output):
        # enc_out: (batch_size, seq_len, hidden_size)
        # pred_out: (batch_size, seq_len, hidden_size)
        # out: (batch_size, seq_len, vocab_size)
        assert padded_pred_output.dim() == 3 and padded_enc_output.dim() == 3
        enc_length = padded_enc_output.size(1)
        pred_length = padded_pred_output.size(1)
        padded_enc_output = padded_enc_output.unsqueeze(2)
        padded_pred_output = padded_pred_output.unsqueeze(1)

        padded_enc_output = padded_enc_output.repeat(1, 1, pred_length, 1)
        padded_pred_output = padded_pred_output.repeat(1, enc_length, 1, 1)
        padded_output = padded_enc_output + padded_pred_output
        # padded_concat_output = torch.cat((padded_enc_output, padded_pred_output), dim=-1)
        # padded_output = self.fc1(padded_concat_output)
        padded_output = self.fc1(padded_output)
        padded_output = self.activation(padded_output)
        padded_output = self.fc2(padded_output)
        return padded_output
