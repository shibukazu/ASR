import torch


class JointNet(torch.nn.Module):
    def __init__(self, enc_hidden_size, pred_hidden_size, hidden_size, vocab_size):
        super().__init__()
        self.fc_enc = torch.nn.Linear(enc_hidden_size, hidden_size)
        self.fc_pred = torch.nn.Linear(pred_hidden_size, hidden_size)
        self.activation = torch.nn.Tanh()
        self.fc = torch.nn.Linear(hidden_size, vocab_size)

    def forward(self, padded_enc_output, padded_pred_output):
        # enc_out: (batch_size, seq_len, hidden_size)
        # pred_out: (batch_size, seq_len, hidden_size)
        # out: (batch_size, seq_len, vocab_size)
        assert padded_pred_output.dim() == 3 and padded_enc_output.dim() == 3

        padded_enc_output = self.fc_enc(padded_enc_output).unsqueeze(2)
        padded_pred_output = self.fc_pred(padded_pred_output).unsqueeze(1)
        padded_output = padded_enc_output + padded_pred_output
        padded_output = self.activation(padded_output)
        padded_output = self.fc(padded_output)
        return padded_output
