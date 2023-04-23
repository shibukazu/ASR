import torch


class VADAdapter(torch.nn.Module):
    def __init__(self, input_size, hidden_size):
        super().__init__()
        self.w_d = torch.nn.Linear(input_size, hidden_size)
        self.activation = torch.nn.ReLU()
        self.w_u = torch.nn.Linear(hidden_size, input_size)
        self.w_vad = torch.nn.Linear(input_size, 1)
        self.sigmoid = torch.nn.Sigmoid()

    def forward(self, bx):
        # bx: [B, T, D]
        # bvad: [B, T] 各時刻のVADの予測値 (0 ~ 1)
        # by: [B, T, D]
        # 各演算はtimewiseであることに注意

        res = bx

        bx = self.w_d(bx)
        bx = self.activation(bx)
        bx = self.w_u(bx)

        bvad = self.w_vad(bx)
        bvad = self.sigmoid(bvad).squeeze(-1)

        bx = bx + res
        by = bx

        return by, bvad
