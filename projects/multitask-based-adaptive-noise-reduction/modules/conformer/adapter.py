import torch


class BottleneckLinearAdapter(torch.nn.Module):
    def __init__(self, input_size, hidden_size):
        super().__init__()
        self.w_d = torch.nn.Linear(input_size, hidden_size)
        self.activation = torch.nn.ReLU()
        self.w_u = torch.nn.Linear(hidden_size, input_size)

    def forward(self, bx):

        bx = self.w_d(bx)
        bx = self.activation(bx)
        bx = self.w_u(bx)

        return bx


class LinearAdapter(torch.nn.Module):
    def __init__(self, input_size, hidden_size):
        super().__init__()
        self.fc = torch.nn.Linear(input_size, hidden_size)

    def forward(self, bx):

        bx = self.fc(bx)

        return bx