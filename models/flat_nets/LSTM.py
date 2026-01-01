import torch
from torch import nn
from torch.nn import functional as F

class LSTMModel(nn.Module):
    def __init__(self, input_size: int, hidden_size: int, output_size: int):
        super().__init__()
        self.hidden_size = hidden_size
        self.lstm = nn.LSTM(input_size=input_size, hidden_size=hidden_size, batch_first=True)
        self.linear = nn.Linear(hidden_size, 512)
        self.linear2 = nn.Linear(512, 256)
        self.linear3 = nn.Linear(256, 128)
        self.out = nn.Linear(128, output_size)

    def forward(self, x, h, c):
        x, (h, c) = self.lstm(x, (h, c))
        x = F.relu(self.linear(x))
        x = F.relu(self.linear2(x))
        x = F.relu(self.linear3(x))
        x = self.out(x)
        return x, (h, c)

    def init_hidden(self, batch_size=1, device="cpu"):
        h = torch.zeros(1, batch_size, self.hidden_size, device=device)
        c = torch.zeros(1, batch_size, self.hidden_size, device=device)
        return (h, c)
