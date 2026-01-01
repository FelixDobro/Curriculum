import torch
from torch import nn
from torch.nn import functional as F

class GRUModel(nn.Module):
    def __init__(self, input_size: int, hidden_size: int, output_size: int):
        super().__init__()
        self.hidden_size = hidden_size
        self.gru = nn.GRU(input_size=input_size, hidden_size=hidden_size, batch_first=True)
        self.seq = nn.Sequential(
            nn.Linear(hidden_size, 256),
            nn.ReLU(),
            nn.Linear(256, 64),
            nn.ReLU(),
            nn.Linear(64, output_size)
        )


    def forward(self, x, h):
        hidden, h = self.gru(x, h)
        x = self.seq(hidden)
        return x, h, hidden

    def init_hidden(self, batch_size=1, device="cpu"):
        h = torch.zeros(1, batch_size, self.hidden_size, device=device)
        return h
