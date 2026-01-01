import torch
from torch import nn
from torch.nn import functional as F


class Embeddings(nn.Module):
    def __init__(self, input_size: int, output_size: int):
        super().__init__()
        self.input = nn.Linear(input_size, 256)
        self.output = nn.Linear(256, output_size)

    def forward(self, x):
        # x: (batch, seq_len, input_size)
        B, T, D = x.shape
        x = x.view(B * T, D)                     # (B*T, D)
        x = F.relu(self.input(x))                # (B*T, 256)
        x = self.output(x)                       # (B*T, output_size)
        return x.view(B, T, -1)                  # (B, T, output_size)
