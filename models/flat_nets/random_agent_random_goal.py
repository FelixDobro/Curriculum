import torch
from torch import nn
import torch.nn.functional as F
from config import *

class SimpleNet(nn.Module):
    def __init__(self, input_dim):
        super().__init__()
        self.input = nn.Linear(input_dim, 512)
        self.hidden = nn.Linear(512, 256)
        self.hidden2 = nn.Linear(256, 64)
        self.output = nn.Linear(64, NUM_ACTIONS)

    def forward(self, x):
        x = F.relu(self.input(x))
        x = F.relu(self.hidden(x))
        x = F.relu(self.hidden2(x))
        x = self.output(x)
        return x