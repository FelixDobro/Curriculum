import torch
import torch.nn as nn
import torch.nn.functional as F

class ConvDoubleQ(nn.Module):
    def __init__(self, embedding_dim, hidden_dim, num_actions):
        super().__init__()
        self.hidden_size = hidden_dim
        self.conv_layers = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1),  # (7x7 -> 7x7)
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1),           # (7x7 -> 7x7)
            nn.ReLU(),
        )
        self.fc = nn.Linear(64 * 7 * 7, embedding_dim)  # 288 -> embedding_dim

        # --- Recurrent Core ---
        self.gru = nn.GRU(embedding_dim, hidden_dim, batch_first=True)


        self.intrinsic = nn.Linear(hidden_dim, num_actions)

        self.extrinsic = nn.Linear(hidden_dim, num_actions)


    def forward(self, x, h):
        B, T, C, H, W = x.shape
        x = x.reshape(B*T, C, H, W)         # Merge Batch & Time
        feats = self.conv_layers(x)             # (B*T, 32, 3, 3)
        feats = feats.view(B*T, -1)      # Flatten
        feats = F.relu(self.fc(feats))   # (B*T, embedding_dim)

        feats = feats.view(B, T, -1)     # (B, T, embedding_dim)

        out, h = self.gru(feats, h)      # (B, T, hidden_dim), h: (1, B, hidden_dim)

        qvals_intr = self.intrinsic(out)         # (B, T, num_actions)
        qvals_extr = self.extrinsic(out)
        return qvals_intr, qvals_extr, h, out

    def init_hidden(self, batch_size=1, device="cpu"):
        h = torch.zeros(1, batch_size, self.hidden_size, device=device)
        return h
