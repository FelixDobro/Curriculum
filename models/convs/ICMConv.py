import torch
import torch.nn as nn


class ICMNetConvSmall(nn.Module):
    def __init__(self, hidden_dim, num_actions, embedding_dim):
        super().__init__()

        # --- Conv-Encoder für 7x7x3 ---
        self.conv = nn.Sequential(
            nn.Conv2d(3, 8, kernel_size=3, stride=1),  # (3,7,7) -> (8,5,5)
            nn.ReLU(),
            nn.Conv2d(8, 16, kernel_size=3, stride=1),  # (8,5,5) -> (16,3,3)
            nn.ReLU()
        )
        conv_out_size = 16*3*3 # = 144

        # --- Encoder: Conv + Hidden -> Embedding ---
        self.encoder = nn.Sequential(
            nn.Linear(conv_out_size + hidden_dim, 64),
            nn.ReLU(),
            nn.Linear(64, embedding_dim),
            nn.ReLU()
        )

        # --- Forward Model (predict next embedding) ---
        self.embedding_pred = nn.Sequential(
            nn.Linear(embedding_dim + num_actions, 64),
            nn.ReLU(),
            nn.Linear(64, embedding_dim)
        )

        # --- Inverse Model (predict action) ---
        self.action_pred = nn.Sequential(
            nn.Linear(embedding_dim * 2, 64),
            nn.ReLU(),
            nn.Linear(64, num_actions)
        )

    def forward(self, x, hidden):
        B, T, C, H, W = x.shape
        x = x.view(B * T, C, H, W)
        hidden = hidden.reshape(B * T, -1)

        feats = self.conv(x).view(B * T, -1)
        x = self.encoder(torch.cat([feats, hidden], dim=-1))
        return x.view(B, T, -1)

    def predict_state(self, x, actions):
        return self.embedding_pred(torch.cat((x, actions), dim=-1))

    def predict_action(self, x, x1):
        return self.action_pred(torch.cat((x, x1), dim=-1))
