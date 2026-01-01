import torch
import torch.nn as nn
import torch.nn.functional as F

class PPONet(nn.Module):
    def __init__(self, embedding_dim, hidden_dim, num_actions):
        super().__init__()
        self.hidden_size = hidden_dim
        self.conv_layers = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
        )
        self.fc = nn.Linear(64 * 7 * 7, embedding_dim)  # 288 -> embedding_dim

        # --- Recurrent Core ---
        self.gru = nn.GRU(embedding_dim, hidden_dim, batch_first=True)


        self.extrinsic = nn.Sequential(
            nn.Linear(hidden_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 128),
            nn.ReLU(),
            nn.Linear(128, 1),
        )

        self.policy_net = nn.Sequential(
            nn.Linear(hidden_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 128),
            nn.ReLU(),
            nn.Linear(128, num_actions),
        )

    def forward(self, x, h):
        B, T, C, H, W = x.shape
        x = x.reshape(B*T, C, H, W)
        feats = self.conv_layers(x)
        feats = feats.view(B*T, -1)
        feats = F.relu(self.fc(feats))

        feats = feats.view(B, T, -1)
        out, h = self.gru(feats, h)

        val_extr = self.extrinsic(out)
        policy_logits = self.policy_net(out)

        return val_extr, policy_logits, h, out

    def init_hidden(self, batch_size=1, device="cpu"):
        h = torch.zeros(1, batch_size, self.hidden_size, device=device)
        return h
