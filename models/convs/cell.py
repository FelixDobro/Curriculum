import torch
import torch.nn as nn
import torch.nn.functional as F

class PPONetCell(nn.Module):
    def __init__(self, embedding_dim, hidden_dim, num_actions):
        super().__init__()
        self.hidden_size = hidden_dim
        self.conv_layers = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
        )
        self.fc = nn.Linear(64 * 7 * 7, embedding_dim)  # 288 -> embedding_dim

        # --- Recurrent Core ---
        self.gru = nn.GRUCell(embedding_dim, hidden_dim)
        self.layer_norm = nn.LayerNorm(hidden_dim)


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
       
        feats = self.conv_layers(x)
        feats = feats.reshape(feats.size(0), -1)
        feats = F.relu(self.fc(feats))
        h = self.gru(feats, h)

        val_extr = self.extrinsic(h)
        policy_logits = self.policy_net(h)

        return val_extr, policy_logits, h

    def only_policy(self, x, h):
        feats = self.conv_layers(x)
        feats = feats.reshape(feats.size(0), -1)
        feats = F.relu(self.fc(feats))
      
        h = self.gru(feats, h)
       
  
        policy_logits = self.policy_net(h)

        return policy_logits, h

    def init_hidden(self, batch_size=1, device="cpu"):
        h = torch.zeros(batch_size, self.hidden_size, device=device)
        return h
