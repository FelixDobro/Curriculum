import torch.nn as nn
import torch.nn.functional as F
import torch

class StatePredictor(nn.Module):
    def __init__(self, input_dim, action_dim, latent_dim):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, 512),
            nn.ReLU(),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Linear(256, latent_dim),
        )

        self.predictor = nn.Sequential(
            nn.Linear(latent_dim + action_dim, 512),
            nn.ReLU(),
            nn.Linear(512, 512),
            nn.ReLU(),
            nn.Linear(512, 256),
            nn.Linear(256, latent_dim),
        )

        self.inverse_head = nn.Sequential(
            nn.Linear(2*latent_dim, 512),
            nn.ReLU(),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Linear(256, action_dim),
        )
    def encode(self, x):
        return self.encoder(x)

    def predict(self, x):
        return self.predictor(x)

    def inverse(self, z, z_next):
        return self.inverse_head(torch.cat((z, z_next), dim=2))