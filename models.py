import torch.nn as nn
import torch

class BaseEstimator(nn.Module):
    def __init__(self, state_size, action_space):
        super().__init__()
        self.state_size = state_size
        self.action_space = action_space

        self.model = nn.Sequential(
            nn.Linear(state_size, 32),
            nn.ReLU(),
            nn.Linear(32, 64),
            nn.ReLU(),
            nn.Linear(64, 128),
            nn.ReLU(),
            nn.Linear(128, action_space)
        )

    def forward(self, x):

        return self.model(x)