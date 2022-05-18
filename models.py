import torch.nn as nn
import torch

class BaseEstimator(nn.Module):
    def __init__(self, state_size, number_of_features, action_space):
        super().__init__()
        self.state_size = state_size
        self.action_space = action_space

        self.model = nn.Sequential(
            nn.Linear(number_of_features * state_size, 32),
            nn.ReLU(),
            nn.Linear(32, 64),
            nn.ReLU(),
            nn.Linear(64, 128),
            nn.ReLU(),
            nn.Linear(128, action_space)
        )

    def forward(self, x):
        x = x.view(x.size(0), -1)

        return self.model(x)

class BaseDuelingEstimator(nn.Module):
    def __init__(self, state_size, number_of_features,action_space):
        super().__init__()

        self.estimator_ = BaseEstimator(state_size, number_of_features, action_space)
        self.adversarial_ = BaseEstimator(state_size, number_of_features, action_space)

    def forward(self, x):
        x = x.view(x.size(0), -1)

        out_est = self.estimator_(x)
        out_adv = self.adversarial_(x)

        out = out_est + out_adv - out_adv.mean()
        
        return out