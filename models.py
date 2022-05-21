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
    def __init__(self, state_size, number_of_features, action_space):
        super().__init__()

        self.estimator_ = BaseEstimator(state_size, number_of_features, action_space)
        self.adversarial_ = BaseEstimator(state_size, number_of_features, action_space)

    def forward(self, x):
        x = x.view(x.size(0), -1)

        out_est = self.estimator_(x)
        out_adv = self.adversarial_(x)

        out = out_est + out_adv - out_adv.mean()
        
        return out

class Attention(nn.Module):
    def __init__(self, feature_dim, step_dim, bias=True, **kwargs):
        super(Attention, self).__init__(**kwargs)

        self.supports_masking = True

        self.bias = bias
        self.feature_dim = feature_dim
        self.step_dim = step_dim
        self.features_dim = 0

        weight = torch.zeros(feature_dim, 1)
        nn.init.kaiming_uniform_(weight)
        self.weight = nn.Parameter(weight)

        if bias:
            self.b = nn.Parameter(torch.zeros(step_dim))

    def forward(self, x, mask=None):
        feature_dim = self.feature_dim
        step_dim = self.step_dim

        eij = torch.mm(
            x.contiguous().view(-1, feature_dim),
            self.weight
        ).view(-1, step_dim)

        if self.bias:
            eij = eij + self.b

        eij = torch.tanh(eij)
        a = torch.exp(eij)

        if mask is not None:
            a = a * mask

        a = a / (torch.sum(a, 1, keepdim=True) + 1e-10)

        weighted_input = x * torch.unsqueeze(a, -1)
        return torch.sum(weighted_input, 1)

class SelectItem(nn.Module):
    def __init__(self, item_index):
        super(SelectItem, self).__init__()
        self._name = 'selectitem'
        self.item_index = item_index

    def forward(self, inputs):
        return inputs[self.item_index]

class BiGRUattentionEstimator(nn.Module):
    def __init__(self, state_size, number_of_features, action_space):
        super().__init__()
        self.state_size = state_size
        self.action_space = action_space
        self.size = number_of_features * self.state_size
        hidden_size = 32
        self.model = nn.Sequential(
            nn.GRU(self.size, self.size, batch_first=False, bidirectional=True, dropout=0.3),
            SelectItem(1),
            nn.GRU(self.size, self.size, batch_first=False, bidirectional=True, dropout=0.3),
            SelectItem(1),
            #Attention(self.size, 10),
            #nn.Flatten(),
            nn.Linear(self.size, 32),
            nn.ReLU(),
            nn.Linear(32, 64),
            nn.ReLU(),
            nn.Linear(64, 128),
            nn.ReLU(),
            nn.Linear(128, self.action_space)
        )

    def forward(self, x):
        x = x.view(x.size(0), -1)
        return self.model(x)