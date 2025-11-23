"""
Policy network for FNI Class
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Normal


class Module(nn.Module):
    def __call__(self, *args, **kwargs):
        args = [x if isinstance(x, torch.Tensor) else x for x in args]
        kwargs = {k: v if isinstance(v, torch.Tensor) else v for k, v in kwargs.items()}
        return super().__call__(*args, **kwargs)

    def save(self, f, prefix='', keep_vars=False):
        state_dict = self.state_dict(prefix=prefix, keep_vars=keep_vars)
        torch.save(state_dict, f)

    def load(self, f, map_location='', strict=True):
        state_dict = torch.load(f, map_location=map_location)
        self.load_state_dict(state_dict, strict=strict)

class FniNet(Module):
    def __init__(self, state_dim, action_dim, hidden_sizes=256, min_log_std=-10.0, max_log_std=10.0):
        super(FniNet, self).__init__()
        self.fc1 = nn.Linear(state_dim, hidden_sizes)
        self.fc2 = nn.Linear(hidden_sizes, hidden_sizes)

        self.mu_head = nn.Linear(hidden_sizes, action_dim)
        self.log_std_head = nn.Linear(hidden_sizes, action_dim)

        self.min_log_std = min_log_std
        self.max_log_std = max_log_std

    def forward(self, x):
        x = torch.as_tensor(x, dtype=torch.float32)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))

        mu = torch.tanh(self.mu_head(x))
        std = torch.exp(torch.clamp(self.log_std_head(x), self.min_log_std, self.max_log_std)).sqrt()

        action = mu + std * torch.randn_like(mu)
        action = torch.clamp(action, -1.0, 1.0)

        return mu, std, action

class DARRLNet(torch.nn.Module):
    def __init__(self, state_dim, action_dim, hidden_dim=256):
        super(DARRLNet, self).__init__()
        self.fc1 = torch.nn.Linear(state_dim, hidden_dim)
        self.fc_mu = torch.nn.Linear(hidden_dim, action_dim)
        self.fc_std = torch.nn.Linear(hidden_dim, action_dim)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        mu = self.fc_mu(x)
        std = F.softplus(self.fc_std(x))
        dist = Normal(mu, std)
        normal_sample = dist.rsample()  # rsample()是重参数化采样
        log_prob = dist.log_prob(normal_sample)
        action = torch.tanh(normal_sample)
        # 计算tanh_normal分布的对数概率密度
        log_prob = log_prob - torch.log(1 - torch.tanh(action).pow(2) + 1e-7)
        mean_action = torch.tanh(mu)
        # action = action * self.action_bound
        return mean_action, log_prob, action

class IGCARLNet(nn.Module):
    def __init__(self, state_dim, action_dim, hidden_sizes=256, min_log_std=-5.0, max_log_std=2.0):
        super(IGCARLNet, self).__init__()
        self.fc1 = nn.Linear(state_dim, hidden_sizes)
        self.fc2 = nn.Linear(hidden_sizes, hidden_sizes)

        self.mu_head = nn.Linear(hidden_sizes, action_dim)
        self.log_std_head = nn.Linear(hidden_sizes, action_dim)

        self.min_log_std = min_log_std
        self.max_log_std = max_log_std

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))

        mu = torch.tanh(self.mu_head(x))
        std = torch.exp(torch.clamp(self.log_std_head(x), self.min_log_std, self.max_log_std)).sqrt()

        action = mu + std * torch.randn_like(mu)
        action = torch.clamp(action, -1.0, 1.0)

        return mu, std, action
