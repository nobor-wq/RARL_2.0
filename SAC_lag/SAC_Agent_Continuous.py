import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import numpy as np


# Gaussian policy network for continuous actions
class GaussianPolicy(nn.Module):
    def __init__(self, state_dim, action_dim, hidden_dim=256, log_std_min=-10, log_std_max=10):
        super(GaussianPolicy, self).__init__()
        self.log_std_min = log_std_min
        self.log_std_max = log_std_max

        self.net = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU()
        )
        self.mean_linear = nn.Linear(hidden_dim, action_dim)
        self.log_std_linear = nn.Linear(hidden_dim, action_dim)

    def forward(self, state):
        x = self.net(state)
        mean = self.mean_linear(x)
        log_std = self.log_std_linear(x)
        log_std = torch.clamp(log_std, self.log_std_min, self.log_std_max)
        return mean, log_std

    def sample(self, state):
        mean, log_std = self.forward(state)
        std = log_std.exp()
        normal = torch.distributions.Normal(mean, std)
        x_t = normal.rsample()  # for reparameterization
        y_t = torch.tanh(x_t)
        action = y_t
        log_prob = normal.log_prob(x_t)
        # Enforcing action bounds
        log_prob -= torch.log(1 - y_t.pow(2) + 1e-6)
        log_prob = log_prob.sum(1, keepdim=True)
        return action, log_prob, torch.tanh(mean)

# Q-network
class QNetwork(nn.Module):
    def __init__(self, state_dim, action_dim, hidden_dim=256):
        super(QNetwork, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(state_dim + action_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1)
        )

    def forward(self, state, action):
        x = torch.cat([state, action], dim=1)
        return self.net(x)

class SAC_Lag:
    def __init__(self,
                 state_dim,
                 action_dim,
                 action_range=1.0,
                 hidden_dim=256,
                 lr=3e-4,
                 gamma=0.99,
                 tau=0.005,
                 target_entropy=None,
                 device=torch.device("cpu")):
        self.device = device
        self.gamma = gamma
        self.tau = tau
        self.action_range = action_range
        self.cost_limit = 1

        # Actor
        self.policy = GaussianPolicy(state_dim, action_dim, hidden_dim).to(device)
        self.policy_optim = optim.Adam(self.policy.parameters(), lr=lr)

        # Critics
        self.critic1 = QNetwork(state_dim, action_dim, hidden_dim).to(device)
        self.critic2 = QNetwork(state_dim, action_dim, hidden_dim).to(device)
        self.critic1_optim = optim.Adam(self.critic1.parameters(), lr=lr)
        self.critic2_optim = optim.Adam(self.critic2.parameters(), lr=lr)

        # Target critics
        self.critic1_target = QNetwork(state_dim, action_dim, hidden_dim).to(device)
        self.critic2_target = QNetwork(state_dim, action_dim, hidden_dim).to(device)
        self.critic1_target.load_state_dict(self.critic1.state_dict())
        self.critic2_target.load_state_dict(self.critic2.state_dict())

        #cost Critic
        self.cost_critic = QNetwork(state_dim, action_dim, hidden_dim).to(device)
        self.cost_critic_optim = optim.Adam(self.cost_critic.parameters(), lr=lr)

        # 成本Critic目标网络
        self.cost_critic_target = QNetwork(state_dim, action_dim, hidden_dim).to(device)
        self.cost_critic_target.load_state_dict(self.cost_critic.state_dict())

        self.log_lambda = torch.zeros(1, requires_grad=True, device=device)
        self.lambda_optim = optim.Adam([self.log_lambda], lr=lr)

        # Temperature parameter (Lagrange for entropy)
        if target_entropy is None:
            self.target_entropy = -action_dim  # heuristic
        else:
            self.target_entropy = target_entropy
        # Log alpha
        self.log_alpha = torch.zeros(1, requires_grad=True, device=device)
        self.alpha_optim = optim.Adam([self.log_alpha], lr=lr)

    def select_action(self, state, deterministic=False):
        state = torch.FloatTensor(state).unsqueeze(0).to(self.device)
        if deterministic:
            _, _, action = self.policy.sample(state)
        else:
            action, _, _ = self.policy.sample(state)
        return action.detach()

    def save_actor(self, save_path):
        """仅保存 actor（策略网络）的参数"""
        torch.save(self.policy.state_dict(), save_path)

    def load_actor(self, load_path):
        """加载 actor（策略网络）的参数"""
        state_dict = torch.load(load_path, map_location=self.device)
        self.policy.load_state_dict(state_dict)
        self.policy.to(self.device)

    def update_parameters(self, replay_buffer, batch_size):
        # Sample a batch from replay buffer
        state, action, next_state, reward, cost, done = replay_buffer.sample(batch_size)

        state = state.to(self.device)
        action = action.to(self.device)
        next_state = next_state.to(self.device)
        reward = reward.view(-1, 1).to(self.device)
        cost = cost.view(-1, 1).to(self.device)
        done = done.view(-1, 1).to(self.device)

        # Update critics
        with torch.no_grad():
            next_action, next_log_pi, _ = self.policy.sample(next_state)
            q1_next = self.critic1_target(next_state, next_action)
            q2_next = self.critic2_target(next_state, next_action)
            q_next = torch.min(q1_next, q2_next) - torch.exp(self.log_alpha) * next_log_pi
            q_target = reward + (1 - done) * self.gamma * q_next

        q1 = self.critic1(state, action)
        q2 = self.critic2(state, action)
        critic1_loss = F.mse_loss(q1, q_target)
        critic2_loss = F.mse_loss(q2, q_target)

        self.critic1_optim.zero_grad()
        critic1_loss.backward()
        self.critic1_optim.step()

        self.critic2_optim.zero_grad()
        critic2_loss.backward()
        self.critic2_optim.step()

        # 在无梯度环境下计算成本目标：
        with torch.no_grad():
            next_action, next_log_pi, _ = self.policy.sample(next_state)
            cost_next = self.cost_critic_target(next_state, next_action)
            cost_target = cost + (1 - done) * self.gamma * cost_next

        # 成本 Critic 预测：
        cost_pred = self.cost_critic(state, action)
        cost_critic_loss = F.mse_loss(cost_pred, cost_target)

        # 优化成本 Critic
        self.cost_critic_optim.zero_grad()
        cost_critic_loss.backward()
        self.cost_critic_optim.step()

        # Update policy
        new_action, log_pi, _ = self.policy.sample(state)
        q1_new = self.critic1(state, new_action)
        q2_new = self.critic2(state, new_action)
        q_new = torch.min(q1_new, q2_new)

        lambda_val = torch.exp(self.log_lambda)
        # 预计成本
        cost_new = self.cost_critic(state, new_action)
        # 约束项：如果 cost_new > cost_limit，就有惩罚
        constraint_term = lambda_val * (cost_new - self.cost_limit).mean()

        policy_loss = (torch.exp(self.log_alpha) * log_pi - q_new).mean() + constraint_term

        # policy_loss = (torch.exp(self.log_alpha) * log_pi - q_new).mean()

        self.policy_optim.zero_grad()
        policy_loss.backward()
        self.policy_optim.step()

        # Update alpha (temperature)
        alpha_loss = -(self.log_alpha * (log_pi + self.target_entropy).detach()).mean()

        self.alpha_optim.zero_grad()
        alpha_loss.backward()
        self.alpha_optim.step()

        # λ 的损失（注意要梯度上升，所以符号取负）
        lambda_loss = -(self.log_lambda * (cost_new.detach() - self.cost_limit)).mean()

        self.lambda_optim.zero_grad()
        lambda_loss.backward()
        self.lambda_optim.step()


        # Soft update target networks
        for target_param, param in zip(self.critic1_target.parameters(), self.critic1.parameters()):
            target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)
        for target_param, param in zip(self.critic2_target.parameters(), self.critic2.parameters()):
            target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)
        for target_param, param in zip(self.cost_critic_target.parameters(), self.cost_critic.parameters()):
            target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)

        return critic1_loss.item(), critic2_loss.item(), policy_loss.item(), alpha_loss.item(), torch.exp(self.log_alpha).item()

