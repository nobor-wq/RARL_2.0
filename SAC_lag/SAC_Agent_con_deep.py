import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# Actor网络（策略网络）
class Actor(nn.Module):
    def __init__(self, state_dim, action_dim, max_action):
        super(Actor, self).__init__()
        self.l1 = nn.Linear(state_dim, 256)
        self.l2 = nn.Linear(256, 256)
        self.mean = nn.Linear(256, action_dim)
        self.log_std = nn.Linear(256, action_dim)
        self.max_action = max_action

    def forward(self, state):
        a = F.relu(self.l1(state))
        a = F.relu(self.l2(a))
        mean = self.mean(a)
        log_std = self.log_std(a)
        log_std = torch.clamp(log_std, min=-20, max=2)  # 限制log标准差范围
        return mean, log_std

    def sample(self, state):
        mean, log_std = self.forward(state)
        std = log_std.exp()
        normal = torch.distributions.Normal(mean, std)
        x_t = normal.rsample()  # 重参数化采样
        action = torch.tanh(x_t)  # 将动作压缩到[-1,1]
        action = action * self.max_action  # 缩放到环境动作范围
        log_prob = normal.log_prob(x_t)
        # 修正对数概率（由于tanh变换）
        log_prob -= torch.log(1 - action.pow(2) + 1e-6)
        log_prob = log_prob.sum(1, keepdim=True)
        return action, log_prob, mean, log_std


# Critic网络（Q函数）
class Critic(nn.Module):
    def __init__(self, state_dim, action_dim):
        super(Critic, self).__init__()
        self.l1 = nn.Linear(state_dim + action_dim, 256)
        self.l2 = nn.Linear(256, 256)
        self.l3 = nn.Linear(256, 1)

    def forward(self, state, action):
        sa = torch.cat([state, action], 1)
        q = F.relu(self.l1(sa))
        q = F.relu(self.l2(q))
        q = self.l3(q)
        return q


# SAC-Lag主体实现
class SAC_Lag:
    def __init__(self, state_dim, action_dim, max_action, cost_threshold=0.1):
        self.actor = Actor(state_dim, action_dim, max_action).to(device)
        self.critic1 = Critic(state_dim, action_dim).to(device)
        self.critic2 = Critic(state_dim, action_dim).to(device)
        self.critic1_target = Critic(state_dim, action_dim).to(device)
        self.critic2_target = Critic(state_dim, action_dim).to(device)

        # 初始化目标网络参数
        self.critic1_target.load_state_dict(self.critic1.state_dict())
        self.critic2_target.load_state_dict(self.critic2.state_dict())

        # 优化器
        self.actor_optimizer = torch.optim.Adam(self.actor.parameters(), lr=3e-4)
        self.critic1_optimizer = torch.optim.Adam(self.critic1.parameters(), lr=3e-4)
        self.critic2_optimizer = torch.optim.Adam(self.critic2.parameters(), lr=3e-4)

        # 超参数
        self.max_action = max_action
        self.gamma = 0.99  # 折扣因子
        self.tau = 0.005  # 目标网络软更新系数
        self.cost_threshold = cost_threshold  # 约束阈值

        # 自动调整温度系数alpha
        self.target_entropy = -action_dim
        self.log_alpha = torch.zeros(1, requires_grad=True, device=device)
        self.alpha_optimizer = torch.optim.Adam([self.log_alpha], lr=3e-4)

        # 拉格朗日乘子（使用对数形式确保非负）
        self.log_lagrangian = torch.zeros(1, requires_grad=True, device=device)
        self.lagrangian_optimizer = torch.optim.Adam([self.log_lagrangian], lr=1e-4)

    @property
    def alpha(self):
        return self.log_alpha.exp()

    @property
    def lagrangian(self):
        return self.log_lagrangian.exp()

    def select_action(self, state):
        state = torch.FloatTensor(state.reshape(1, -1)).to(device)
        action, _, _, _ = self.actor.sample(state)
        return action.cpu().data.numpy().flatten()

    def update(self, replay_buffer, batch_size=256):
        # 从经验池采样
        state, action, reward, next_state, done, cost = replay_buffer.sample(batch_size)

        state = torch.FloatTensor(state).to(device)
        next_state = torch.FloatTensor(next_state).to(device)
        action = torch.FloatTensor(action).to(device)
        reward = torch.FloatTensor(reward).to(device).unsqueeze(1)
        done = torch.FloatTensor(done).to(device).unsqueeze(1)
        cost = torch.FloatTensor(cost).to(device).unsqueeze(1)

        # --------------------- 更新Critic网络 ---------------------
        with torch.no_grad():
            # 计算目标Q值
            next_action, next_log_prob, _, _ = self.actor.sample(next_state)
            target_q1 = self.critic1_target(next_state, next_action)
            target_q2 = self.critic2_target(next_state, next_action)
            target_q = torch.min(target_q1, target_q2) - self.alpha * next_log_prob
            target_q = reward + (1 - done) * self.gamma * target_q

        # 当前Q值估计
        current_q1 = self.critic1(state, action)
        current_q2 = self.critic2(state, action)

        # Critic损失
        critic1_loss = F.mse_loss(current_q1, target_q)
        critic2_loss = F.mse_loss(current_q2, target_q)

        # 优化Critic
        self.critic1_optimizer.zero_grad()
        critic1_loss.backward()
        self.critic1_optimizer.step()

        self.critic2_optimizer.zero_grad()
        critic2_loss.backward()
        self.critic2_optimizer.step()

        # --------------------- 更新Actor网络 ---------------------
        new_action, log_prob, _, _ = self.actor.sample(state)

        # 计算Q值
        q1 = self.critic1(state, new_action)
        q2 = self.critic2(state, new_action)
        min_q = torch.min(q1, q2)

        # 计算当前策略的成本（需根据实际问题定义）
        # 示例：假设成本与动作幅值相关
        current_cost = (new_action.pow(2).sum(dim=1, keepdim=True)).mean()

        # Actor损失（含拉格朗日项）
        actor_loss = (self.alpha * log_prob - min_q + self.lagrangian * current_cost).mean()

        # 优化Actor
        self.actor_optimizer.zero_grad()
        actor_loss.backward()
        self.actor_optimizer.step()

        # ----------------- 调整温度系数alpha -----------------
        alpha_loss = -(self.log_alpha * (log_prob + self.target_entropy).detach()).mean()
        self.alpha_optimizer.zero_grad()
        alpha_loss.backward()
        self.alpha_optimizer.step()

        # -------------- 更新拉格朗日乘子 ---------------
        # 计算约束违规量（需根据实际问题调整）
        constraint_violation = current_cost.detach() - self.cost_threshold
        lagrangian_loss = -self.lagrangian * constraint_violation

        self.lagrangian_optimizer.zero_grad()
        lagrangian_loss.backward()
        self.lagrangian_optimizer.step()

        # ----------------- 软更新目标网络 -----------------
        for param, target_param in zip(self.critic1.parameters(), self.critic1_target.parameters()):
            target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)

        for param, target_param in zip(self.critic2.parameters(), self.critic2_target.parameters()):
            target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)


# 经验回放池
class ReplayBuffer:
    def __init__(self, max_size=1e6):
        self.storage = []
        self.max_size = max_size
        self.ptr = 0

    def add(self, state, action, reward, next_state, done, cost):
        if len(self.storage) == self.max_size:
            self.storage[int(self.ptr)] = (state, action, reward, next_state, done, cost)
            self.ptr = (self.ptr + 1) % self.max_size
        else:
            self.storage.append((state, action, reward, next_state, done, cost))

    def sample(self, batch_size):
        ind = np.random.randint(0, len(self.storage), size=batch_size)
        states, actions, rewards, next_states, dones, costs = [], [], [], [], [], []

        for i in ind:
            s, a, r, s_, d, c = self.storage[i]
            states.append(s)
            actions.append(a)
            rewards.append(r)
            next_states.append(s_)
            dones.append(d)
            costs.append(c)

        return (np.array(states), np.array(actions), np.array(rewards),
                np.array(next_states), np.array(dones), np.array(costs))