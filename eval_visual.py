import gymnasium as gym
import numpy as np
from stable_baselines3 import SAC, PPO, TD3  # 或者您使用的其他算法
from gymnasium.wrappers import TimeLimit
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.utils import obs_as_tensor

import Environment.environment
import os
import torch as th
import argparse
from FGSM import *
from DARRLNetworkParams import ActorNet, ActorNet_adv, SAC_lag_Net, FniNet
from PIL import Image

import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from scipy.stats import norm
from matplotlib.colors import LogNorm


import random

parser = argparse.ArgumentParser()

parser.add_argument('--addition_msg', default="", help='additional message of the training process')

parser.add_argument('--age_path', default="./models/")
parser.add_argument('--adv_path', default="./models/")
parser.add_argument('--attack_method', default="fgsm", help='which attack method to be applied')
parser.add_argument('--action_dim', default=1)
parser.add_argument('--max_a', default=7.6, help='Maximum Acceleration')
parser.add_argument('--print_interval', default=10)
parser.add_argument('--speed_range', default=15.0, help='Maximum speed')
parser.add_argument('--state_dim', default=26)
parser.add_argument('--train_step', type=int, default=1, help='number of training episodes')
parser.add_argument('--T_horizon', type=int, default=30, help='number of training steps per episode')
parser.add_argument('--env_name', default="TrafficEnv3-v1", help='name of the environment to run')
parser.add_argument('--adv_algo', default="PPO", help='training adv algorithm')
parser.add_argument('--algo', default="FNI", help='training algorithm')
parser.add_argument('--epsilon', type=float, default=0.05, help='扰动强度')
parser.add_argument('--attack', action='store_true', help='control n_rollout_steps, for PPO')
parser.add_argument('--algo_name', default="defender_v253_20250801_1455_1_0_001.pth", help='defender algorithm')
parser.add_argument('--device', default="cuda:0", help='cpu or cuda:0 pr cuda:1')
parser.add_argument('--adv_algo_name', default="attacker_v177_20250808_2238_1_0_001.pth", help='attack algorithm')
parser.add_argument('--seed', type=int, default=1, help='random seed for network')
parser.add_argument('--env_seed', type=int, default=1, help='random seed for env')
parser.add_argument('--method', default="m1", help='防御者约束方法')

args = parser.parse_args()

device = th.device(args.device)

# 创建环境
if args.env_name == "TrafficEnv8-v0":
    env = gym.make(args.env_name)
else:
    env = gym.make(args.env_name, attack=False)
env = TimeLimit(env, max_episode_steps=args.T_horizon)
env = Monitor(env)
env.unwrapped.start()

# print("="*30)
# print(f"正在运行第 {idx+1} 对防御者-攻击者组合")
# print("="*30)



if args.attack:
    if args.algo == 'drl':
        adv_model_path = os.path.join(args.adv_path, "TrafficEnv3-v1", '2000', args.algo, str(args.epsilon), str(args.method),
                                      str(args.seed), 'attacker/attacker.pth')
    else:
        adv_model_path = os.path.join(args.adv_path, "TrafficEnv3-v1", '2000', args.algo, str(args.epsilon), str(args.seed)
                                      , 'attacker/attacker.pth')
    model = ActorNet_adv(state_dim=26, action_dim=1).to(device)
    model.load_state_dict(torch.load(adv_model_path, map_location=device))
    model.eval()


# 加载训练好的自动驾驶模型
model_path = os.path.join(args.age_path, "TrafficEnv3-v1", '2000', args.algo, 'defender', 'lunar_baseline')

if args.algo == 'PPO':
    print('*******************algo is PPO*******************')
    trained_agent = PPO.load(model_path, device=device)
elif args.algo == 'SAC':
    print('*******************algo is SAC*******************')
    trained_agent = SAC.load(model_path, device=device)
elif args.algo == 'SAC_lag':
    print('*******************algo is SAC_lag*******************')
    trained_agent = SAC_lag_Net(26, 1)
    state_dict = torch.load(model_path+".pt", map_location=device)
    trained_agent.load_state_dict(state_dict)
    trained_agent.eval()
    trained_agent.to(device)  # 再次确保
elif args.algo == 'TD3':
    print('*******************algo is TD3*******************')
    trained_agent = TD3.load(model_path, device=device)
elif args.algo == 'drl':
    print('*******************algo is drl*******************')
    model_path_drl = os.path.join(args.age_path, "TrafficEnv3-v1", '2000', args.algo, str(args.epsilon), str(args.method), str(args.seed), 'defender/defender.pth')
    trained_agent = ActorNet(state_dim=26, action_dim=1).to(device)
    trained_agent.load_state_dict(torch.load(model_path_drl, map_location=device))
    trained_agent.eval()
#
elif args.algo == 'DARRL':
    trained_agent = FniNet(26,1)
    model_path_drl = os.path.join(args.age_path, "TrafficEnv3-v1", '2000', args.algo, 'defender', 'policy2000_actor.pth')
    state_dict = torch.load(model_path_drl, map_location=device)
    trained_agent.load_state_dict(state_dict)
    trained_agent.eval()
    trained_agent.to(device)  # 再次确保
elif args.algo == "FNI":
    trained_agent = FniNet(26, 1)
    score = f"policy_v{411}"
    model_path_drl = os.path.join('models', "TrafficEnv3-v1", '2000', args.algo, 'defender', score) + '.pth'
    state_dict = torch.load(model_path_drl, map_location=device)
    trained_agent.load_state_dict(state_dict)
    trained_agent.eval()
    trained_agent.to(device)  # 再次确保

# 进行验证
rewards = []
steps = []

maxSpeed = 15.0
ct = 0
sn = 0
sat = 0
speed_list = []
attack_count_list = []
mean_attack_reward_list = []

# pic save path
if args.algo == 'drl':
    save_dir = os.path.join(args.adv_path, "TrafficEnv3-v1", '2000', args.algo, str(args.epsilon), str(args.method),
                                      str(args.seed), 'render')
else:
    save_dir = os.path.join(args.adv_path, "TrafficEnv3-v1", '2000', args.algo, str(args.epsilon), str(args.seed), 'render')
if os.path.exists(save_dir):
    for filename in os.listdir(save_dir):
        file_path = os.path.join(save_dir, filename)
        if os.path.isfile(file_path) or os.path.islink(file_path):
            os.unlink(file_path)
        elif os.path.isdir(file_path):
            shutil.rmtree(file_path)
else:
    os.makedirs(save_dir)
for episode in range(args.train_step):
    obs, info = env.reset(seed=args.env_seed)
    img = env.render()
    speed = 0
    episode_reward = 0
    episode_steps = 0
    episode_mean = []
    episode_std = []
    episode_mean_after_attack = []
    episode_std_after_attack = []
    action_dist_records = {args.algo: {'mean': [], 'std': []}}
    action_dist_records_after_attack = {args.algo: {'mean': [], 'std': []}}
    for _ in range(args.T_horizon):
        obs_tensor = obs_as_tensor(obs, device)
        if args.attack:
            speed_list.append(obs[-2])
            with th.no_grad():
                if args.algo in ('FNI', 'DARRL'):
                    mu, std, _ = trained_agent(obs_tensor)  # 输出均值和std
                    mu = mu.cpu().numpy()
                    std = std.cpu().numpy()
                elif args.algo == 'drl':
                    mu, std, _ = trained_agent(obs_tensor)
                elif args.algo == 'SAC_lag':
                    mu, log_std = trained_agent.forward(obs_tensor)
                    std = log_std.exp()
                    mu = torch.tanh(mu)
                elif args.algo == 'PPO':
                    obs_tensor_1 = th.as_tensor(obs, device=trained_agent.device).unsqueeze(0)
                    dist = trained_agent.policy.get_distribution(obs_tensor_1)
                    # 获取均值和标准差
                    mu = dist.distribution.mean.detach().cpu().numpy()  # shape: (1, action_dim)
                    mu = np.tanh(mu)
                    std = dist.distribution.stddev.detach().cpu().numpy()
                    actions, _ = trained_agent.predict(obs, deterministic=True)
                    print('mu', mu, 'std', std, 'actions', actions)
                elif args.algo == 'SAC':
                    obs_tensor_1 = th.as_tensor(obs, device=trained_agent.device).unsqueeze(0)
                    with torch.no_grad():
                        # policy.forward 返回 (actions, log_prob)
                        mu, log_std, _ = trained_agent.actor.get_action_dist_params(obs_tensor_1)
                        std = torch.exp(log_std)
                    actions, _ = trained_agent.predict(obs, deterministic=True)
                    print('mu', mu, 'std', std, 'actions', actions)
                elif args.algo == 'TD3':
                    obs_tensor_1 = th.as_tensor(obs, device=trained_agent.device).unsqueeze(0)
                    with torch.no_grad():
                        # obs_tensor: torch.Tensor([batch, obs_dim])
                        mu = trained_agent.actor(obs_tensor_1)
                        # 对应的 std 显式设置为 0.2
                        std = th.full_like(mu, 0.2)
                    actions, _ = trained_agent.predict(obs, deterministic=True)
                    print('mu', mu, 'std', std, 'actions', actions)
                # 记录
                episode_mean.append(mu.item() if mu.size == 1 else mu.tolist())
                episode_std.append(std.item() if std.size == 1 else std.tolist())
                # 获取动作执行环境
                actions = mu  # 可直接用均值作为动作
                if isinstance(actions, np.ndarray):
                    # 假设 action 是一个标量数组，比如 array([0])，用 item() 取标量
                    actions_tensor = th.tensor(actions, device=obs_tensor.device)
                elif isinstance(actions, torch.Tensor):
                    actions_tensor = actions
                adv_actions, _, _  = model(obs_tensor)
            # print(episode_steps, 'attack', 'Victim action is', actions, 'adv actions is', adv_actions)

            if args.attack_method == 'fgsm':
                if args.algo == 'drl' or args.algo == 'DARRL' or args.algo == 'SAC_lag' or args.algo == 'FNI':
                    # print("states: ", obs_tensor)
                    adv_state = FGSM_vdarrl(adv_actions, victim_agent=trained_agent,last_state=obs_tensor,
                                            algo=args.algo, epsilon=args.epsilon, device=args.device)
                    # print("adv_states: ", adv_state)
                else:
                    adv_state = FGSM_v2(adv_actions, victim_agent=trained_agent, last_state=obs_tensor,
                                        epsilon=args.epsilon, device=args.device)

            with th.no_grad():
                if args.algo in ('FNI', 'DARRL'):
                    mu, std, _ = trained_agent(adv_state)  # 输出均值和std
                    mu = mu.cpu().numpy()
                    std = std.cpu().numpy()
                    adv_action_fromState, _, _ = trained_agent(adv_state)
                    actions = adv_action_fromState.detach().cpu().numpy()
                elif args.algo == 'drl':
                    mu, std, _ = trained_agent(adv_state)
                    adv_action_fromState, _, _ = trained_agent(adv_state)
                    actions = adv_action_fromState.detach().cpu().numpy()
                elif args.algo == 'PPO':
                    adv_state = adv_state.detach().cpu().numpy()
                    obs_tensor_1 = th.as_tensor(adv_state, device=trained_agent.device).unsqueeze(0)
                    dist = trained_agent.policy.get_distribution(obs_tensor_1)
                    # 获取均值和标准差
                    mu = dist.distribution.mean.detach().cpu().numpy()  # shape: (1, action_dim)
                    mu = np.tanh(mu)
                    std = dist.distribution.stddev.detach().cpu().numpy()
                    actions, _ = trained_agent.predict(adv_state, deterministic=True)
                    print('mu', mu, 'std', std, 'actions', actions)
                elif args.algo == 'SAC':
                    adv_state = adv_state.detach().cpu().numpy()
                    obs_tensor_1 = th.as_tensor(adv_state, device=trained_agent.device).unsqueeze(0)
                    with torch.no_grad():
                        # policy.forward 返回 (actions, log_prob)
                        mu, log_std, _ = trained_agent.actor.get_action_dist_params(obs_tensor_1)
                        std = torch.exp(log_std)
                    actions, _ = trained_agent.predict(adv_state, deterministic=True)
                    print('mu', mu, 'std', std, 'actions', actions)
                elif args.algo == 'TD3':
                    adv_state = adv_state.detach().cpu().numpy()
                    obs_tensor_1 = th.as_tensor(adv_state, device=trained_agent.device).unsqueeze(0)
                    with torch.no_grad():
                        # obs_tensor: torch.Tensor([batch, obs_dim])
                        mu = trained_agent.actor(obs_tensor_1)
                        std = th.full_like(mu, 0.2)
                    actions, _ = trained_agent.predict(adv_state, deterministic=True)
                    print('mu', mu, 'std', std, 'actions', actions)
                elif args.algo == 'SAC_lag':
                    # adv_state = adv_state.detach().cpu().numpy()
                    # obs_tensor_1 = th.as_tensor(adv_state, device=device).unsqueeze(0)
                    mu, log_std = trained_agent.forward(adv_state)
                    std = log_std.exp()
                    mu = torch.tanh(mu)
                    _, _, adv_action_fromState = trained_agent.sample(adv_state)
                    actions = adv_action_fromState.detach().cpu().numpy()
                    print('mu', mu, 'std', std, 'actions', adv_action_fromState)
                # print(episode_steps, 'attack', '{} action is'.format(args.attack_method), adv_action_fromState)
                # 记录
                episode_mean_after_attack.append(mu.item() if mu.size == 1 else mu.tolist())
                episode_std_after_attack.append(std.item() if std.size == 1 else std.tolist())
            obs, reward, done, T, info = env.step(actions)
        else:
            # print("state: ", obs_tensor)
            speed_list.append(obs[-2])
            with th.no_grad():
                if args.algo in ('FNI', 'DARRL'):
                    mu, std, _ = trained_agent(obs_tensor)  # 输出均值和std
                    mu = mu.cpu().numpy()
                    std = std.cpu().numpy()
                elif args.algo == 'drl':
                    mu, std, _ = trained_agent(obs_tensor)
                elif args.algo == 'SAC_lag':
                    mu, log_std = trained_agent.forward(obs_tensor)
                    mu = torch.tanh(mu)
                    std = log_std.exp()
                elif args.algo == 'PPO':
                    obs_tensor_1 = th.as_tensor(obs, device=trained_agent.device).unsqueeze(0)
                    dist = trained_agent.policy.get_distribution(obs_tensor_1)
                    # 获取均值和标准差
                    mu = dist.distribution.mean.detach().cpu().numpy()  # shape: (1, action_dim)
                    std = dist.distribution.stddev.detach().cpu().numpy()
                    actions = trained_agent.predict(obs, deterministic=True)
                    print('mu', mu, 'std', std, 'actions', actions)
                elif args.algo == 'SAC':
                    obs_tensor_1 = th.as_tensor(obs, device=trained_agent.device).unsqueeze(0)
                    with torch.no_grad():
                        # policy.forward 返回 (actions, log_prob)
                        mu, log_std, _ = trained_agent.actor.get_action_dist_params(obs_tensor_1)
                        std = torch.exp(log_std)
                    actions = trained_agent.predict(obs, deterministic=True)
                    print('mu', mu, 'std', std, 'actions', actions)
                elif args.algo == 'TD3':
                    obs_tensor_1 = th.as_tensor(obs, device=trained_agent.device).unsqueeze(0)
                    with torch.no_grad():
                        # obs_tensor: torch.Tensor([batch, obs_dim])
                        mu = trained_agent.actor(obs_tensor_1)
                        std = th.full_like(mu, 0.2)
                    actions = trained_agent.predict(obs, deterministic=True)
                    print('mu', mu, 'std', std, 'actions', actions)

            # print('action is ', actions)
            actions = mu
            # 记录
            episode_mean.append(mu.item() if mu.size == 1 else mu.tolist())
            episode_std.append(std.item() if std.size == 1 else std.tolist())
            obs, reward, done, T, info = env.step(actions)
        episode_reward += reward
        episode_steps += 1
        img = env.render()
        img = Image.fromarray(img)
        img = img.copy()
        img.save(f'{save_dir}/{episode_steps}.jpg')
        if done:
            ct += 1
            break

    xa = info['x_position']
    ya = info['y_position']
    if (args.env_name == 'TrafficEnv1-v0' or args.env_name == 'TrafficEnv3-v1' or args.env_name == 'TrafficEnv6-v0'
            or args.env_name == 'TrafficEnv3-v2' or args.env_name == 'TrafficEnv3-v3' or args.env_name == 'TrafficEnv3-visual'):
        if xa < -50.0 and ya > 4.0 and done is False:
            sn += 1
    elif args.env_name == 'TrafficEnv2-v0':
        if xa > 50.0 and ya > -5.0 and done is False:
            sn += 1
    elif args.env_name == 'TrafficEnv4-v0':
        if ya < -50.0 and done is False:
            sn += 1
    elif args.env_name == 'TrafficEnv8-v0':
        if ya == 10.0 and done is False:
            sn += 1
    rewards.append(episode_reward)
    steps.append(episode_steps)

    # 保存整个 episode
    action_dist_records[args.algo]['mean'].append(episode_mean)
    action_dist_records[args.algo]['std'].append(episode_std)
    if args.attack:
        action_dist_records_after_attack[args.algo]['mean'].append(episode_mean_after_attack)
        action_dist_records_after_attack[args.algo]['std'].append(episode_std_after_attack)

# 计算平均奖励和步数
mean_reward = np.mean(rewards)
std_reward = np.std(rewards)
mean_steps = np.mean(steps)
std_steps = np.std(steps)

# 计算碰撞率
cr = ct / args.train_step * 100
sr = sn / args.train_step * 100

# 计算平均速度
mean_speed = np.mean(speed_list)
std_speed = np.std(speed_list)
print(f"Mean reward: {mean_reward:.2f} +/- {std_reward:.2f}")
print(f"Mean steps: {mean_steps:.2f} +/- {std_steps:.2f}")
print(f"Mean speed: {mean_speed * maxSpeed:.2f} +/- {std_speed * maxSpeed:.2f}")
# print(f"Mean attack times: {mean_attack_times:.2f} +/- {std_attack_times:.2f}")
print(f"Collision rate: {cr:.2f}")
print(f"Success rate: {sr:.2f}")


# 绘图
# 全局字体大小
plt.rcParams.update({
    "font.size": 14,       # 图例、坐标轴刻度、标题等字体大小
    "axes.titlesize": 14,  # 子图标题
    "axes.labelsize": 14,  # x/y 轴标签
    "xtick.labelsize": 14, # x 轴刻度
    "ytick.labelsize": 14, # y 轴刻度
    "legend.fontsize": 14  # 图例
})

# 将所有 episode 均值和std堆叠
mean_all = np.array(action_dist_records[args.algo]['mean'], dtype=object)  # 每个元素是(steps,)
std_all = np.array(action_dist_records[args.algo]['std'], dtype=object)
print('Mean list', mean_all)
print('Std list', std_all)

num_bins = 100
bin_edges = np.linspace(-1, 1, num_bins+1)
heatmap_data = np.full((num_bins, args.T_horizon), np.nan)

# ---- 控制 y 轴的刻度 ----
# 固定在 [-1, 1]，一共 11 个刻度
ytick_positions = np.linspace(0, num_bins - 1, 11)   # 坐标位置
ytick_labels = np.round(np.linspace(-1, 1, 11), 2)  # 保留两位小数

for step_idx in range(args.T_horizon):
    step_means, step_stds = [], []
    collision_flag = False
    for ep_means, ep_stds in zip(mean_all, std_all):
        if step_idx < len(ep_means):  # 该episode没碰撞
            step_means.append(ep_means[step_idx])
            step_stds.append(ep_stds[step_idx])
        else:  # 该episode在此之前已经碰撞
            collision_flag = True

    if len(step_means) > 0:
        mu = np.mean(step_means)
        sigma = np.mean(step_stds)
        cdf_vals = norm.cdf(bin_edges, loc=mu, scale=sigma+1e-6)
        probs = np.diff(cdf_vals)
        heatmap_data[:, step_idx] = probs
    else:
        break

# 归一化
with np.errstate(invalid='ignore'):
    max_vals = np.nanmax(heatmap_data, axis=0, keepdims=True)
    max_vals[np.isnan(max_vals)] = 1
    heatmap_data_norm_attack = heatmap_data / (max_vals + 1e-8)

# 绘制
plt.figure(figsize=(10,5))
heatmap_data_norm = heatmap_data / (heatmap_data.max(axis=0, keepdims=True) + 1e-8)

attack_mask = np.isnan(heatmap_data_norm_attack)
# 绘制热力图
ax = sns.heatmap(
    heatmap_data_norm,
    cmap="viridis",
    xticklabels=range(args.T_horizon),
    yticklabels=False,
    linecolor='gray',
    cbar=False,
    mask=attack_mask
)
# cbar_kws={'label': 'Normalized probability'}
# 增加外边框
for spine in ax.spines.values():
    spine.set_visible(True)
    spine.set_linewidth(1.5)  # 边框宽度
    spine.set_color('black')
plt.yticks(ytick_positions + 0.5, ytick_labels, rotation=0)  # 对齐bin中心
plt.xticks(ticks=range(0, args.T_horizon, 3), labels=[str(i) for i in range(0, args.T_horizon, 3)])
plt.xlabel("Step")
plt.ylabel("Action")
# plt.title(f"Action Distribution Heatmap (Mean ± Std): {args.algo}")
plt.tight_layout()
if args.attack:
    plt.savefig(f"heatmap_{args.algo}_attack.png", dpi=300)
else:
    plt.savefig(f"heatmap_{args.algo}.png", dpi=300)


if args.attack:
    # 将所有 episode 均值和std堆叠
    mean_all = np.array(action_dist_records_after_attack[args.algo]['mean'], dtype=object)  # 每个元素是(steps,)
    std_all = np.array(action_dist_records_after_attack[args.algo]['std'], dtype=object)
    print('Mean list', mean_all)
    print('Std list', std_all)

    num_bins = 100
    bin_edges = np.linspace(-1, 1, num_bins + 1)
    heatmap_data = np.full((num_bins, args.T_horizon), np.nan)

    # ---- 控制 y 轴的刻度 ----
    # 固定在 [-1, 1]，一共 11 个刻度
    ytick_positions = np.linspace(0, num_bins - 1, 11)  # 坐标位置
    ytick_labels = np.round(np.linspace(-1, 1, 11), 2)  # 保留两位小数


    for step_idx in range(args.T_horizon):
        step_means, step_stds = [], []
        collision_flag = False
        for ep_means, ep_stds in zip(mean_all, std_all):
            if step_idx < len(ep_means):  # 该episode没碰撞
                step_means.append(ep_means[step_idx])
                step_stds.append(ep_stds[step_idx])
            else:  # 该episode在此之前已经碰撞
                collision_flag = True

        if len(step_means) > 0:
            mu = np.mean(step_means)
            sigma = np.mean(step_stds)
            cdf_vals = norm.cdf(bin_edges, loc=mu, scale=sigma + 1e-6)
            probs = np.diff(cdf_vals)
            heatmap_data[:, step_idx] = probs
        else:
            break

    # 归一化
    with np.errstate(invalid='ignore'):
        max_vals = np.nanmax(heatmap_data, axis=0, keepdims=True)
        max_vals[np.isnan(max_vals)] = 1
        heatmap_data_norm_attack = heatmap_data / (max_vals + 1e-8)

    # 绘制
    plt.figure(figsize=(10, 5))
    heatmap_data_norm_attack = heatmap_data / (heatmap_data.max(axis=0, keepdims=True) + 1e-8)

    # --- 修改点: 创建一个显式的遮罩 ---
    attack_mask = np.isnan(heatmap_data_norm_attack)

    ax = sns.heatmap(
        heatmap_data_norm_attack,
        cmap="viridis",
        xticklabels=range(args.T_horizon),
        yticklabels=ytick_labels,
        cbar=False,
        mask=attack_mask
        # vmin=0, vmax=1 # 可以选择性地固定颜色范围
    )
    # cbar_kws = {'label': 'Normalized probability'}
    # 增加外边框
    for spine in ax.spines.values():
        spine.set_visible(True)
        spine.set_linewidth(1.5)  # 边框宽度
        spine.set_color('black')
    plt.xticks(ticks=range(0, args.T_horizon, 3), labels=[str(i) for i in range(0, args.T_horizon, 3)])
    plt.yticks(ytick_positions + 0.5, ytick_labels, rotation=0)  # 对齐bin中心
    plt.xlabel("Step")
    plt.ylabel("Action")
    # plt.title(f"Action Distribution Heatmap (Mean ± Std): {args.algo}")
    plt.tight_layout()
    plt.savefig(f"heatmap_{args.algo}_after_attack.png", dpi=300)

    # 差分图
    diff = heatmap_data_norm_attack - heatmap_data_norm
    diff_mask = np.isnan(diff)
    print('Mean diff', diff)
    plt.figure(figsize=(10,5))
    ax = sns.heatmap(diff, cmap="bwr", center=0, cbar=False, xticklabels=range(args.T_horizon), yticklabels=ytick_labels, mask=diff_mask)
    # 增加外边框
    for spine in ax.spines.values():
        spine.set_visible(True)
        spine.set_linewidth(1.5)  # 边框宽度
        spine.set_color('black')
    plt.xticks(ticks=range(0, args.T_horizon, 3), labels=[str(i) for i in range(0, args.T_horizon, 3)])
    plt.yticks(ytick_positions + 0.5, ytick_labels, rotation=0)  # 对齐bin中心
    # plt.title("Distribution Difference (Adv - Clean)")
    plt.xlabel("Step")
    plt.ylabel("Action")
    plt.tight_layout()
    plt.savefig(f"heatmap_{args.algo}_diff.png", dpi=300)
    # 从 heatmap 对象中提取 colorbar
    cbar = ax.collections[0].colorbar

    # 新建 figure，只显示横向 colorbar
    # plt.figure(figsize=(6, 1))
    # plt.subplots_adjust(bottom=0.5)
    # cbar_horizontal = plt.colorbar(cbar.mappable, orientation='horizontal')
    # cbar_horizontal.set_label('Normalized probability')
    # plt.savefig('cbar_heatmap.png', dpi=300)
