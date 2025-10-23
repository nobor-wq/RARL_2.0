# 对所有算法绘制0.05扰动下的动作偏移
import gymnasium as gym
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import torch as th
import os
from stable_baselines3 import SAC, PPO, TD3
from gymnasium.wrappers import TimeLimit
from stable_baselines3.common.monitor import Monitor
from tqdm import tqdm
from DARRLNetworkParams import ActorNet, ActorNet_adv, SAC_lag_Net, FniNet
from stable_baselines3 import SAC, PPO, TD3  # 或者您使用的其他算法

# =============================================================================
# 2. 辅助函数
# =============================================================================
def get_action_from_my_policy(state, algo, agent, device):
    """
    Gets the deterministic action from a trained agent for a given state.

    Args:
        state (np.ndarray): The input state (observation).
        algo (str): The name of the algorithm (e.g., 'PPO', 'SAC', 'FNI').
        agent: The trained agent/policy object.
        device (torch.device): The device to run inference on (e.g., 'cuda:0' or 'cpu').

    Returns:
        np.ndarray: The resulting action.
    """
    # Ensure inference is done without calculating gradients
    with th.no_grad():
        # Convert numpy state to a torch tensor with a batch dimension
        obs_tensor = th.as_tensor(state, dtype=th.float32, device=device).unsqueeze(0)

        action = None
        if algo in ('FNI', 'DARRL', 'IGCARL'):
            # These models return mu, std, _
            mu, _, _ = agent(obs_tensor)
            action = mu
            action = action.cpu().numpy().flatten()
        elif algo == 'SAC_Lag':
            # This model returns mu, log_std
            _, _, action = trained_agent.sample(obs_tensor)
            action = action.cpu().numpy().flatten()
        elif algo in ('PPO', 'SAC', 'TD3'):
            action, _ = trained_agent.predict(state, deterministic=True)
            action = action.flatten()
        if action is None:
            raise NotImplementedError(f"Algorithm '{algo}' is not supported in this function.")

        # Move action to CPU and convert to numpy array
        return action


def get_action_gradient(state, algo, agent, device):
    """
    计算动作关于输入状态的梯度
    """
    obs_tensor = th.as_tensor(state, dtype=th.float32, device=device).unsqueeze(0)
    obs_tensor.requires_grad = True

    action_tensor = None
    if algo in ('PPO', 'SAC', 'TD3'):
        # 对于SB3模型，我们需要直接调用其actor网络以获取可微分的张量
        if algo == 'PPO':
            dist = agent.policy.get_distribution(obs_tensor)
            action_tensor = dist.distribution.mean
        else:  # SAC, TD3
            action_tensor = agent.actor(obs_tensor)
            if isinstance(action_tensor, tuple):
                action_tensor = action_tensor[0]
    elif algo in ('FNI', 'DARRL', 'IGCARL'):
        action_tensor, _, _ = agent(obs_tensor)
    elif algo == 'SAC_Lag':
        action_tensor, _, _ = agent.sample(obs_tensor)

    if action_tensor is None:
        raise NotImplementedError(f"Gradient calculation for '{algo}' is not supported.")

    action_tensor.sum().backward()
    grad = obs_tensor.grad.squeeze().cpu().numpy()
    return grad


# =============================================================================
# 3. 主分析流程
# =============================================================================

# --- 分析参数 ---
STATE_DIM = 26
RESOLUTION = 100
DEVICE = th.device("cuda:0" if th.cuda.is_available() else "cpu")
ENV_NAME = "TrafficEnv3-v1"
BASE_MODEL_PATH = "./models/"

# 要比较的算法列表
EPSILONS = [0.1]
ALGOS_TO_COMPARE = ['PPO', 'SAC', 'TD3', 'SAC_Lag', 'FNI', 'DARRL', 'IGCARL']


# 获取一个初始状态作为分析中心
# s_0_list = [
# [0.09909227,0.0,0.46403053,0.75,0.27803245,0.20093006,0.79522604,0.5,1.0,0.0,0.0,0.0,0.10142445,-0.4215388,0.5704252,
#  0.25,0.04866946,-0.22357178,0.37614575,0.25,0.039262325,0.5,0.66966283,0.75,1.0,0.75]
# ]
s_0_list_old = [
[0.7977517,0.1577962,0.93034124,0.75,0.21962911,0.27327174,0.77657455,0.5,0.7572472,0.3474857,0.6482767,0.25,
 1.0,0.0,0.0,0.0,1.0,0.0,0.0,0.0,1.0,0.0,0.0,0.0,1.0,0.0],
[0.6970827,0.16172528,0.92838246,0.75,0.08866517,0.30876663,0.7946154,0.5,0.6678262,0.348244,0.6508252,0.25,
 1.0,0.0,0.0,0.0,1.0,0.0,0.0,0.0,1.0,0.0,0.0,0.0,1.0,0.0],
# [0.60052097,0.16535841,0.8345101,0.75,0.07479899,0.32035795,0.7935997,0.5,0.5789093,0.3494217,0.63977635,0.25,
#  0.05985176,-0.33977908,0.7769097,0.5,1.0,0.0,0.0,0.0,1.0,0.0,0.0,0.0,1.0,0.0],
[0.09816216,0.10622387,0.28014633,0.75,0.062797114,0.29066005,0.47732234,0.75,0.031482898,0.43358952,0.4282253,
 0.25,1.0,0.0,0.0,0.0,0.101141006,-0.29512632,0.7742487,0.5,1.0,0.0,0.0,0.0,1.0,0.98121536],
[0.07024323,0.060844433,0.34176302,0.75,0.10997936,0.2711736,0.77353585,0.5,0.05014201,0.41248032,0.54115963,0.75,
 0.073639,-0.4521771,0.2655102,0.25,0.035101138,-0.3182154,0.7925001,0.5,0.032950073,-0.115034215,0.5304923,0.25,0.50105566,0.92676276],
# [0.040507667,0.11337618,0.7817835,0.5,0.18617517,0.22367246,0.7745424,0.5,0.042812925,0.49141803,0.60328126,0.75,
#  0.086867556,-0.41184512,0.3975589,0.25,0.045703318,-0.24654646,0.34781283,0.25,0.12427995,-0.0599202,0.60824966,0.25,0.70840377,0.7909674],
[0.1042355,0.15856001,0.615245,0.25,0.1364679,0.23103333,0.4116078,0.75,0.047947843,0.36629468,0.7996352,0.5,
1.0,0.0,0.0,0.0,0.12228283,-0.29213965,0.77727324,0.5,1.0,0.0,0.0,0.0,1.0,0.0],
# [0.07024323,0.060844433,0.34176302,0.75,0.10997936,0.2711736,0.77353585,0.5,0.05014201,0.41248032,0.54115963,0.75,
# 0.073639,-0.4521771,0.2655102,0.25,0.035101138,-0.3182154,0.7925001,0.5,0.032950073,-0.115034215,0.5304923,0.25,0.50105566,0.92676276]
]
# s_0 = [0.040507667,0.11337618,0.7817835,0.5,0.18617517,0.22367246,0.7745424,0.5,0.042812925,0.49141803,0.60328126,0.75, 0.086867556,-0.41184512,0.3975589,0.25,0.045703318,-0.24654646,0.34781283,0.25,0.12427995,-0.0599202,0.60824966,0.25,0.70840377,0.7909674]
# s_0 = [0.1042355,0.15856001,0.615245,0.25,0.1364679,0.23103333,0.4116078,0.75,0.047947843,0.36629468,0.7996352,0.5,
# 1.0,0.0,0.0,0.0,0.12228283,-0.29213965,0.77727324,0.5,1.0,0.0,0.0,0.0,1.0,0.0]
# s_0 = [0.07024323,0.060844433,0.34176302,0.75,0.10997936,0.2711736,0.77353585,0.5,0.05014201,0.41248032,0.54115963,0.75,
# 0.073639,-0.4521771,0.2655102,0.25,0.035101138,-0.3182154,0.7925001,0.5,0.032950073,-0.115034215,0.5304923,0.25,0.50105566,0.92676276]

# s_0_list = [
# [0.08654119,0.016529402,0.34176302,0.75,0.091991864,0.2390672,0.77353585,0.5,0.0236178,0.4379837,0.54115963,0.75,
#  0.06298592,-0.39364055,0.2655102,0.25,0.049562525,-0.22966859,0.7925001,0.5,0.060031988,-0.112641454,0.5304923,0.25,0.8689022,0.8533303],
# [0.09049825,0.010381286,0.34176302,0.75,0.08932654,0.23116758,0.77353585,0.5,0.018577801,0.44857523,0.54115963,0.75,
#  0.061774652,-0.3806577,0.2655102,0.25,0.053285517,-0.21829413,0.7925001,0.5,0.06524592,-0.111627474,0.5304923,0.25,0.93672794,0.84997636],
# [0.09909227,0.0,0.46403053,0.75,0.27803245,0.20093006,0.79522604,0.5,1.0,0.0,0.0,0.0,0.10142445,-0.4215388,
#  0.5704252,0.25,0.04866946,-0.22357178,0.37614575,0.25,0.039262325,0.5,0.66966283,0.75,1.0,0.75],
# [0.594429,0.16439703,0.8345101,0.75,0.06846599,0.32740197,0.7935997,0.5,0.5731807,0.35057157,0.63977635,
#  0.25,1.0,0.0,0.0,0.0,0.06594423,-0.33063725,0.7769097,0.5,1.0,0.0,0.0,0.0,1.0,0.0],
# [0.89304006,0.15401646,0.9280561,0.75,0.3449353,0.2647862,0.7827482,0.5,0.8407728,0.3476244,0.6384082,
#  0.25,1.0,0.0,0.0,0.0,1.0,0.0,0.0,0.0,1.0,0.0,0.0,0.0,1.0,0.0],
# ]

s_0_list = [
[1.0,0.0,0.0,0.0,0.84218675,0.25604877,0.7884088,0.5,1.0,0.0,0.0,0.0,1.0,0.0,0.0,0.0,1.0,0.0,0.0,0.0,1.0,0.0,0.0,0.0,0.0,0.0],
[0.9943977,0.15164976,0.9323587,0.75,0.4782257,0.26065767,0.79861337,0.5,0.9296712,0.34693593,0.6366861,0.25,1.0,0.0,0.0,0.0,1.0,0.0,0.0,0.0,1.0,0.0,0.0,0.0,1.0,0.0],
[0.69107354,0.16086648,0.92838246,0.75,0.08210146,0.3137213,0.7946154,0.5,0.66206485,0.34922925,0.6508252,0.25,1.0,0.0,0.0,0.0,1.0,0.0,0.0,0.0,1.0,0.0,0.0,0.0,1.0,0.0],
[0.61510015,0.1252045,0.9286797,0.75,0.10761138,0.29805413,0.78600657,0.5,0.48318028,0.35189053,0.6658786,0.25,1.0,0.0,0.0,0.0,0.079305775,-0.3161035,0.7745433,0.5,1.0,0.0,0.0,0.0,1.0,0.0],
[0.22282372,0.1300715,0.6376857,0.25,0.16283907,0.2397277,0.52179575,0.25,0.035458326,0.42912328,0.78937477,0.5,1.0,0.0,0.0,0.0,0.12820108,-0.29015085,0.7996682,0.5,1.0,0.0,0.0,0.0,1.0,0.0],
[0.07024323,0.060844433,0.34176302,0.75,0.10997936,0.2711736,0.77353585,0.5,0.05014201,0.41248032,0.54115963,0.75,0.073639,-0.4521771,0.2655102,0.25,0.035101138,-0.3182154,0.7925001,0.5,0.032950073,-0.115034215,0.5304923,0.25,0.50105566,0.92676276],
[0.09385265,0.0,0.5289496,0.75,0.23851956,0.16701941,0.77459085,0.5,1.0,0.0,0.0,0.0,0.1593269,-0.45129517,0.6583022,0.25,0.05048467,-0.30014032,0.52363515,0.25,0.059703574,0.5,0.731638,0.75,0.33593234,0.75],
[0.08303008,0.0,0.28938824,0.75,0.3728997,0.1692704,0.7774285,0.5,1.0,0.0,0.0,0.0,0.0952156,-0.41590863,0.66253346,0.25,0.45049047,-0.18414032,0.7865022,0.5,0.06247909,-0.13943785,0.51415986,0.25,0.52589774,0.75],
[0.1599409,0.0,0.48187014,0.75,0.8106836,0.1785545,0.77347684,0.5,1.0,0.0,0.0,0.0,0.07129907,-0.38245523,0.6534352,0.25,0.8248173,-0.17986563,0.7846309,0.5,0.10176676,-0.078173734,0.65420383,0.25,0.8847849,0.75],
[0.24818082,0.0,0.67077035,0.75,1.0,0.0,0.0,0.0,1.0,0.0,0.0,0.0,0.14250404,-0.4453222,0.652128,0.25,1.0,0.0,0.0,0.0,0.058242872,-0.15416919,0.64918244,0.25,1.0,0.75],
]

all_results = []
sns.set(style="ticks", font_scale=1.2)
# --- 创建 3x7 子图网格 ---
# fig, axes = plt.subplots(len(EPSILONS), len(ALGOS_TO_COMPARE), figsize=(28, 12))

# 用于存储最终平均Z矩阵的字典
averaged_z_matrices = {}

# --- 主循环：遍历所有算法 ---
for algo_name in ALGOS_TO_COMPARE:
    for epsilon in EPSILONS:
        print(f"\n--- Processing combination: Algorithm={algo_name}, Epsilon={epsilon} ---")

        # --- 加载模型 (每个模型只加载一次) ---
        trained_agent = None
        try:
            if algo_name == "IGCARL":
                model_path_drl = os.path.join(BASE_MODEL_PATH, ENV_NAME, '2000', 'drl', '0.05', 'm4', '5',
                                              'defender/defender.pth')
                trained_agent = ActorNet(STATE_DIM, 1).to(DEVICE)
                trained_agent.load_state_dict(th.load(model_path_drl, map_location=DEVICE))
            # ... 其他模型的加载逻辑 ...
            elif algo_name == 'PPO':
                model_path = os.path.join(BASE_MODEL_PATH, ENV_NAME, '2000', algo_name, 'defender',
                                          'lunar_baseline')
                trained_agent = PPO.load(model_path, device=DEVICE)
            elif algo_name == 'SAC':
                model_path = os.path.join(BASE_MODEL_PATH, ENV_NAME, '2000', algo_name, 'defender',
                                          'lunar_baseline')
                trained_agent = SAC.load(model_path, device=DEVICE)
            elif algo_name == 'TD3':
                model_path = os.path.join(BASE_MODEL_PATH, ENV_NAME, '2000', algo_name, 'defender',
                                          'lunar_baseline')
                trained_agent = TD3.load(model_path, device=DEVICE)
            elif algo_name == 'SAC_Lag':
                model_path = os.path.join(BASE_MODEL_PATH, ENV_NAME, '2000', 'SAC_lag', 'defender',
                                          'lunar_baseline.pt')
                trained_agent = SAC_lag_Net(STATE_DIM, 1).to(DEVICE)
                trained_agent.load_state_dict(th.load(model_path, map_location=DEVICE))
            elif algo_name == 'DARRL':
                model_path = os.path.join(BASE_MODEL_PATH, ENV_NAME, '2000', algo_name, 'defender',
                                          'policy2000_actor.pth')
                trained_agent = FniNet(STATE_DIM, 1).to(DEVICE)
                trained_agent.load_state_dict(th.load(model_path, map_location=DEVICE))
            elif algo_name == "FNI":
                model_path = os.path.join(BASE_MODEL_PATH, ENV_NAME, '2000', algo_name, 'defender',
                                          'policy_v411.pth')
                trained_agent = FniNet(STATE_DIM, 1).to(DEVICE)
                trained_agent.load_state_dict(th.load(model_path, map_location=DEVICE))

            if hasattr(trained_agent, 'eval'): trained_agent.eval()

            # --- 遍历 s_0 列表，生成多个Z矩阵 ---
            z_matrices_for_s0_list = []
            for s_0_raw in tqdm(s_0_list, desc=f"Processing s_0 for {algo_name} ε={epsilon}"):
                s_0 = np.array(s_0_raw, dtype=np.float32)
                gradient_vector = get_action_gradient(s_0, algo_name, trained_agent, DEVICE)
                v1 = gradient_vector / (np.linalg.norm(gradient_vector) + 1e-8)
                v2_raw = np.random.randn(STATE_DIM);
                v2_ortho = v2_raw - np.dot(v2_raw, v1) * v1
                v2 = v2_ortho / (np.linalg.norm(v2_ortho) + 1e-8)
                c1_coords = np.linspace(-epsilon, epsilon, RESOLUTION);
                c2_coords = np.linspace(-epsilon, epsilon, RESOLUTION)
                C1, C2 = np.meshgrid(c1_coords, c2_coords)
                Z = np.zeros_like(C1)
                action_base = get_action_from_my_policy(s_0, algo_name, trained_agent, DEVICE)[0]
                for i in range(RESOLUTION):
                    for j in range(RESOLUTION):
                        delta = C1[i, j] * v1 + C2[i, j] * v2
                        s_perturbed = s_0 + delta
                        action_perturbed = get_action_from_my_policy(s_perturbed, algo_name, trained_agent, DEVICE)[0]
                        Z[i, j] = action_perturbed - action_base
                z_matrices_for_s0_list.append(Z)

            # --- 计算平均Z矩阵 ---
            if z_matrices_for_s0_list:
                z_avg = np.mean(z_matrices_for_s0_list, axis=0)
                averaged_z_matrices[(algo_name, epsilon)] = z_avg

        except Exception as e:
            print(f"Error processing {algo_name} with epsilon={epsilon}: {e}")
            averaged_z_matrices[(algo_name, epsilon)] = np.zeros((RESOLUTION, RESOLUTION))

# 设置共享的行和列标题
fig, axes = plt.subplots(len(EPSILONS), len(ALGOS_TO_COMPARE), figsize=(26,5))
mappable = None

print("\n--- Generating final plot ---")
for i, epsilon in enumerate(EPSILONS):
    for j, algo_name in enumerate(ALGOS_TO_COMPARE):
        ax = axes[j]
        z_to_plot = averaged_z_matrices.get((algo_name, epsilon), np.zeros((RESOLUTION, RESOLUTION)))

        sns.heatmap(
            np.transpose(z_to_plot), ax=ax, cmap='coolwarm', center=0, vmin=-0.3, vmax=0.3, cbar=False
        )
        ax.set_aspect('equal', adjustable='box')
        # --- 设置每个子图的刻度 ---

        tick_positions = np.linspace(0, RESOLUTION, num=3)

        tick_labels_float = np.linspace(-epsilon, epsilon, num=3)

        ax.set_xticks(tick_positions, labels=[f'{val:.2f}' for val in tick_labels_float], rotation='horizontal')

        ax.set_yticks(tick_positions, labels=[f'{val:.2f}' for val in tick_labels_float], rotation='horizontal')
        if (i, j) == (0, 0):  # 只需获取一次mappable
            mappable = ax.collections[0]

# --- 调整和美化整个图表 ---
for i, epsilon in enumerate(EPSILONS):
    axes[0].set_ylabel(f'Action Offset from Clean Observation')
for j, algo_name in enumerate(ALGOS_TO_COMPARE):
    axes[j].set_xlabel(algo_name, labelpad=15)

# 隐藏内部子图多余的标签
for j in range(len(ALGOS_TO_COMPARE)):
    if j > 0:
        axes[j].set_ylabel('')

cbar_ax = fig.add_axes([0.3, 0.9, 0.4, 0.02])
if mappable:
    cbar = fig.colorbar(mappable, cax=cbar_ax, orientation='horizontal')
cbar_ax.xaxis.set_ticks_position('top')
cbar_ax.xaxis.set_label_position('top')

fig.subplots_adjust(
    left=0.05, right=0.98, top=0.92, bottom=0.06, wspace=0.17, hspace=0.17
)
plt.savefig('heatmap_grid_comparison_averaged.pdf', bbox_inches='tight')