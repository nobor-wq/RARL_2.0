import os
import numpy as np
import torch
import gymnasium as gym
import time
import matplotlib.pyplot as plt
from SAC_Agent_Continuous import SAC_Lag
from sampling import SampleBuffer
from torch_util import DummyModuleWrapper
import Environment.environment
import datetime
import argparse
import random
from gymnasium.wrappers import TimeLimit
from stable_baselines3.common.monitor import Monitor

# os.environ["WANDB_DISABLE_SSL_VERIFY"] = "true"
import wandb



parser = argparse.ArgumentParser()
parser.add_argument('--env_name', default="TrafficEnv3-v1", help='name of the environment to run')
parser.add_argument('--algo_name', default="SAC_lag", help='name of the alg')
parser.add_argument('--seed', default=5)
parser.add_argument('--episodes', default=100)
parser.add_argument('--T_horizon', default=30)
parser.add_argument('--print_interval', default=10)
parser.add_argument('--speed_range', default=15.0, help='Maximum speed')
parser.add_argument('--max_a', default=7.6, help='Maximum Acceleration')
parser.add_argument('--state_dim', default=26)
parser.add_argument('--action_dim', default=1)
parser.add_argument('--buffer_max',  type=int, default=10**6, help='size of buffer')
parser.add_argument('--batch_size',  type=int, default=128, help='each update needs data')
parser.add_argument('--min_data',  type=int, default=256, help='min data to update')
parser.add_argument('--addition_msg', default="", help='additional message of the training process')
parser.add_argument('--wandb', type=bool, default=True, help='whether to use wandb logging')
args = parser.parse_args()



random.seed(args.seed)  # 设置 Python 随机种子
np.random.seed(args.seed)  # 设置 NumPy 随机种子
torch.manual_seed(args.seed)  # 设置 CPU 随机种子
if torch.cuda.is_available():
    torch.cuda.manual_seed(args.seed)  # 设置 CUDA 随机种子
    torch.cuda.manual_seed_all(args.seed)  # 设置所有 GPU 随机种子
torch.backends.cudnn.deterministic = True  # 确保 CUDA 的确定性
torch.backends.cudnn.benchmark = False  # 禁用 CuDNN 自动优化

# 保存目录
current_date = datetime.datetime.now().strftime("%Y-%m-%d_%H%M")
save_dir = os.path.join("logs", "age_eval", args.env_name, args.algo_name, current_date)
os.makedirs(save_dir, exist_ok=True)

device = torch.device("cpu")
# device = torch.device("cuda:1" if th.cuda.is_available() else "cpu")

env = gym.make(args.env_name)
env = TimeLimit(env, max_episode_steps=args.T_horizon)
env = Monitor(env)
env.unwrapped.start(gui=False)

agent = SAC_Lag(state_dim=args.state_dim, action_dim=args.action_dim, device=device)
model_path = "sac_lag_5000_271.pt"
agent.load_actor(model_path)
agent.policy.eval()

scores_final = []
cvs_final = []

score, sn  = 0.0, 0.0

for i in range(args.episodes):
    state, _ = env.reset(options="seed")
    done = False
    ep_step = 0
    while not done and ep_step < args.T_horizon:
        ep_step += 1
        # 选择动作
        action = agent.select_action(state, deterministic=True)  # 改为 select_action
        print("select action is: ", action)
        next_state, reward, done, _, info = env.step(action)

        state = next_state
        score += reward
        xa = info['x_position']
        ya = info['y_position']

    if done is False:
        if args.env_name == 'TrafficEnv1-v0' or args.env_name == 'TrafficEnv3-v0' or args.env_name == 'TrafficEnv3-v1':
            if xa < -50.0 and ya > 4.0 :
                sn += 1
        elif args.env_name == 'TrafficEnv2-v0':
            if xa > 50.0 and ya > -5.0 :
                sn += 1
        elif args.env_name == 'TrafficEnv4-v0':
            if ya < -50.0 :
                sn += 1
env.close()

sn_rate = sn / args.episodes
score_mean = score / args.episodes

print("成功率：", sn_rate)
print("平均奖励：", score_mean)

log_file = "eval_log.txt"

# 将参数和结果写入日志文件
with open(log_file, 'a') as f:  # 使用 'a' 模式以追加方式写入文件
    # 写入参数
    f.write("Parameters:\n")
    for arg in vars(args):  # 遍历 args 中的所有参数
        f.write(f"{arg}: {getattr(args, arg)}\n")

    # 写入结果
    f.write("\nResults:\n")
    f.write(f"Mean reward: {score_mean:.2f} \n")

    f.write(f"Success rate: {sn_rate:.2f}\n")


