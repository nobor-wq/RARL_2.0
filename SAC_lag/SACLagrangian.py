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
parser.add_argument('--seed', default=0)
parser.add_argument('--train_step', default=500)
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

scores_final = []
cvs_final = []

class SafetySampleBuffer(SampleBuffer):
    # COMPONENT_NAMES = (*SampleBuffer.COMPONENT_NAMES, 'violations')
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        # self._create_buffer('violations', torch.bool, [])


def create_buffer(capacity):
    buffer = SafetySampleBuffer(args.state_dim, args.action_dim, capacity)
    buffer.to(device)
    return DummyModuleWrapper(buffer)

replay_buffer = create_buffer(args.buffer_max)

if args.wandb:
    run_name = f"{args.env_name}-{args.algo_name}-{args.addition_msg}"
    run = wandb.init(project="run_result", name=run_name, config=args)


score, sn, mean_step = 0.0, 0.0, 0.0

for i in range(args.train_step):

    evaluation_episode = ( (i+1) % args.print_interval == 0)
    print(f'\rEpisode: {i + 1}/{args.train_step}', end='')

    state, _ = env.reset()
    done = False
    ep_step = 0

    while not done and ep_step < args.T_horizon:
        if replay_buffer.__len__() > args.min_data:
            agent.update_parameters(replay_buffer, args.batch_size)
        ep_step += 1
        # 选择动作
        action = agent.select_action(state)  # 改为 select_action
        next_state, reward, done, _, info = env.step(action)

        for buffer in [replay_buffer]:
            buffer.append(states=torch.tensor(state), actions=action, next_states=torch.tensor(next_state),
                          rewards=reward, costs=info['cost'], dones=done)

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
    mean_step += ep_step

    if (i+1) % 100 == 0:
        ckpt_path = os.path.join(save_dir, f"sac_lag_{i+1}_{score}.pt")
        agent.save_actor(ckpt_path)
        print(f"    Model saved to {ckpt_path}")

    # 评估结束，记录并可能保存模型
    if args.wandb:
        if evaluation_episode:
            wandb.log(
                {
                    "sn": sn/args.print_interval,
                    "rew_mean": score/args.print_interval,
                    "len_mean": mean_step/args.print_interval
                }
            )
            score, sn, mean_step = 0.0, 0.0, 0.0

env.close()
