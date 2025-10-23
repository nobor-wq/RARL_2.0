from stable_baselines3.common.callbacks import CheckpointCallback
import torch as th
import numpy as np
from config import get_config
from gymnasium.wrappers import TimeLimit
import gymnasium as gym
from stable_baselines3.common.monitor import Monitor
import swanlab
import Environment.environment
import random
from defensive_sac import BaseSAC
from stable_baselines3.common.vec_env import DummyVecEnv, SubprocVecEnv
import os
from swanlab.integration.sb3 import SwanLabCallback

def create_model_def(args, env, device):
    model_class = BaseSAC
    model = model_class("MlpPolicy", env, verbose=1, learning_rate=args.lr, batch_size=args.batch_size, device=device)
    return model

def main():
    # get parameters from config.py
    parser = get_config()
    args = parser.parse_args()

    # defenfer and attacker log path
    eval_def_log_path = os.path.join(args.path_def, "base", args.algo, args.env_name, str(args.seed), args.addition_msg)
    os.makedirs(eval_def_log_path, exist_ok=True)
    best_model_path_def = os.path.join(eval_def_log_path, "best_model")
    eval_best_model_path_def = os.path.join(eval_def_log_path, "eval_best_model")

    # 设置设备
    if args.use_cuda and th.cuda.is_available():
        device = th.device(f"cuda:{args.cuda_number}")
    else:
        device = th.device("cpu")

    # 设置随机种子
    random.seed(args.seed)  # 设置 Python 随机种子
    np.random.seed(args.seed)  # 设置 NumPy 随机种子
    th.manual_seed(args.seed)  # 设置 CPU 随机种子
    if th.cuda.is_available():
        th.cuda.manual_seed(args.seed)  # 设置 CUDA 随机种子
        th.cuda.manual_seed_all(args.seed)  # 设置所有 GPU 随机种子
    th.backends.cudnn.deterministic = True  # 确保 CUDA 的确定性
    th.backends.cudnn.benchmark = False  # 禁用 CuDNN 自动优化

    # 2025-10-01 wq 模型保存地址
    model_path_def = os.path.join(eval_def_log_path, 'model')
    os.makedirs(model_path_def, exist_ok=True)
    checkpoint_callback_def = CheckpointCallback(save_freq=args.save_freq, save_path=model_path_def)


    def make_env(seed, rank, attack, defender_first=False):
        def _init():
            env = gym.make(args.env_name, attack=attack, defender_first = defender_first, adv_steps=args.adv_steps)
            env = TimeLimit(env, max_episode_steps=args.T_horizon)
            env = Monitor(env)
            env.unwrapped.start()
            env.reset(seed=seed + rank)
            return env

        return _init

    num_envs = args.num_envs if hasattr(args, 'num_envs') else 1
    if num_envs > 1:
        env_def_first= SubprocVecEnv([make_env(args.seed, i, False, defender_first=True) for i in range(num_envs)])
    else:
        env_def_first = DummyVecEnv([make_env(args.seed, 0, False, defender_first=True)])

    if args.swanlab:
        run_name = f"base-{args.algo}-{args.seed}-{args.addition_msg}"
        run = swanlab.init(project="RARL", name=run_name, config=args)
        swan_cb = SwanLabCallback(project="RARL", experiment_name=run_name, verbose=2)
        # 2025-10-02 wq 初始化训练
        model_def_first = create_model_def(args, env_def_first, device)
        model_def_first.learn(total_timesteps=args.train_step * args.n_steps, progress_bar=True,
                              callback=[checkpoint_callback_def, swan_cb])
        env_def_first.close()
    else:
        model_def_first = create_model_def(args, env_def_first, device)
        model_def_first.learn(total_timesteps=args.train_step * args.n_steps, progress_bar=True,
                              callback=checkpoint_callback_def)
        env_def_first.close()
    model_def_first.save(os.path.join(eval_def_log_path, "lunar"))
    del model_def_first

if __name__ == '__main__':
    main()
