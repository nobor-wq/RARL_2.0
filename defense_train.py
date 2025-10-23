from stable_baselines3.common.callbacks import BaseCallback, CheckpointCallback
import torch as th
import numpy as np
from config import get_config
from gymnasium.wrappers import TimeLimit
import gymnasium as gym
from stable_baselines3.common.monitor import Monitor
import Environment.environment
import matplotlib.pyplot as plt
import pandas as pd
import wandb
from wandb.integration.sb3 import WandbCallback
from callback import CustomEvalCallback
import random
from buffer import PaddingRolloutBuffer, DecoupleRolloutBuffer, DecouplePaddingRolloutBuffer
from defense_ppo import DefensePPO
from stable_baselines3.common.buffers import RolloutBuffer
from stable_baselines3.common.vec_env import DummyVecEnv, SubprocVecEnv
from stable_baselines3 import PPO
import os

class MySubprocVecEnv(SubprocVecEnv):
    def get_remain_attack_times(self):
        remain_attacks = self.env_method('get_remain_attack_times')  # 用call批量调用每个子环境的方法
        return np.array(remain_attacks)


class MyDummyVecEnv(DummyVecEnv):
    def get_remain_attack_times(self):
        remain_attacks = self.env_method('get_remain_attack_times')  # 用call批量调用每个子环境的方法
        return np.array(remain_attacks)


def create_model(args, env, rollout_buffer_class, device, adv_model_path, best_model_path, last_agent_path=None,
                 run=None, base_model_path=None):
    """
    根据 args 配置来创建 基于PPO的敌手模型。
    如果需要 wandb，返回模型时会附带 wandb 配置信息。
    :param args: 系统参数
    :param env: 环境名
    :param rollout_buffer_class: 回放池类型
    :param device: 模型加载设备名
    :param best_model_path: 最优模型存储路径
    :param run: 是否使用wandb进行日志记录
    """
    model_class = DefensePPO
    if run:
        model = model_class(args, adv_model_path, best_model_path, last_agent_path, "MlpPolicy",
                            env, n_steps=args.n_steps, batch_size=args.batch_size, verbose=1,
                            tensorboard_log=f"runs/{run.id}",
                            rollout_buffer_class=rollout_buffer_class, device=device, ent_coef=args.ent_coef)
    else:
        model = model_class(args, adv_model_path, best_model_path, last_agent_path, "MlpPolicy",
                            env, n_steps=args.n_steps, batch_size=args.batch_size, verbose=1,
                            rollout_buffer_class=rollout_buffer_class,
                            device=device, ent_coef=args.ent_coef)
    if args.use_base_model:
        old_model = PPO.load(base_model_path)
        model.policy.load_state_dict(old_model.policy.state_dict())
    return model


def main():
    # get parameters from config.py
    parser = get_config()
    args = parser.parse_args()

    # log path
    eval_log_path = "./logs/defense/" + os.path.join(args.adv_algo, args.env_name, args.algo, args.addition_msg)
    os.makedirs(eval_log_path, exist_ok=True)

    best_model_path = os.path.join(eval_log_path, "best_model")
    eval_best_model_path = os.path.join(eval_log_path, "eval_best_model")

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

    # model_path = os.path.join(eval_log_path, 'model')
    # os.makedirs(model_path, exist_ok=True)
    # checkpoint_callback = CheckpointCallback(save_freq=args.save_freq, save_path=model_path)

    # whether padding
    # rollout_buffer_map = {
    #     (True, True): DecouplePaddingRolloutBuffer,
    #     (True, False): PaddingRolloutBuffer,
    #     (False, True): DecoupleRolloutBuffer,
    #     (False, False): RolloutBuffer
    # }
    # rollout_buffer_class = rollout_buffer_map[(args.padding, args.decouple)]
    rollout_buffer_class = RolloutBuffer

    def make_env(seed, rank):
        def _init():
            env = gym.make(args.env_name, attack=args.attack, adv_steps=args.adv_steps, defense=True)
            env = TimeLimit(env, max_episode_steps=args.T_horizon)
            env = Monitor(env)
            env.unwrapped.start()
            env.reset(seed=seed + rank)
            return env

        return _init

    num_envs = args.num_envs if hasattr(args, 'num_envs') else 1
    if num_envs > 1:
        env = MySubprocVecEnv([make_env(args.seed, i) for i in range(num_envs)])
        # eval_env = SubprocVecEnv([make_env(args.seed + 1000, i) for i in range(num_envs)])
    else:
        env = MyDummyVecEnv([make_env(args.seed, 0)])
    eval_env = MyDummyVecEnv([make_env(args.seed + 1000, 0)])

    # 设置模型路径
    if args.last_agent:
        last_agent_path = os.path.join(args.path, args.env_name, args.algo, 'best_model/best_model')
    else:
        last_agent_path = None
    adv_model_path = os.path.join('./logs/adv_eval', args.adv_algo, args.env_name, args.algo, args.adv_model_path, "best_model")

    best_model_path = os.path.join('./logs/defense', args.adv_algo, args.env_name, args.algo, args.addition_msg, "best_model")
    if args.use_base_model:
        base_model_path = os.path.join(args.path, args.env_name, args.algo, 'best_model/best_model')
    else:
        base_model_path = None
    os.makedirs(best_model_path, exist_ok=True)

    # 初始化wandb并构建敌手模型
    if not args.no_wandb:
        run_name = f"{args.attack_method}-{args.algo}-{args.addition_msg}"
        run = wandb.init(project="RARL_Defense", name=run_name, config=args, sync_tensorboard=True)
        model = create_model(args, env, rollout_buffer_class, device, adv_model_path, best_model_path, last_agent_path,
                             run, base_model_path)
        wandb_callback = WandbCallback(gradient_save_freq=500, verbose=2)
        # eval_callback = CustomEvalCallback(eval_env, trained_agent=model.trained_agent,
        #                                best_model_save_path=eval_best_model_path, n_eval_episodes=20, eval_freq=args.n_steps,
        #                                unlimited_attack=args.unlimited_attack, attack_method=args.attack_method)
        model.learn(total_timesteps=args.train_step * args.n_steps, progress_bar=True,
                    callback=[wandb_callback])
    else:
        model = create_model(args, env, rollout_buffer_class, device, adv_model_path, best_model_path, last_agent_path,
                             base_model_path=base_model_path)
        # eval_callback = CustomEvalCallback(eval_env, trained_agent=model.trained_agent,
        #                                best_model_save_path=eval_best_model_path, n_eval_episodes=20, eval_freq=args.n_steps,
        #                                unlimited_attack=args.unlimited_attack, attack_method=args.attack_method)
        model.learn(total_timesteps=args.train_step * args.n_steps, progress_bar=True,
                    callback=[])

    # 读取评估日志文件
    # eval_log_file = os.path.join(eval_log_path, "evaluations.npz")

    # 绘制评估奖励曲线
    # plt.plot(eval_env.get_episode_rewards())
    # plt.title('Rewards per episode')
    # plt.xlabel('Episode')
    # plt.ylabel('Reward')
    # plt.savefig(os.path.join(eval_log_path, "rewards.png"), dpi=300)
    # plt.close()
    #
    # plt.plot(eval_env.get_episode_lengths())
    # plt.title('Steps per episode')
    # plt.xlabel('Episode')
    # plt.ylabel('Steps')
    # plt.savefig(os.path.join(eval_log_path, "steps.png"), dpi=300)
    # plt.close()
    #
    # reward_df = pd.DataFrame(eval_env.get_episode_rewards())
    # step_df = pd.DataFrame(eval_env.get_episode_lengths())
    # reward_df.to_csv(os.path.join(eval_log_path, "rewards.csv"), index=False)
    # step_df.to_csv(os.path.join(eval_log_path, "steps.csv"), index=False)
    # Save the log

    # Save the agent
    model.save(os.path.join(eval_log_path, "lunar"))
    del model  # delete trained model to demonstrate loading
    env.close()


if __name__ == '__main__':
    main()
