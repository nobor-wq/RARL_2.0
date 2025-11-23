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
from stable_baselines3 import PPO, SAC, TD3
from stable_baselines3.common.vec_env import DummyVecEnv, SubprocVecEnv
import os
from swanlab.integration.sb3 import SwanLabCallback
from SAC_lag.SAC_Agent_Continuous import SAC_Lag
from SAC_lag.sampling import SampleBuffer
from SAC_lag.torch_util import DummyModuleWrapper



def main():
    # get parameters from config.py
    parser = get_config()
    args = parser.parse_args()

    # defenfer and attacker log path
    eval_def_log_path = os.path.join("logs", args.env_name, args.algo, str(args.seed), args.addition_msg)
    os.makedirs(eval_def_log_path, exist_ok=True)

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


    def make_env(seed, rank, attack):
        def _init():
            env = gym.make(args.env_name, attack=attack, adv_steps=args.adv_steps)
            env = TimeLimit(env, max_episode_steps=args.T_horizon)
            env = Monitor(env)
            env.unwrapped.start()
            env.reset(seed=seed + rank)
            return env

        return _init

    callbacks_common = []

    if args.swanlab:
        run_name = f"base-{args.algo}-{args.seed}-{args.addition_msg}"
        run = swanlab.init(project="RARL", name=run_name, config=args)
        swan_cb = SwanLabCallback(project="RARL", experiment_name=run_name, verbose=2, log_interval=100)
        callbacks_common.append(swan_cb)

    # ====== 基于 SB3 的算法（PPO / SAC / TD3）训练 ======
    if args.algo in ['PPO', 'SAC', 'TD3']:
        vec_env = DummyVecEnv([make_env(args.seed, 0, False)])

        if args.algo == 'PPO':
            model = PPO(
                "MlpPolicy",
                vec_env,
                verbose=1,
                learning_rate=args.lr,
                n_epochs=args.n_epochs,
                batch_size=args.batch_size,
                device=device,
                n_steps=args.n_steps,
            )
        elif args.algo == 'SAC':
            model = SAC(
                "MlpPolicy",
                vec_env,
                verbose=1,
                learning_rate=args.lr,
                batch_size=args.batch_size,
                device=device,
            )
        else:  # TD3
            model = TD3(
                "MlpPolicy",
                vec_env,
                verbose=1,
                learning_rate=args.lr,
                batch_size=args.batch_size,
                device=device,
            )

        checkpoint_callback_def = CheckpointCallback(save_freq=args.save_freq, save_path=eval_def_log_path)
        model.learn(
            total_timesteps=args.train_step * args.n_steps,
            progress_bar=True,
            callback=[checkpoint_callback_def] + callbacks_common,
        )
        model.save(os.path.join(eval_def_log_path, "lunar"))
        del model
        vec_env.close()

    # ====== 自定义 SAC_Lag 算法训练（无约束地保持原脚本逻辑） ======
    elif args.algo == 'SAC_lag':
        # 单环境（不使用 VecEnv），尽量复用原 SACLagrangian 逻辑
        env = make_env(args.seed, 0, False)()

        agent = SAC_Lag(state_dim=args.state_dim, action_dim=args.action_dim, device=device)

        # 重放缓冲区：使用简单的 SampleBuffer + DummyModuleWrapper
        buffer_capacity = 10 ** 6
        min_data = 256
        replay_buffer = DummyModuleWrapper(
            SampleBuffer(args.state_dim, args.action_dim, buffer_capacity, device=device)
        )

        # 保存目录与 PPO/SAC/TD3 一致：使用 eval_def_log_path
        os.makedirs(eval_def_log_path, exist_ok=True)

        score, sn, mean_step = 0.0, 0.0, 0.0

        for ep in range(args.train_step):
            evaluation_episode = ((ep + 1) % args.print_interval == 0)
            print(f"\r[SAC_lag] Episode: {ep + 1}/{args.train_step}", end="")

            state, _ = env.reset()
            done = False
            ep_step = 0

            while not done and ep_step < args.T_horizon:
                if len(replay_buffer) > min_data:
                    agent.update_parameters(replay_buffer, args.batch_size)

                ep_step += 1
                action = agent.select_action(state)  # 连续动作
                next_state, reward, done, _, info = env.step(action)

                replay_buffer.append(
                    states=state,
                    actions=action,
                    next_states=next_state,
                    rewards=reward,
                    costs=info.get("cost", 0.0),
                    dones=done,
                )

                state = next_state
                score += reward
                xa = info.get("x_position", 0.0)
                ya = info.get("y_position", 0.0)

            if not done:
                if args.env_name in ['TrafficEnv1-v0', 'TrafficEnv3-v0', 'TrafficEnv3-v1']:
                    if xa < -50.0 and ya > 4.0:
                        sn += 1
                elif args.env_name == 'TrafficEnv2-v0':
                    if xa > 50.0 and ya > -5.0:
                        sn += 1
                elif args.env_name == 'TrafficEnv4-v0':
                    if ya < -50.0:
                        sn += 1
            mean_step += ep_step

            # 定期保存策略网络（保存到与 PPO 等相同的目录）
            if (ep + 1) % 100 == 0:
                ckpt_path = os.path.join(eval_def_log_path, f"sac_lag_{ep + 1}.pt")
                agent.save_actor(ckpt_path)
                print(f"\n[SAC_lag] Model saved to {ckpt_path}")

            # 简单日志（如果使用 swanlab，则记录基础统计）
            if args.swanlab and evaluation_episode:
                avg_sn = sn / args.print_interval
                avg_rew = score / args.print_interval
                avg_len = mean_step / args.print_interval
                swanlab.log({"sn": avg_sn, "rew_mean": avg_rew, "len_mean": avg_len})
                score, sn, mean_step = 0.0, 0.0, 0.0

        env.close()


if __name__ == '__main__':
    main()
