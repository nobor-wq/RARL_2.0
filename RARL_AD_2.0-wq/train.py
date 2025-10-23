from stable_baselines3.common.callbacks import CheckpointCallback
import torch as th
import numpy as np
from config import get_config
from gymnasium.wrappers import TimeLimit
import gymnasium as gym
from stable_baselines3.common.monitor import Monitor
import swanlab
import Environment.environment
from stable_baselines3 import PPO, SAC, TD3
from wandb.integration.sb3 import WandbCallback
from callback import CustomEvalCallback_adv, CustomEvalCallback_def
import random
from buffer import PaddingRolloutBuffer, DecoupleRolloutBuffer, DecouplePaddingRolloutBuffer, ReplayBufferDefender
from adversarial_ppo import AdversarialPPO, AdversarialDecouplePPO
from defensive_sac import DefensiveSAC
from stable_baselines3.common.buffers import RolloutBuffer
from stable_baselines3.common.vec_env import DummyVecEnv, SubprocVecEnv
import os
from swanlab.integration.sb3 import SwanLabCallback

def create_model_adv(args, env, rollout_buffer_class, device, best_model_path, run=None):
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
    # 根据 decouple 来选择模型
    # if args.decouple:

    model_class = AdversarialDecouplePPO

    if run:
        model = model_class(args, best_model_path, "MlpPolicy",
                            env, n_steps=args.n_steps, batch_size=args.batch_size, verbose=1,
                            tensorboard_log=f"runs/{run.id}",
                            rollout_buffer_class=rollout_buffer_class, device=device)
    else:
        model = model_class(args, best_model_path,  "MlpPolicy",
                            env, n_steps=args.n_steps, batch_size=args.batch_size, verbose=1,
                            rollout_buffer_class=rollout_buffer_class,
                            device=device)
    # else:
    #     model_class = AdversarialPPO
    #     if run:
    #         model = model_class(args, best_model_path, "MlpPolicy",
    #                             env, n_steps=args.n_steps, batch_size=args.batch_size, verbose=1, tensorboard_log=f"runs/{run.id}",
    #                             rollout_buffer_class=rollout_buffer_class, device=device)
    #     else:
    #         model = model_class(args, best_model_path, "MlpPolicy",
    #                             env, n_steps=args.n_steps, batch_size=args.batch_size, verbose=1, rollout_buffer_class=rollout_buffer_class,
    #                             device=device)
    return model

def create_model_def(args, env, replay_buffer_class_def, device, best_model_path, start, run=None):
    """
    根据 args 配置来创建 基于PPO的敌手模型。
    如果需要 wandb，返回模型时会附带 wandb 配置信息。
    :param args: 系统参数
    :param env: 环境名
    :param replay_buffer_class_def: 回放池类型
    :param device: 模型加载设备名
    :param best_model_path: 最优模型存储路径
    :param start: 判断是否是刚开始的初始化
    :param run: 是否使用wandb进行日志记录
    """
    # 根据 decouple 来选择模型
    # if args.decouple:
    if start:
        model_class = SAC
    else:
        model_class = DefensiveSAC

    if run:
        model = model_class(args, best_model_path, "MlpPolicy",
                            env,  batch_size=args.batch_size, verbose=1,
                            tensorboard_log=f"runs/{run.id}",
                            rollout_buffer_class=replay_buffer_class_def, device=device)
    else:
        if start:
            model = model_class("MlpPolicy", env, verbose=1, learning_rate=args.lr, batch_size=args.batch_size, device=device,)

        else:
            model = model_class(args, best_model_path, "MlpPolicy",
                                env,  batch_size=args.batch_size, verbose=1,
                                replay_buffer_class=replay_buffer_class_def,
                                device=device)
    # else:
    #     model_class = AdversarialPPO
    #     if run:
    #         model = model_class(args, best_model_path, "MlpPolicy",
    #                             env, n_steps=args.n_steps, batch_size=args.batch_size, verbose=1, tensorboard_log=f"runs/{run.id}",
    #                             rollout_buffer_class=rollout_buffer_class, device=device)
    #     else:
    #         model = model_class(args, best_model_path, "MlpPolicy",
    #                             env, n_steps=args.n_steps, batch_size=args.batch_size, verbose=1, rollout_buffer_class=rollout_buffer_class,
    #                             device=device)
    return model



def main():
    # get parameters from config.py
    parser = get_config()
    args = parser.parse_args()


    # defenfer and attacker log path
    eval_def_log_path = os.path.join(args.path_def, args.algo, args.env_name, str(args.attack_eps), str(args.seed), args.addition_msg)
    os.makedirs(eval_def_log_path, exist_ok=True)
    best_model_path_def = os.path.join(eval_def_log_path, "best_model")
    eval_best_model_path_def = os.path.join(eval_def_log_path, "eval_best_model")

    eval_adv_log_path = os.path.join(args.path_adv, args.algo_adv, args.env_name, str(args.attack_eps), str(args.seed), args.addition_msg)
    os.makedirs(eval_adv_log_path, exist_ok=True)
    best_model_path_adv = os.path.join(eval_adv_log_path, "best_model")
    eval_best_model_path_adv = os.path.join(eval_adv_log_path, "eval_best_model")

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

    model_path_adv = os.path.join(eval_adv_log_path, 'model')
    os.makedirs(model_path_adv, exist_ok=True)
    checkpoint_callback_adv = CheckpointCallback(save_freq=args.save_freq, save_path=model_path_adv)

    # whether padding
    # rollout_buffer_map = {
    #     (True, True): DecouplePaddingRolloutBuffer,
    #     (True, False): PaddingRolloutBuffer,
    #     (False, True): DecoupleRolloutBuffer,
    #     (False, False): RolloutBuffer
    # }
    replay_buffer_class_def= ReplayBufferDefender
    rollout_buffer_class_adv = DecoupleRolloutBuffer

    def make_env(seed, rank, attack, eval_t=False):
        def _init():
            env = gym.make(args.env_name, attack=attack, eval=eval_t, adv_steps=args.adv_steps)
            env = TimeLimit(env, max_episode_steps=args.T_horizon)
            env = Monitor(env)
            env.unwrapped.start()
            env.reset(seed=seed + rank)
            return env

        return _init

    num_envs = args.num_envs if hasattr(args, 'num_envs') else 1
    if num_envs > 1:
        env_def_first= SubprocVecEnv([make_env(args.seed, i, False) for i in range(num_envs)])
        env_def = SubprocVecEnv([make_env(args.seed, i, False) for i in range(num_envs)])
        env_adv = SubprocVecEnv([make_env(args.seed, i, True) for i in range(num_envs)])
        env_adv_last = SubprocVecEnv([make_env(args.seed, i, True) for i in range(num_envs)])

    else:
        env_def_first = DummyVecEnv([make_env(args.seed, 0, False)])
        env_def = DummyVecEnv([make_env(args.seed, 0, False)])
        env_adv = DummyVecEnv([make_env(args.seed, 0, True)])
        env_adv_last = DummyVecEnv([make_env(args.seed, 0, True)])


    eval_env_def = DummyVecEnv([make_env(args.seed + 1000, 0, False)])
    eval_env_adv = DummyVecEnv([make_env(args.seed + 1000, 0, True, eval_t=True)])
    eval_env_adv_last = DummyVecEnv([make_env(args.seed + 1000, 0, True, eval_t=True)])

    # eval_env_adv = make_env(args.seed + 1000, 0, True, eval_t=True)()

    model_old_def = SAC("MlpPolicy", env_def, verbose=1, learning_rate=args.lr, batch_size=args.batch_size, device=device)
    model_old_adv = PPO("MlpPolicy", env_adv, verbose=1, learning_rate=args.lr, n_epochs=args.n_epochs, batch_size=args.batch_size, device=device, n_steps=args.n_steps)


    if args.swanlab:
        if args.adv_test:
            run_name = f"{args.attack_method}-{args.algo}-{args.seed}-only_attacker-{args.attack_eps}"
            run = swanlab.init(project="RARL", name=run_name, config=args)
            swan_cb = SwanLabCallback(project="RARL", experiment_name=run_name, verbose=2)

            model_adv = create_model_adv(args, env_adv, rollout_buffer_class_adv, device, best_model_path_adv)

            # 2025-10-16 wq 测试
            defense_base_model_path = "./logs/eval_def/" + os.path.join(args.algo, args.env_name, str(args.attack_eps), str(args.seed),  "lunar")
            model_def = SAC.load(defense_base_model_path, device=device)


            eval_callback_adv = CustomEvalCallback_adv(eval_env_adv, trained_agent=model_def,
                                                       attack_eps=args.attack_eps,
                                                       best_model_save_path=eval_best_model_path_adv,
                                                       n_eval_episodes=20,
                                                       eval_freq=args.n_steps * 10,
                                                       unlimited_attack=args.unlimited_attack,
                                                       attack_method=args.attack_method)

            model_adv.learn(total_timesteps=args.train_step * args.n_steps, progress_bar=True,
                            callback=[checkpoint_callback_adv, swan_cb, eval_callback_adv],
                            trained_def=model_def, reset_num_timesteps=False, log_interval=args.print_interval)
        else:
            run_name = f"{args.attack_method}-{args.algo}-{args.seed}-{args.addition_msg}"
            run = swanlab.init(project="RARL", name=run_name, config=args)
            swan_cb = SwanLabCallback(project="RARL", experiment_name=run_name, verbose=2)

            # 2025-10-02 wq 初始化训练
            model_def_first = create_model_def(args, env_def_first, replay_buffer_class_def, device,
                                               best_model_path_def, True)
            model_def_first.learn(total_timesteps=args.train_step * args.n_steps, progress_bar=True,
                                  callback=checkpoint_callback_def)

            env_def_first.close()

            model_adv = create_model_adv(args, env_adv, rollout_buffer_class_adv, device, best_model_path_adv)

            # 2025-10-04 wq 需要另外的模型来加载上次训练好的模型
            model_def = create_model_def(args, env_def, replay_buffer_class_def, device, best_model_path_def, False)
            model_def.actor.load_state_dict(model_def_first.actor.state_dict())

            for i in range(args.loop_nums):
                model_old_def.actor.load_state_dict(model_def.actor.state_dict())

                eval_callback_adv = CustomEvalCallback_adv(eval_env_adv, trained_agent=model_old_def,
                                                           attack_eps = args.attack_eps,
                                                           best_model_save_path=eval_best_model_path_adv,
                                                           n_eval_episodes=20,
                                                           eval_freq=args.n_steps * 10,
                                                           unlimited_attack=args.unlimited_attack,
                                                           attack_method=args.attack_method)
                #
                model_adv.learn(total_timesteps=args.train_step * args.n_steps, progress_bar=True,
                                callback=[checkpoint_callback_adv, swan_cb, eval_callback_adv],
                                trained_def=model_old_def,  reset_num_timesteps=False, log_interval = args.print_interval)

                model_old_adv.policy.load_state_dict(model_adv.policy.state_dict())

                eval_callback_def = CustomEvalCallback_def(eval_env_adv, trained_agent=model_old_def,
                                                           trained_adv=model_old_adv,
                                                           attack_eps=args.attack_eps,
                                                           best_model_save_path=eval_best_model_path_def,
                                                           n_eval_episodes=20,
                                                           eval_freq=args.n_steps * 10,
                                                           unlimited_attack=args.unlimited_attack,
                                                           attack_method=args.attack_method)

                model_def.learn(total_timesteps=args.train_step * args.n_steps, progress_bar=True,
                                callback=[checkpoint_callback_def, swan_cb, eval_callback_def], trained_agent=model_old_def,
                                trained_adv=model_old_adv,  reset_num_timesteps=False, log_interval = args.print_interval)

            model_adv_last = create_model_adv(args, env_adv_last, rollout_buffer_class_adv, device, best_model_path_adv)
            eval_callback_adv = CustomEvalCallback_adv(eval_env_adv, trained_agent=model_def,
                                                       attack_eps=args.attack_eps,
                                                       best_model_save_path=eval_best_model_path_adv,
                                                       n_eval_episodes=20,
                                                       eval_freq=args.n_steps * 10,
                                                       unlimited_attack=args.unlimited_attack,
                                                       attack_method=args.attack_method)
            model_adv_last.learn(total_timesteps=args.train_step * args.n_steps * args.loop_nums, progress_bar=True,
                            callback=[checkpoint_callback_adv, swan_cb, eval_callback_adv],
                            trained_def=model_def, reset_num_timesteps=False, log_interval=args.print_interval)



    else:
        # 2025-10-02 wq 初始化训练
        model_def_first = create_model_def(args, env_def_first, replay_buffer_class_def, device,
                                           best_model_path_def, True)
        model_def_first.learn(total_timesteps=args.train_step * args.n_steps, progress_bar=True,
                              callback=checkpoint_callback_def)

        env_def_first.close()

        model_adv = create_model_adv(args, env_adv, rollout_buffer_class_adv, device, best_model_path_adv)

        # 2025-10-04 wq 需要另外的模型来加载上次训练好的模型
        model_def = create_model_def(args, env_def, replay_buffer_class_def, device, best_model_path_def, False)
        model_def.actor.load_state_dict(model_def_first.actor.state_dict())

        for i in range(args.loop_nums):
            model_old_def.actor.load_state_dict(model_def.actor.state_dict())

            eval_callback_adv = CustomEvalCallback_adv(eval_env_adv, trained_agent=model_old_def,
                                                       attack_eps=args.attack_eps,
                                                       best_model_save_path=eval_best_model_path_adv,
                                                       n_eval_episodes=20,
                                                       eval_freq=args.n_steps * 10,
                                                       unlimited_attack=args.unlimited_attack,
                                                       attack_method=args.attack_method)
            #
            model_adv.learn(total_timesteps=args.train_step * args.n_steps, progress_bar=True,
                            callback=[checkpoint_callback_adv, eval_callback_adv],
                            trained_def=model_old_def, reset_num_timesteps=False, log_interval=args.print_interval)

            model_old_adv.policy.load_state_dict(model_adv.policy.state_dict())

            eval_callback_def = CustomEvalCallback_def(eval_env_adv, trained_agent=model_old_def,
                                                       trained_adv=model_old_adv,
                                                       attack_eps=args.attack_eps,
                                                       best_model_save_path=eval_best_model_path_def,
                                                       n_eval_episodes=20,
                                                       eval_freq=args.n_steps * 10,
                                                       unlimited_attack=args.unlimited_attack,
                                                       attack_method=args.attack_method)

            model_def.learn(total_timesteps=args.train_step * args.n_steps, progress_bar=True,
                            callback=[checkpoint_callback_def, eval_callback_def], trained_agent=model_old_def,
                            trained_adv=model_old_adv, reset_num_timesteps=False, log_interval=args.print_interval)


    # Save the agent
    eval_env_def.close()
    eval_env_adv.close()

    model_adv.save(os.path.join(eval_adv_log_path, "lunar"))
    del model_adv
    env_adv.close()

    model_def.save(os.path.join(eval_def_log_path, "lunar"))
    del model_def
    env_def.close()


if __name__ == '__main__':
    main()
