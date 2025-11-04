from stable_baselines3.common.callbacks import CheckpointCallback
import torch as th
import numpy as np
from config import get_config
from gymnasium.wrappers import TimeLimit
import gymnasium as gym
from stable_baselines3.common.monitor import Monitor
import swanlab
import Environment.environment
from stable_baselines3 import PPO, SAC
from callback import CustomEvalCallback_adv, CustomEvalCallback_def
import random
from buffer import DecoupleRolloutBuffer, ReplayBufferDefender, DualReplayBufferDefender
from adversarial_ppo import AdversarialDecouplePPO
from defensive_sac import DefensiveSAC
from stable_baselines3.common.vec_env import DummyVecEnv, SubprocVecEnv
import os
from swanlab.integration.sb3 import SwanLabCallback
from policy import IGCARLNet
from stable_baselines3.common.utils import obs_as_tensor
from fgsm import FGSM_v2

def final_evaluation(args, final_defender, final_attacker, device):
    """
    在整个训练流程结束后，对最终的防御者和攻击者进行一次独立的评估。
    模仿 defense_test.py 的逻辑。
    """
    print("\n--- Starting Final Evaluation ---")

    np.random.seed(args.seed + 100)  # 设置 NumPy 随机种子
    th.manual_seed(args.seed + 100)  # 设置 CPU 随机种子
    if th.cuda.is_available():
        th.cuda.manual_seed(args.seed + 100)  # 设置 CUDA 随机种子
        th.cuda.manual_seed_all(args.seed + 100)  # 设置所有 GPU 随机种子
    th.backends.cudnn.deterministic = True  # 确保 CUDA 的确定性
    th.backends.cudnn.benchmark = False  # 禁用 CuDNN 自动优化

    # 设置eval标志
    args.eval = True
    # 创建环境
    eval_env = gym.make(args.env_name, attack=True, adv_steps=args.adv_steps, eval=args.eval, use_gui=args.use_gui,
                   render_mode=args.render_mode)
    eval_env.unwrapped.start()
    # 2. 准备模型 (它们已经从参数传入)
    # 确保模型处于评估模式
    final_defender.policy.set_training_mode(False)
    final_attacker.policy.set_training_mode(False)

    # 3. 循环评估
    rewards = []
    num_eval_episodes = 500  # 可以设一个比训练中评估更精确的次数
    sn = 0

    for _ in range(num_eval_episodes):
        obs, _ = eval_env.reset(options="random")
        episode_reward = 0.0
        for _ in range(args.T_horizon):
            # --- 这部分逻辑完全来自于你的 defense_test.py ---
            obs_tensor = obs_as_tensor(obs, device)

            # 防御者生成原始动作
            with th.no_grad():
                actions, _ = final_defender.predict(obs_tensor[:-2].cpu(), deterministic=True)
            actions_tensor = th.tensor(actions, device=obs_tensor.device)  # 确保在同一设备上
            obs_tensor[-1] = actions_tensor
            # 攻击者生成攻击动作
            with th.no_grad():
                adv_actions, _ = final_attacker.predict(obs_tensor.cpu(), deterministic=True)

            adv_action_mask = (adv_actions[0] > 0) & (obs[-2] > 0)
            # print("DEBUG train.py adv_action_mask:", adv_action_mask, "adv_actions:", adv_actions, "obs[-2]:", obs[-2])

            if adv_action_mask.any():
                adv_state = FGSM_v2(adv_actions[1], victim_agent=final_defender,
                                    last_state=obs_tensor[:-2].unsqueeze(0), epsilon=args.attack_eps, device=device)
                action_perturbed, _ = final_defender.predict(adv_state.cpu(), deterministic=True)
                final_action = action_perturbed
                print("DEBUG train.py action before attack:", actions, "attack action is: ", adv_actions, "after attack:", final_action)
            else:
                final_action = actions

            # 组合最终动作并与环境交互
            action = np.column_stack((final_action, adv_action_mask))
            obs, reward, done, terminate, info = eval_env.step(action)

            if isinstance(info, dict):
                info0 = info
            elif isinstance(info, (list, tuple)) and len(info) > 0:
                info0 = info[0]
            else:
                raise ValueError(f"Invalid infos format: {type(info)}")
            if 'reward' not in info0:
                raise KeyError(f"'reward' key not found in info: {info0}")

            r_def = float(info0['reward'])
            c_def = float(info0['cost'])
            episode_reward += r_def - c_def

            xa = info['x_position']
            ya = info['y_position']

            if done:
                break
        if args.env_name == 'TrafficEnv1-v0' or args.env_name == 'TrafficEnv3-v0' or args.env_name == 'TrafficEnv6-v0':
            if xa < -50.0 and ya > 4.0 and done is False:
                sn += 1

        rewards.append(episode_reward)

    eval_env.close()

    mean_reward = np.mean(rewards)
    mean_success = sn / num_eval_episodes
    print(f"--- Final Evaluation Result: Mean Reward = {mean_reward:.2f} ---")
    print(f"--- Final Evaluation Result: Success Rate = {mean_success:.2f} ---\n")

    return mean_success



def create_model_adv(args, env, device, best_model_path):
    """
    根据 args 配置来创建 基于PPO的敌手模型。
    如果需要 wandb，返回模型时会附带 wandb 配置信息。
    :param args: 系统参数
    :param env: 环境名
    :param rollout_buffer_class: 回放池类型
    :param device: 模型加载设备名
    :param best_model_path: 最优模型存储路径
    """
    # 根据 decouple 来选择模型
    # if args.decouple:
    model_class = AdversarialDecouplePPO
    rollout_buffer_class_adv = DecoupleRolloutBuffer
    ppo_device = 'cpu'

    model = model_class(args, best_model_path,  "MlpPolicy",
                        env, n_steps=args.n_steps, batch_size=args.batch_size, verbose=1,
                        rollout_buffer_class=rollout_buffer_class_adv,
                        device=device)
    return model

def create_model_def(args, env, device, best_model_path, start):
    """
    根据 args 配置来创建 基于PPO的敌手模型。
    如果需要 wandb，返回模型时会附带 wandb 配置信息。
    :param args: 系统参数
    :param env: 环境名
    :param device: 模型加载设备名
    :param best_model_path: 最优模型存储路径
    :param start: 判断是否是刚开始的初始化
    """
    # 根据 decouple 来选择模型
    if start:
        # 预训练阶段，使用标准的SAC和它默认的Buffer
        model_class = SAC
        model = model_class(
            "MlpPolicy",
            env,
            verbose=1,
            learning_rate=args.lr_def,
            batch_size=args.batch_size,
            device=device,
        )
    else:
        # 对抗训练阶段，使用我们自定义的DefensiveSAC
        model_class = DefensiveSAC

        # 1. 初始化一个空的参数字典
        replay_buffer_kwargs = {}

        # 2. 根据条件选择Buffer类并填充参数字典
        if args.use_DualBuffer:
            replay_buffer_class_def = DualReplayBufferDefender
            # 只有当使用DualBuffer时，才需要这个特定参数
            replay_buffer_kwargs["adv_sample_ratio"] = args.adv_sample_ratio  # 您可以将其改为 args.adv_sample_ratio
            # 如果未来还有其他参数，可以继续添加，例如:
            # replay_buffer_kwargs["another_param"] = 123
        else:
            replay_buffer_class_def = ReplayBufferDefender
            # 使用旧的Buffer时，kwargs为空字典，不会传入任何额外参数，是安全的

        # 3. 在创建模型时，同时传入 replay_buffer_class 和 replay_buffer_kwargs
        model = model_class(
            args,
            best_model_path,
            "MlpPolicy",
            env,
            batch_size=args.batch_size,
            learning_rate=args.lr_def,
            verbose=1,
            replay_buffer_class=replay_buffer_class_def,
            replay_buffer_kwargs=replay_buffer_kwargs,  # <--- 将参数字典传递给模型
            device=device,
        )

    return model



def run_training(args):
    # get parameters from config.py
    # parser = get_config()
    # args = parser.parse_args()

    msg_parts = []
    if args.action_diff:
        msg_parts.append("action_diff")
        if args.use_expert:
            msg_parts.append("expert")
        # 依赖于 expert 的技术，可以进行嵌套
        if args.use_kl:
            msg_parts.append("kl")
        elif args.use_lagrangian:
            msg_parts.append("lag")
        # 新增的 Buffer 技术
    if args.use_DualBuffer:
        msg_parts.append("DualBuffer")
    if not msg_parts:
        # 如果没有任何特殊技术，就是一个基础版本
        addition_msg = "base"
    else:
        addition_msg = "_".join(msg_parts)

    # (可选但推荐) 将生成的消息存回args，方便后续使用
    args.addition_msg = addition_msg


    print(f"[*] 本次实验标签: {args.addition_msg}")  # 打印出来方便确认
    # defenfer and attacker log path
    eval_def_log_path = os.path.join(args.path_def, args.algo, args.env_name, args.addition_msg, str(args.attack_eps), str(args.seed), str(args.train_step))
    os.makedirs(eval_def_log_path, exist_ok=True)
    best_model_path_def = os.path.join(eval_def_log_path, "best_model")
    eval_best_model_path_def = os.path.join(eval_def_log_path, "eval_best_model")

    eval_adv_log_path = os.path.join(args.path_adv, args.algo_adv, args.env_name,  args.algo, args.addition_msg, str(args.attack_eps), str(args.seed), str(args.train_step))
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
    # 2025-10-26 wq 统一的回调函数列表初始化
    callbacks_common = []

    if args.swanlab:
        run_name = f"{args.attack_method}-{args.algo}-{args.seed}-{args.attack_eps}-{args.addition_msg}-{args.train_step}"
        run = swanlab.init(project="RARL", name=run_name, config=args)
        swan_cb = SwanLabCallback(project="RARL", experiment_name=run_name, verbose=2)
        callbacks_common.append(swan_cb)

    if args.adv_test:
        run_name_adv = f"{args.attack_method}-{args.algo}-{args.seed}-only_attacker-{args.attack_eps}"
        # run = swanlab.init(project="RARL", name=run_name, config=args)
        swan_cb = SwanLabCallback(project="RARL", experiment_name=run_name_adv, verbose=2)
        model_adv = create_model_adv(args, env_adv, device, best_model_path_adv)

        # 2025-10-16 wq 测试
        if args.algo == "IGCARL":
            prefix = "./logs/eval_def/" + os.path.join(args.algo, args.env_name)
            filename = f'{args.model_name}.pth'
            model_path_drl = os.path.join(prefix, filename)
            if not os.path.isfile(model_path_drl):
                raise FileNotFoundError(f"找不到模型文件：{model_path_drl}")
            model_def = IGCARLNet(state_dim=26, action_dim=1).to(device)
            model_def.load_state_dict(th.load(model_path_drl, map_location=device))
            model_def.eval()
        elif args.algo == "RARL":
            defense_base_model_path = "./logs/eval_def/" + os.path.join(args.algo, args.env_name, str(args.attack_eps), str(args.seed),  "lunar")
            model_def = SAC.load(defense_base_model_path, device=device)


        eval_callback_adv = CustomEvalCallback_adv(eval_env_adv, trained_agent=model_def,
                                                   attack_eps=args.attack_eps,
                                                   best_model_save_path=eval_best_model_path_adv,
                                                   n_eval_episodes=20,
                                                   eval_freq=args.n_steps * 10,
                                                   unlimited_attack=args.unlimited_attack,
                                                   attack_method=args.attack_method)

        model_adv.learn(total_timesteps=args.train_step * args.n_steps * args.loop_nums, progress_bar=True,
                        callback=[checkpoint_callback_adv, swan_cb, eval_callback_adv],
                        trained_def=model_def, reset_num_timesteps=False, log_interval=args.print_interval)

        eval_env_def.close()
        eval_env_adv.close()


        model_adv.save(os.path.join(eval_adv_log_path, "lunar"))
        del model_adv
        env_adv.close()

        del model_def
        env_def.close()

    else:
        # 2025-10-26 wq 加载专家模型
        # eval_env_adv = make_env(args.seed + 1000, 0, True, eval_t=True)()
        defense_model_expert_path = os.path.join(args.path_def, "base", args.algo, args.env_name, "1", "lunar")
        trained_expert = SAC.load(defense_model_expert_path, device=device)
        # 2025-10-26 wq 初始化占位用的“旧”模型
        model_old_def = SAC("MlpPolicy", env_def, device=device)  # 只需要结构，不需要训练
        model_old_adv = PPO("MlpPolicy", env_adv, device=device)  # PPO结构

        # 2025-10-02 wq 防御者预训练 (Standard SAC)
        print("=== Phase 1: Pre-training Defender (Standard SAC) ===")
        # model_def_pre = create_model_def(args, env_def_first, device, best_model_path_def, True)
        # model_def_pre.learn(total_timesteps=args.train_step * args.n_steps, progress_bar=True,
        #                       callback=checkpoint_callback_def)
        #
        # print("Defender pre-training completed.")

        # 2025-10-04 wq 需要另外的模型来加载上次训练好的模型
        model_def = create_model_def(args, env_def, device, best_model_path_def, False)
        model_def.policy.load_state_dict(trained_expert.policy.state_dict())  # 复制完整策略


        env_def_first.close()
        # del model_def_pre
        model_adv = create_model_adv(args, env_adv, device, best_model_path_adv)


        print("=== Phase 2: Adversarial Training Loop ===")
        for i in range(args.loop_nums):
            print(f"--- Loop {i + 1}/{args.loop_nums} ---")
            # 2025-10-26 wq 更新旧防御者用于攻击者训练
            # model_old_def.actor.load_state_dict(model_def.actor.state_dict())
            model_old_def.policy.load_state_dict(model_def.policy.state_dict())
            model_old_def.policy.set_training_mode(False)  # 确保是评估模式

            eval_callback_adv = CustomEvalCallback_adv(eval_env_adv, trained_agent=model_old_def,
                                                       attack_eps = args.attack_eps,
                                                       best_model_save_path=eval_best_model_path_adv,
                                                       n_eval_episodes=args.n_eval_episodes,
                                                       eval_freq=args.n_steps * 10,
                                                       unlimited_attack=args.unlimited_attack,
                                                       attack_method=args.attack_method)
            #
            model_adv.learn(total_timesteps=args.train_step * args.n_steps, progress_bar=True,
                            callback=[checkpoint_callback_adv, eval_callback_adv] + callbacks_common,
                            trained_def=model_old_def,  reset_num_timesteps=False, log_interval = args.print_interval)
            # 2025-10-26 wq 更新旧攻击者用于防御者训练
            model_old_adv.policy.load_state_dict(model_adv.policy.state_dict())
            model_old_adv.policy.set_training_mode(False)

            eval_callback_def = CustomEvalCallback_def(eval_env_adv, trained_agent=model_old_def,
                                                       trained_adv=model_old_adv,
                                                       attack_eps=args.attack_eps,
                                                       best_model_save_path=eval_best_model_path_def,
                                                       n_eval_episodes=args.n_eval_episodes,
                                                       eval_freq=args.n_steps * 10,
                                                       unlimited_attack=args.unlimited_attack,
                                                       attack_method=args.attack_method)


            print("Training Defender...")
            model_def.learn(total_timesteps=args.train_step * args.n_steps, progress_bar=True,
                            callback=[checkpoint_callback_def, eval_callback_def] + callbacks_common, trained_agent=model_old_def,
                            trained_adv=model_old_adv, trained_expert=trained_expert, reset_num_timesteps=False, log_interval = args.print_interval)
        # 2025-10-26 wq 最后阶段：攻击测试
        print("=== Phase 3: Final Attack Phase ===")

        # model_old_def.actor.load_state_dict(model_def.actor.state_dict())
        model_old_def.policy.load_state_dict(model_def.policy.state_dict())
        model_old_def.policy.set_training_mode(False)

        model_adv_last = create_model_adv(args, env_adv_last, device, best_model_path_adv)
        eval_callback_adv_last = CustomEvalCallback_adv(eval_env_adv_last, trained_agent=model_old_def,
                                                   attack_eps=args.attack_eps,
                                                   best_model_save_path=eval_best_model_path_adv,
                                                   n_eval_episodes=args.n_eval_episodes,
                                                   eval_freq=args.n_steps * 10,
                                                   unlimited_attack=args.unlimited_attack,
                                                   attack_method=args.attack_method)
        model_adv_last.learn(total_timesteps=args.train_step * args.n_steps * args.loop_nums, progress_bar=True,
                        callback=[checkpoint_callback_adv, eval_callback_adv_last] + callbacks_common,
                        trained_def=model_old_def, reset_num_timesteps=True, log_interval=args.print_interval)

        # final_mean_reward = final_evaluation(args, model_def, model_adv_last, device)

        # Save the agent
        eval_env_def.close()
        eval_env_adv.close()
        eval_env_adv_last.close()

        th.save(model_adv_last.policy.state_dict(), os.path.join(eval_adv_log_path, "policy.pth"))
        del model_adv_last
        env_adv_last.close()

        del model_adv
        env_adv.close()

        # model_def.save(os.path.join(eval_def_log_path, "lunar"), exclude=["replay_buffer"])
        th.save(model_def.policy.state_dict(), os.path.join(eval_def_log_path, "policy.pth"))

        del model_def
        env_def.close()

        # return final_mean_reward

def main():
    # get parameters from config.py
    parser = get_config()
    args = parser.parse_args()
    run_training(args) # 直接调用新的训练函数

if __name__ == '__main__':
    main()