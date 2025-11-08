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
from defensive_sac import DefensiveSAC, BaseSAC
from stable_baselines3.common.vec_env import DummyVecEnv, SubprocVecEnv
import os
from swanlab.integration.sb3 import SwanLabCallback
from policy import IGCARLNet
from stable_baselines3.common.utils import obs_as_tensor
from fgsm import FGSM_v2

def final_evaluation(args, final_defender, final_attacker, device, attack, is_swanlab, eval_episode=200):
    """
    在整个训练流程结束后，对最终的防御者和攻击者进行一次独立的评估。
    模仿 defense_test.py 的逻辑。
    """
    print("\n--- Starting Evaluation ---")

    np.random.seed(args.seed + 300)  # 设置 NumPy 随机种子
    th.manual_seed(args.seed + 300)  # 设置 CPU 随机种子
    if th.cuda.is_available():
        th.cuda.manual_seed(args.seed + 300)  # 设置 CUDA 随机种子
        th.cuda.manual_seed_all(args.seed + 300)  # 设置所有 GPU 随机种子
    th.backends.cudnn.deterministic = True  # 确保 CUDA 的确定性
    th.backends.cudnn.benchmark = False  # 禁用 CuDNN 自动优化

    # 设置eval标志
    args.eval = True
    # 创建环境
    eval_env = gym.make(args.env_name, attack=attack, adv_steps=args.adv_steps, eval=args.eval, use_gui=args.use_gui,
                   render_mode=args.render_mode)
    eval_env.unwrapped.start()

    # 3. 循环评估
    rewards = []
    num_eval_episodes = eval_episode
    sn = 0
    # 2025-11-04 wq 成功率，奖励，攻击成功率，平均攻击次数
    attack_sn = 0
    total_attack_times = 0


    for _ in range(num_eval_episodes):
        obs, _ = eval_env.reset(options="random")
        episode_reward = 0.0
        for _ in range(args.T_horizon):
            # --- 这部分逻辑完全来自于你的 defense_test.py ---
            obs_tensor = obs_as_tensor(obs, device)

            if attack:
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
                    total_attack_times += 1
                    if is_swanlab:
                        swanlab.log({"Eval/agent action before:": actions.item()})
                        swanlab.log({"Eval/agent adv action:": adv_actions[1].item()})
                        swanlab.log({"Eval/agent action after attack:": final_action.item()})

                    print("DEBUG train.py action before attack:", actions, "attack action is: ", adv_actions, "after attack:", final_action)
                else:
                    final_action = actions

                # 组合最终动作并与环境交互
                action = np.column_stack((final_action, adv_action_mask))
                obs, reward, done, terminate, info = eval_env.step(action)

                if adv_action_mask.any() and done is True:
                    attack_sn += 1

                # if isinstance(info, dict):
                #     info0 = info
                # elif isinstance(info, (list, tuple)) and len(info) > 0:
                #     info0 = info[0]
                # else:
                #     raise ValueError(f"Invalid infos format: {type(info)}")
                # if 'reward' not in info0:
                #     raise KeyError(f"'reward' key not found in info: {info0}")
                r_def = float(info['reward'])
                c_def = float(info['cost'])
                episode_reward += r_def - c_def
            else:
                with th.no_grad():
                    actions, _ = final_defender.predict(obs_tensor.cpu(), deterministic=True)
                obs, reward, done, terminate, info = eval_env.step(actions)
                episode_reward += reward
            if done:
                break

        xa = info['x_position']
        ya = info['y_position']

        if args.env_name == 'TrafficEnv1-v0' or args.env_name == 'TrafficEnv3-v0' or args.env_name == 'TrafficEnv6-v0':
            if xa < -50.0 and ya > 4.0 and done is False:
                sn += 1
        rewards.append(episode_reward)
    eval_env.close()

    mean_reward = np.mean(rewards)
    mean_success = sn / num_eval_episodes
    if is_swanlab:
        if attack:
            swanlab.log({"Eval/With_Attacker_Mean_Reward": mean_reward})
            swanlab.log({"Eval/With_Attacker_Success_Rate": mean_success})
            mean_attack_success = attack_sn / num_eval_episodes
            mean_attack_times = total_attack_times / num_eval_episodes
            swanlab.log({"Eval/Attack_Success_Rate": mean_attack_success})
            swanlab.log({"Eval/Average_Attack_Times": mean_attack_times})

            print(f"--- Final Evaluation Result: Mean Reward = {mean_reward:.2f} ---")
            print(f"--- Final Evaluation Result: Success Rate = {mean_success:.2f} ---")
            print(f"--- Final Evaluation Result: Attack Success Rate = {mean_attack_success:.2f} ---")
            print(f"--- Final Evaluation Result: Average Attack Times = {mean_attack_times:.2f} ---\n")
        else:
            swanlab.log({"Eval/No_Attacker_Mean_Reward": mean_reward})
            swanlab.log({"Eval/No_Attacker_Success_Rate": mean_success})
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

def create_model_def(args, env, device, best_model_path):
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

    print("Using device:", device)


    # 设置随机种子
    def set_seeds(seed):
        random.seed(seed)  # 设置 Python 随机种子
        np.random.seed(seed)  # 设置 NumPy 随机种子
        th.manual_seed(seed)  # 设置 CPU 随机种子
        if th.cuda.is_available():
            th.cuda.manual_seed(seed)  # 设置 CUDA 随机种子
            th.cuda.manual_seed_all(seed)  # 设置所有 GPU 随机种子
        th.backends.cudnn.deterministic = True  # 确保 CUDA 的确定性
        th.backends.cudnn.benchmark = False  # 禁用 CuDNN 自动优化

    set_seeds(args.seed)
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
        env_def = SubprocVecEnv([make_env(args.seed, i, False) for i in range(num_envs)])
        env_adv = SubprocVecEnv([make_env(args.seed, i, True) for i in range(num_envs)])
        env_adv_last = SubprocVecEnv([make_env(args.seed, i, True) for i in range(num_envs)])

    else:
        env_def = DummyVecEnv([make_env(args.seed, 0, False)])
        env_adv = DummyVecEnv([make_env(args.seed, 0, True)])
        env_adv_last = DummyVecEnv([make_env(args.seed, 0, True)])


    eval_env_def = DummyVecEnv([make_env(args.seed + 1000, 0, False)])
    eval_env_adv = DummyVecEnv([make_env(args.seed + 1000, 0, True, eval_t=True)])
    eval_env_adv_last = DummyVecEnv([make_env(args.seed + 100, 0, True, eval_t=True)])
    # 2025-10-26 wq 统一的回调函数列表初始化
    callbacks_common = []

    if args.swanlab:
        if args.adv_test:
            run_name = f"attacker-only-{args.algo}-{args.seed}-{args.attack_eps}-{args.addition_msg}-{args.train_step}"
        else:
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
            defense_model_path = "./logs/eval_def/" + os.path.join(args.algo, args.env_name, addition_msg, str(args.train_eps), str(args.trained_seed), str(args.trained_step))
            if args.best_model:
                model_path = os.path.join(defense_model_path, 'policy_best.pth')
            elif args.eval_best_model:
                model_path = os.path.join(defense_model_path, 'eval_best_model', 'policy_eval_best.pth')
            else:
                model_path = os.path.join(defense_model_path, 'policy.pth')

            print("DEBUG def model path: ", model_path)
            temp_def_env = gym.make(args.env_name, attack=False)
            trained_agent = SAC("MlpPolicy", temp_def_env, verbose=1, device=device)
            state_dict = th.load(model_path, map_location=device)
            trained_agent.policy.load_state_dict(state_dict)
        elif args.algo == "SAC":
            defense_model_expert_path = os.path.join(args.path_def, args.algo, args.env_name, "lunar")
            trained_agent = SAC.load(defense_model_expert_path, device=device)

        model_adv.learn(total_timesteps=args.train_step * args.n_steps * args.loop_nums, progress_bar=True,
                        callback=[checkpoint_callback_adv] + callbacks_common, trained_def=trained_agent,
                        reset_num_timesteps=False, log_interval=args.print_interval)

        mean_success = final_evaluation(args, trained_agent, model_adv, device, attack=False, is_swanlab=args.swanlab, eval_episode=args.eval_episode)
        final_mean_success = final_evaluation(args, trained_agent, model_adv, device, attack=True, is_swanlab=args.swanlab, eval_episode=args.eval_episode)

        eval_env_def.close()
        eval_env_adv.close()

        th.save(model_adv.policy.state_dict(), os.path.join(eval_adv_log_path, "policy.pth"))
        del model_adv
        env_adv.close()
        env_def.close()
        if args.algo == "RARL":
            temp_def_env.close()

    else:
        # 2025-10-26 wq 加载专家模型
        # eval_env_adv = make_env(args.seed + 1000, 0, True, eval_t=True)()
        defense_model_expert_path = os.path.join(args.path_def, "base", args.algo, args.env_name, "1", "lunar")
        trained_expert = SAC.load(defense_model_expert_path, device=device)

        pretrain_model = BaseSAC(
            "MlpPolicy",
            env_def,
            verbose=1,
            learning_rate=args.lr_def,
            batch_size=args.batch_size,
            device=device,
        )
        pretrain_model.learn(
            total_timesteps=args.train_step * args.n_steps * args.loop_nums,
            log_interval=args.print_interval,
            progress_bar=True,
            callback = [checkpoint_callback_def] + callbacks_common,
            reset_num_timesteps=False
        )
        pretrain_policy_state_dict = pretrain_model.policy.state_dict()
        del pretrain_model

        # 2025-10-02 wq 防御者预训练 (Standard SAC)
        model_def = create_model_def(args, env_def, device, best_model_path_def)
        model_adv = create_model_adv(args, env_adv, device, best_model_path_adv)
        from copy import deepcopy
        model_def.policy.load_state_dict(deepcopy(pretrain_policy_state_dict))

        # 2025-10-26 wq 初始化占位用的“旧”模型
        model_old_def = SAC("MlpPolicy", env_def, device=device)  # 只需要结构，不需要训练
        model_old_adv = PPO("MlpPolicy", env_adv, device=device)  # PPO结构


        mean_success = final_evaluation(args, model_def, model_adv, device, attack=False, is_swanlab=args.swanlab, eval_episode=args.eval_episode)

        print("=== Phase 2: Adversarial Training Loop ===")
        for i in range(args.loop_nums):
            env_adv.reset()
            print(f"--- Loop {i + 1}/{args.loop_nums} --- attacker training ---")
            # 2025-10-26 wq 更新旧防御者用于攻击者训练
            # model_old_def.actor.load_state_dict(model_def.actor.state_dict())
            model_old_def.policy.load_state_dict(model_def.policy.state_dict())
            model_old_def.policy.set_training_mode(False)  # 确保是评估模式

            model_adv.learn(total_timesteps=args.train_step * args.n_steps, progress_bar=True,
                            callback=[checkpoint_callback_adv] + callbacks_common,
                            trained_def=model_old_def, log_interval = args.print_interval, reset_num_timesteps=False)
            print(f"--- Loop {i + 1}/{args.loop_nums} --- attacker training over---")

            mean_success = final_evaluation(args, model_def, model_adv, device, attack=False, is_swanlab=args.swanlab, eval_episode=args.eval_episode)
            mean_success = final_evaluation(args, model_def, model_adv, device, attack=True, is_swanlab=args.swanlab, eval_episode=args.eval_episode)

            print(f"--- Loop {i + 1}/{args.loop_nums} --- defender training ---")
            # 2025-10-26 wq 更新旧攻击者用于防御者训练
            env_def.reset()
            model_old_adv.policy.load_state_dict(model_adv.policy.state_dict())
            model_old_adv.policy.set_training_mode(False)

            print("Training Defender...")
            model_def.learn(total_timesteps=args.train_step * args.n_steps, progress_bar=True,
                            callback=[checkpoint_callback_def] + callbacks_common, trained_agent=model_old_def,
                            trained_adv=model_old_adv, trained_expert=trained_expert, reset_num_timesteps=False, log_interval = args.print_interval)
            print(f"--- Loop {i + 1}/{args.loop_nums} --- defender training over---")

            mean_success = final_evaluation(args, model_def, model_adv, device, attack=False, is_swanlab=args.swanlab, eval_episode=args.eval_episode)
            mean_success = final_evaluation(args, model_def, model_adv, device, attack=True, is_swanlab=args.swanlab, eval_episode=args.eval_episode)


        # 2025-10-26 wq 最后阶段：攻击测试
        print("=== Phase 3: Final Attack Phase ===")
        set_seeds(args.seed + 200)
        # model_old_def.actor.load_state_dict(model_def.actor.state_dict())
        model_old_def.policy.load_state_dict(model_def.policy.state_dict())
        model_old_def.policy.set_training_mode(False)

        model_adv_last = create_model_adv(args, env_adv_last, device, best_model_path_adv)

        model_adv_last.learn(total_timesteps=args.train_step * args.n_steps * args.loop_nums, progress_bar=True,
                        callback=[checkpoint_callback_adv] + callbacks_common,
                        trained_def=model_old_def, reset_num_timesteps=True, log_interval=args.print_interval)

        mean_success = final_evaluation(args, model_def, model_adv_last, device, attack=False, is_swanlab=args.swanlab, eval_episode=args.eval_episode)
        final_mean_success = final_evaluation(args, model_def, model_adv_last, device, attack=True, is_swanlab=args.swanlab, eval_episode=args.eval_episode)

        # Save the agent
        eval_env_def.close()
        eval_env_adv.close()
        eval_env_adv_last.close()

        th.save(model_adv_last.policy.state_dict(), os.path.join(eval_adv_log_path, "policy.pth"))
        del model_adv_last
        env_adv_last.close()

        del model_adv
        env_adv.close()

        th.save(model_def.policy.state_dict(), os.path.join(eval_def_log_path, "policy.pth"))

        del model_def
        env_def.close()



        return final_mean_success

def main():
    # get parameters from config.py
    parser = get_config()
    args = parser.parse_args()
    run_training(args) # 直接调用新的训练函数

if __name__ == '__main__':
    main()