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
from callback import CustomEvalCallback_adv, CustomEvalCallback_def
import random
from buffer import DecoupleRolloutBuffer, ReplayBufferDefender, DualReplayBufferDefender
from adversarial_ppo import AdversarialDecouplePPO
from defensive_sac import DefensiveSAC, BaseSAC
from stable_baselines3.common.vec_env import DummyVecEnv, SubprocVecEnv
import os
import glob
from swanlab.integration.sb3 import SwanLabCallback
from policy import IGCARLNet
from stable_baselines3.common.utils import obs_as_tensor
from fgsm import FGSM_v2
from SAC_lag.SAC_Agent_Continuous import SAC_Lag


class SACLagPolicyAdapter(th.nn.Module):
    """让自定义 SAC_Lag 策略在 FGSM、predict 接口下表现得像 SB3 模型。"""

    def __init__(self, agent: SAC_Lag):
        super().__init__()
        self.agent = agent

    def _prepare_obs(self, obs_tensor):
        if not isinstance(obs_tensor, th.Tensor):
            obs_tensor = th.as_tensor(obs_tensor, dtype=th.float32, device=self.agent.device)
        else:
            obs_tensor = obs_tensor.to(self.agent.device)
        if obs_tensor.dim() == 1:
            obs_tensor = obs_tensor.unsqueeze(0)
        return obs_tensor

    def forward(self, obs_tensor, deterministic=True):
        obs_tensor = self._prepare_obs(obs_tensor)
        if deterministic:
            mean, _ = self.agent.policy.forward(obs_tensor)
            action = th.tanh(mean)
        else:
            action, _, _ = self.agent.policy.sample(obs_tensor)
        return action, None


class SACLagSB3Wrapper:
    """
    让 SAC_Lag 看起来像 SB3 模型，提供 predict/device/policy 等属性，
    便于现有对抗训练和 FGSM 攻击逻辑复用。
    """

    def __init__(self, agent: SAC_Lag):
        self.agent = agent
        self.device = agent.device
        self.policy = SACLagPolicyAdapter(agent)

    def predict(self, observation, deterministic=True):
        obs_tensor = th.as_tensor(observation, dtype=th.float32, device=self.device)
        if obs_tensor.dim() == 1:
            obs_tensor = obs_tensor.unsqueeze(0)
        action, _ = self.policy(obs_tensor, deterministic=deterministic)
        action_np = action.squeeze(0).detach().cpu().numpy()
        return action_np, None


def _resolve_sac_lag_checkpoint(args):
    """
    根据 env/algo/seed(/addition_msg) 自动寻找最新的 sac_lag checkpoint。
    """
    base_dir = os.path.join("logs", args.env_name, args.algo, str(args.trained_seed))
    search_dirs = []
    if args.addition_msg:
        search_dirs.append(os.path.join(base_dir, args.addition_msg))
    search_dirs.append(base_dir)

    candidates = []
    for directory in search_dirs:
        if not os.path.isdir(directory):
            continue
        for pattern in ("sac_lag_*.pt", "sac_lag_*.pth"):
            candidates.extend(glob.glob(os.path.join(directory, pattern)))

    if not candidates:
        raise FileNotFoundError(
            f"未在以下目录找到 sac_lag_*.pt/pth: {search_dirs}"
        )

    candidates.sort(key=os.path.getmtime, reverse=True)
    return candidates[0]

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
    mean_attack_success = 0.0

    for _ in range(num_eval_episodes):
        obs, _ = eval_env.reset(options="random")
        episode_reward = 0.0
        for _ in range(args.T_horizon):
            # --- 这部分逻辑完全来自于你的 defense_test.py ---
            obs_tensor = obs_as_tensor(obs, device)

            if attack:
                # 防御者生成原始动作
                with th.no_grad():
                    if isinstance(final_defender, IGCARLNet):
                        actions, std, _action = final_defender(obs_tensor[:-2])
                    else:
                        actions, _ = final_defender.predict(obs_tensor[:-2].cpu(), deterministic=True)
                actions_tensor = th.tensor(actions, device=obs_tensor.device)  # 确保在同一设备上
                obs_tensor[-1] = actions_tensor
                # 攻击者生成攻击动作
                with th.no_grad():
                    adv_actions, _ = final_attacker.predict(obs_tensor.cpu(), deterministic=True)

                adv_action_mask = (adv_actions[0] > 0) & (obs[-2] > 0)
                if args.unlimited_attack:
                    adv_action_mask = np.ones_like(adv_action_mask, dtype=bool)
                # print("DEBUG train.py adv_action_mask:", adv_action_mask, "adv_actions:", adv_actions, "obs[-2]:", obs[-2])

                if adv_action_mask.any():
                    adv_state = FGSM_v2(adv_actions[1], victim_agent=final_defender,
                                        last_state=obs_tensor[:-2].unsqueeze(0), epsilon=args.attack_eps, device=device)
                    if isinstance(final_defender, IGCARLNet):
                        action_perturbed, std, _action = final_defender(adv_state)
                    else:
                        action_perturbed, _ = final_defender.predict(adv_state.cpu(), deterministic=True)
                    final_action = action_perturbed
                    total_attack_times += 1
                    if is_swanlab:
                        swanlab.log({"Eval/agent action before:": actions.item()})
                        swanlab.log({"Eval/agent adv action:": adv_actions[1].item()})
                        swanlab.log({"Eval/agent action after attack:": final_action.item()})

                    print("DEBUG train.py final_evaluation action before attack:", actions, "attack action is: ", adv_actions[1], "after attack:", final_action)
                else:
                    final_action = actions

                # 组合最终动作并与环境交互（确保 numpy 类型）
                if isinstance(final_action, th.Tensor):
                    final_action_np = final_action.detach().cpu().numpy()
                else:
                    final_action_np = np.asarray(final_action)

                adv_mask_np = adv_action_mask
                if isinstance(adv_action_mask, th.Tensor):
                    adv_mask_np = adv_action_mask.detach().cpu().numpy()
                else:
                    adv_mask_np = np.asarray(adv_action_mask)

                action = np.column_stack((final_action_np, adv_mask_np))
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
                    if isinstance(final_defender, IGCARLNet):
                        actions, std, _action = final_defender(obs_tensor)
                    else:
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

    return mean_success, mean_attack_success



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

def create_model_def(args, env, device, best_model_path, expert_path):
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
        expert_path,
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

    hyperparameters_to_track = {
        # --- 布尔类型的开关 (如果为 True, 则只显示名字) ---
        'action_diff': 'action_diff',
        'expert': 'use_expert',
        'lag': 'use_lag',
        'DualBuffer': 'use_DualBuffer',

        # --- 带值的参数 (会显示为 '名字-值' 的格式) ---
        'eps': 'attack_eps',
        'steps': 'train_step',
        'adv_ratio': 'adv_sample_ratio',
        'lag_eps': 'lag_eps',
        'batch': 'batch_size'

    }

    def generate_experiment_name(args, params_map):
        """
        根据args和配置字典，自动生成一个描述性的实验文件夹名。
        """
        name_parts = []
        for name, attr in params_map.items():
            if hasattr(args, attr):
                value = getattr(args, attr)

                # 对布尔类型的开关进行处理
                if isinstance(value, bool):
                    if value:
                        name_parts.append(name)
                # 对其他所有带值的参数进行处理
                else:
                    name_parts.append(f"{name}-{value}")

        if not name_parts:
            return "base_experiment"  # 如果没有任何特殊参数，返回一个默认名

        return "_".join(name_parts)

    # ==============================================================================
    # 2. 调用函数生成实验名，并创建路径
    # ==============================================================================
    experiment_name = generate_experiment_name(args, hyperparameters_to_track)

    print(f"[*] 本次实验标签: {experiment_name}")

    # defenfer and attacker log path
    if not args.adv_test:
        log_path = os.path.join("logs", args.env_name, args.algo, experiment_name, str(args.seed))
        os.makedirs(log_path, exist_ok=True)
        best_model_path_def = os.path.join(log_path, "best_model", "def")
        best_model_path_adv = os.path.join(log_path, "best_model", "adv")

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
    if not args.adv_test:
        model_path_def = os.path.join(log_path, "model", 'def')
        os.makedirs(model_path_def, exist_ok=True)
        checkpoint_callback_def = CheckpointCallback(save_freq=args.save_freq, save_path=model_path_def)

        model_path_adv = os.path.join(log_path, "model", 'adv')
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

    else:
        env_def = DummyVecEnv([make_env(args.seed, 0, False)])
        env_adv = DummyVecEnv([make_env(args.seed, 0, True)])

    # 2025-10-26 wq 统一的回调函数列表初始化
    callbacks_common = []

    if args.swanlab:
        if args.adv_test:
            run_name = f"attacker-only-{args.algo}-{args.seed}-{args.attack_eps}-{args.addition_msg}-{args.train_step}"
        else:
            run_name = f"{experiment_name}_seed-{args.seed}"

        run = swanlab.init(project="RARL", name=run_name, config=args)
        swan_cb = SwanLabCallback(project="RARL", experiment_name=run_name, verbose=2, log_interval=100)
        callbacks_common.append(swan_cb)

    if args.adv_test:
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
        elif args.algo == 'DARRL':
            model_path = IGCARLNet(26, 1)
            score = f"policy2000_actor"
            model_path_drl = os.path.join("logs", args.env_name, args.algo, str(args.trained_seed), score) + '.pth'
            state_dict = th.load(model_path_drl, map_location=device)
            model_path.load_state_dict(state_dict)
            model_path.eval()
            model_path.to(device)  # 再次确保
            trained_agent = model_path
        elif args.algo == "SAC_lag":
            ckpt_path = _resolve_sac_lag_checkpoint(args)
            print("DEBUG def model path: ", ckpt_path)
            lag_agent = SAC_Lag(state_dim=args.state_dim, action_dim=args.action_dim, device=device)
            lag_agent.load_actor(ckpt_path)
            trained_agent = SACLagSB3Wrapper(lag_agent)
            model_path = trained_agent
        elif args.algo == "RARL":
            model_path = os.path.join("logs", args.env_name, args.algo, str(args.trained_seed), "lunar")
            print("DEBUG def model path: ", model_path)
            trained_agent = SAC.load(model_path, device=device)
        elif args.algo == "SAC":
            model_path = os.path.join("logs", args.env_name, args.algo, str(args.trained_seed), "lunar")
            print("DEBUG def model path: ", model_path)
            trained_agent = SAC.load(model_path, device=device)
        elif args.algo == "PPO":
            model_path = os.path.join("logs", args.env_name, args.algo, str(args.trained_seed), "lunar")
            print("DEBUG def model path: ", model_path)
            trained_agent = PPO.load(model_path, device=device)
        elif args.algo == "TD3":
            model_path = os.path.join("logs", args.env_name, args.algo, str(args.trained_seed), "lunar")
            print("DEBUG def model path: ", model_path)
            trained_agent = TD3.load(model_path, device=device)

        adv_log_path = os.path.join("logs", args.env_name, args.algo, str(args.trained_seed), "adv")
        os.makedirs(adv_log_path, exist_ok=True)
        best_model_path_adv = os.path.join(adv_log_path, "best_model")
        model_path_adv = os.path.join(adv_log_path, "model")
        os.makedirs(model_path_adv, exist_ok=True)
        checkpoint_callback_adv = CheckpointCallback(save_freq=args.save_freq, save_path=model_path_adv)

        model_adv = create_model_adv(args, env_adv, device, best_model_path_adv)

        model_adv.learn(total_timesteps=args.train_step * args.n_steps, progress_bar=True,
                        callback=[checkpoint_callback_adv] + callbacks_common, log_interval=args.print_interval,
                        age_model_path = model_path)
        model_adv.save(os.path.join(adv_log_path, "lunar"), exclude=["optimizer", "replay_buffer", "trained_agent"])

        mean_success = final_evaluation(args, trained_agent, model_adv, device, attack=False, is_swanlab=args.swanlab, eval_episode=args.eval_episode)
        final_mean_success = final_evaluation(args, trained_agent, model_adv, device, attack=True, is_swanlab=args.swanlab, eval_episode=args.eval_episode)

        del model_adv
        env_adv.close()
        env_def.close()

    else:
        # 2025-10-02 wq 防御者预训练 (Standard SAC)
        env_def = DummyVecEnv([make_env(args.seed, 0, False)])
        env_adv = DummyVecEnv([make_env(args.seed, 0, True)])

        model_def_pre = BaseSAC("MlpPolicy", env_def, batch_size=args.batch_size,
                                learning_rate=args.lr_def, verbose=1, device=device)
        model_def_pre.learn(
            total_timesteps=args.train_step * args.n_steps,
            progress_bar=True,
            callback=[checkpoint_callback_def] + callbacks_common
        )

        base_def_path = os.path.join(log_path, "0", "def", "lunar")

        model_def_pre.save(base_def_path, exclude=["trained_agent", "trained_adv"])

        # 2025-11-11 wq 攻击者预训练
        model_adv = create_model_adv(args, env_adv, device, best_model_path_adv)
        model_adv.learn(total_timesteps=args.train_step * args.n_steps, progress_bar=True,
                        callback=[checkpoint_callback_adv] + callbacks_common, reset_num_timesteps=False,
                        age_model_path=base_def_path)

        model_adv.save(os.path.join(log_path, "0", "adv", "lunar"), exclude=["trained_agent"])

        mean_success_noAttack, _ = final_evaluation(args, model_def_pre, model_adv, device, attack=False,
                                                    is_swanlab=args.swanlab, eval_episode=args.eval_episode)
        mean_success_attack, attack_success = final_evaluation(args, model_def_pre, model_adv, device, attack=True,
                                                               is_swanlab=args.swanlab, eval_episode=args.eval_episode)
        del model_def_pre
        expert_path = os.path.join("logs", args.env_name, args.algo, "expert", "lunar")

        model_def = create_model_def(args, env_def, device, best_model_path_def, expert_path)

        for i in range(args.loop_nums):
            env_def.reset()
            env_adv.reset()
            print(f"\n{'=' * 15} Loop {i + 1}/{args.loop_nums} {'=' * 15}")
            # 2025-11-11 wq 防御者训练

            base_adv_path = os.path.join(log_path, str(i), "adv", "lunar")
            base_def_path = os.path.join(log_path, str(i), "def", "lunar")

            model_def.learn(total_timesteps=args.train_step * args.n_steps, progress_bar=True,
                            callback=[checkpoint_callback_def] + callbacks_common, reset_num_timesteps=False,
                            trained_age_path = base_def_path, adv_path = base_adv_path)

            def_path = os.path.join(log_path, str(i + 1), "def", "lunar")
            model_def.save(def_path, exclude=["trained_agent", "trained_adv", "trained_expert"])

            # 2025-11-11 wq 攻击者训练

            model_adv.learn(total_timesteps=args.train_step * args.n_steps, progress_bar=True,
                            callback=[checkpoint_callback_adv] + callbacks_common, reset_num_timesteps=False,
                            age_model_path = def_path)
            adv_path = os.path.join(log_path, str(i + 1), "adv", "lunar")
            model_adv.save(adv_path, exclude=["trained_agent"])


            mean_success_noAttack, _ = final_evaluation(args, model_def, model_adv, device, attack=False,
                                                     is_swanlab=args.swanlab, eval_episode=args.eval_episode)
            mean_success_attack, attack_success = final_evaluation(args, model_def, model_adv, device, attack=True,
                                                   is_swanlab=args.swanlab, eval_episode=args.eval_episode)


        # return mean_success_attack

def main():
    # get parameters from config.py
    parser = get_config()
    args = parser.parse_args()
    run_training(args) # 直接调用新的训练函数

if __name__ == '__main__':
    main()