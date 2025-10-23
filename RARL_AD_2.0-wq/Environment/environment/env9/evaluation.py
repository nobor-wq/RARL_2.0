import gymnasium as gym
import numpy as np
from stable_baselines3 import SAC, PPO, TD3  # 或者您使用的其他算法
from stable_baselines3.common import policies
from gymnasium.wrappers import TimeLimit
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.utils import obs_as_tensor
from config import get_config
import Environment.environment
import os
import torch as th
from perturbation import *
import pandas as pd
from utils import get_attack_prob, get_trained_agent
from PIL import Image


# get parameters from config.py
parser = get_config()
args = parser.parse_args()

# 设置随机种子
np.random.seed(args.seed)  # 设置 NumPy 随机种子
th.manual_seed(args.seed)  # 设置 CPU 随机种子
if th.cuda.is_available():
    th.cuda.manual_seed(args.seed)  # 设置 CUDA 随机种子
    th.cuda.manual_seed_all(args.seed)  # 设置所有 GPU 随机种子
th.backends.cudnn.deterministic = True  # 确保 CUDA 的确定性
th.backends.cudnn.benchmark = False  # 禁用 CuDNN 自动优化

# 设置设备
if args.use_cuda and th.cuda.is_available():
    device = th.device(f"cuda:{args.cuda_number}")
else:
    device = th.device("cpu")

# 设置eval标志
args.eval = True
# 创建环境
env = gym.make(args.env_name, attack=args.attack, adv_steps=args.adv_steps, eval=args.eval, use_gui=True, render_mode='rgb_array')
env.unwrapped.start()

if args.attack:
    if args.best_model:
        advmodel_path = "./logs/adv_eval/" + os.path.join(args.adv_algo, args.env_name, args.algo, args.addition_msg,
                                                          'best_model')
    else:
        # advmodel_path = "./logs/adv_eval/" + os.path.join(args.adv_algo, args.env_name, args.algo, args.addition_msg, 'lunar')
        # advmodel_path = "./logs/adv_eval/" + os.path.join(args.adv_algo, args.env_name, args.algo, args.addition_msg,
        #                                                   'lunar')
        advmodel_path = "./logs/adv_eval/" + os.path.join(args.adv_algo, args.env_name, args.algo, args.addition_msg,
                                                          'model/rl_model_100000_steps')
    # 加载训练好的攻击者模型
    if args.adv_algo == 'SAC':
        model = SAC.load(advmodel_path, device=device)
    elif args.adv_algo == 'PPO':
        # advmodel_path = "./logs/adv_eval/TrafficEnv2-v0/SAC/std/lunar"
        model = PPO.load(advmodel_path, device=device)
    else:
        # advmodel_path = "./logs/adv_eval/TrafficEnv2-v0/SAC/std/lunar"
        model = PPO.load(advmodel_path, device=device)

# 加载训练好的自动驾驶模型
trained_agent = get_trained_agent(args, device)

# 进行验证
rewards = []
ep_steps = []

maxSpeed = 15.0
ct = 0
sn = 0
sat = 0
speed_list = []
attack_count_list = []
mean_attack_reward_list = []
log_list = []
last_act_list = []
for episode in range(args.train_step):
    obs, info = env.reset()
    img = env.render()
    speed = 0
    episode_reward = 0
    episode_steps = 0
    save_dir = f'./render/{episode}'
    # 创建目录（如果不存在的话）
    os.makedirs(save_dir, exist_ok=True)
    for steps in range(args.T_horizon):
        obs_tensor = obs_as_tensor(obs, device)
        if args.attack:
            speed_list.append(obs[-4])
            if args.algo in ('FNI', 'DARRL'):
                actions, std, _action = trained_agent(obs_tensor[:-2])
                actions = actions.detach().cpu().numpy()
            elif args.algo == 'IL':
                actions, _, _ = trained_agent(obs_tensor[:-2])
                actions = actions.detach().cpu().numpy()
            else:
                actions, _ = trained_agent.predict(obs_tensor[:-2].cpu(), deterministic=True)
            actions_tensor = th.tensor(actions, device=obs_tensor.device)  # 确保在同一设备上
            obs_tensor[-1] = actions_tensor
            adv_actions, _ = model.predict(obs_tensor.cpu(), deterministic=True)

            # print('adv_state',adv_state)
            # print(episode_steps, 'attack', 'Victim action is', actions, 'adv actions is', adv_actions, 'obs ', obs_tensor[:-2])
            act_list = env.unwrapped.get_act()
            # alpha = 4
            # beta = 1.5
            #
            # top_k = 6
            # k = min(top_k, len(act_list))
            # smallest_rates = np.partition(act_list, k - 1)[:k]
            # print('smallest_rates', smallest_rates)
            # # individual_probs = 1 / (1 + np.exp(alpha * (smallest_rates - beta)))
            # # attack_prob = np.mean(individual_probs)
            # # attack_prob = np.clip(attack_prob, 0, 1)
            # attack_prob = 1 - np.prod([1 - 1 / (1 + np.exp(alpha * (act - beta))) for act in act_list])
            # act_list = env.get_act()
            # print('act list ', act_list)
            # lens = min(len(act_list), len(last_act_list))
            # for i in range(lens):
            #     if act_list[i] > last_act_list[i]:
            #         act_list[i] = 15
            # last_act_list = act_list
            attack_prob = get_attack_prob(act_list)
            # print('attack prob is ', attack_prob)

            adv_action_mask = (adv_actions[0] > 0) & (obs[-2] > 0)
            adv_flag = 1 if adv_actions[0] > 0 else 0
            if adv_flag:
                print('steps ', episode_steps, 'pre actions', actions, 'adv_actions', adv_actions, 'attack prob ', attack_prob)
            log_list.append([args.algo, args.attack_method, args.adv_steps, adv_flag, steps, attack_prob])

            # print(adv_action_MAD,actions)
            if adv_action_mask or args.unlimited_attack:
                if args.attack_method == 'fgsm':
                    adv_state = FGSM_v2(adv_actions[1], victim_agent=trained_agent, last_state=obs_tensor[:-2],
                                        device=device)
                elif args.attack_method == 'pgd':
                    adv_state = PGD(adv_actions[1], trained_agent, obs_tensor[:-2], device=device)

                if args.attack_method == 'direct':
                    action = adv_actions[1]
                else:
                    if args.algo in ('FNI', 'DARRL'):
                        adv_action_fromState, _, _ = trained_agent(adv_state)
                        action = adv_action_fromState.detach().cpu().numpy()
                    elif args.algo == 'IL':
                        adv_action_fromState, _, _ = trained_agent(adv_state)
                        action = adv_action_fromState.detach().cpu().numpy()
                    else:
                        adv_action_fromState, _ = trained_agent.predict(adv_state.cpu(), deterministic=True)
                        print(episode_steps, 'attack', '{} action is'.format(args.attack_method), adv_action_fromState)
                        action = adv_action_fromState
            else:
                action = actions
            # action = adv_action_FGSM[0]
            action = np.column_stack((action, adv_action_mask))
            obs, reward, done, terminate, info = env.step(action[0])
        else:
            speed_list.append(obs[-2])
            # actions = trained_agent.policy(obs_tensor.unsqueeze(0))
            # actions1 = trained_agent.policy(obs_tensor.unsqueeze(0), deterministic=True)
            if args.algo in ('FNI', 'DARRL'):
                actions, std, _action = trained_agent(obs_tensor)
                actions = actions.cpu().detach().numpy()
            elif args.algo == 'IL':
                # actions, _, _ = trained_agent(obs_tensor)
                # actions = actions.cpu().detach().numpy()
                actions, _ = trained_agent.predict(obs, deterministic=True)
            else:
                actions, _ = trained_agent.predict(obs, deterministic=True)
            # actions3,_ = trained_agent.predict(obs)
            # print(actions,actions1,actions2,actions3)
            obs, reward, done, terminate, info = env.step(actions)

        print('steps ', episode_steps, 'obs is ', obs, 'actions are ', actions, 'reward is ', reward, 'done is ', done)
        img = env.render()
        img = Image.fromarray(img)
        img.save(f'{save_dir}/{steps}.jpg')

        episode_reward += reward
        episode_steps += 1
        if done:
            if round(args.adv_steps - obs[-2] * args.adv_steps) != 0:
                mean_attack_reward_list.append(1 / round(args.adv_steps - obs[-2] * args.adv_steps))
            ct += 1
            if args.attack:
                if round(args.adv_steps - obs[-2] * args.adv_steps):
                    sat += 1
            break
    xa = info['x_position']
    ya = info['y_position']
    if args.unlimited_attack:
        attack_count_list.append(episode_steps)
    else:
        attack_count_list.append(round(args.adv_steps - obs[-2] * args.adv_steps))
    if args.env_name == 'TrafficEnv1-v0' or args.env_name == 'TrafficEnv3-v0' or args.env_name == 'TrafficEnv6-v0':
        if xa < -50.0 and ya > 4.0 and done is False:
            sn += 1
    elif args.env_name == 'TrafficEnv2-v0':
        if xa > 50.0 and ya > -5.0 and done is False:
            sn += 1
    elif args.env_name == 'TrafficEnv4-v0':
        if ya < -50.0 and done is False:
            sn += 1
    rewards.append(episode_reward)
    ep_steps.append(episode_steps)

log_df = pd.DataFrame(log_list, columns=['algo', 'attack_method', 'adv_steps', 'adv_flag', 'steps', 'attack_prob'])
file_name = 'log_new.csv'
# 判断文件是否存在
if not os.path.exists(file_name):
    # 如果文件不存在，写入数据和列名
    log_df.to_csv(file_name, mode='w', header=True, index=False)
else:
    # 如果文件存在，只追加数据，不写入列名
    log_df.to_csv(file_name, mode='a', header=False, index=False)

# 计算平均奖励和步数
mean_reward = np.mean(rewards)
std_reward = np.std(rewards)
mean_steps = np.mean(steps)
std_steps = np.std(steps)

# 计算碰撞率
cr = ct / args.train_step * 100
sr = sn / args.train_step * 100
if args.attack:
    if sum(1 for x in attack_count_list if x > 0) > 0:
        asr = sat / sum(1 for x in attack_count_list if x > 0) * 100
    else:
        asr = 0.00
else:
    asr = 0.00

# 计算平均速度
mean_speed = np.mean(speed_list)
std_speed = np.std(speed_list)

# 计算平均攻击次数
attack_list = [x for x in attack_count_list if x != 0]
mean_attack_times = np.mean(attack_list)
std_attack_times = np.std(attack_list)

# 计算单次攻击的收益
mean_attack_reward = np.mean(mean_attack_reward_list)
std_attack_reward = np.std(mean_attack_reward_list)

print('attack lists ', attack_count_list, 'attack times ', len(attack_list))
print(f"Mean reward: {mean_reward:.2f} +/- {std_reward:.2f}")
print(f"Mean steps: {mean_steps:.2f} +/- {std_steps:.2f}")
print(f"Mean speed: {mean_speed * maxSpeed:.2f} +/- {std_speed * maxSpeed:.2f}")
print(f"Mean attack times: {mean_attack_times:.2f} +/- {std_attack_times:.2f}")
print(f"Collision rate: {cr:.2f}")
print(f"Success rate: {sr:.2f}")
print(f"Success attack rate: {asr:.2f}")
print(f"Reward per attack: {mean_attack_reward:.2f} +/- {std_attack_reward:.2f}")

# 定义日志文件路径
log_file = "eval_attack_log.txt"

# 将参数和结果写入日志文件
with open(log_file, 'a') as f:  # 使用 'a' 模式以追加方式写入文件
    # 写入参数
    f.write("Parameters:\n")
    for arg in vars(args):  # 遍历 args 中的所有参数
        f.write(f"{arg}: {getattr(args, arg)}\n")

    # 写入结果
    f.write("\nResults:\n")
    f.write(f"Mean reward: {mean_reward:.2f} +/- {std_reward:.2f}\n")
    f.write(f"Mean steps: {mean_steps:.2f} +/- {std_steps:.2f}\n")
    f.write(f"Mean speed: {mean_speed * maxSpeed:.2f} +/- {std_speed * maxSpeed:.2f}\n")
    f.write(f"Mean attack times: {mean_attack_times:.2f} +/- {std_attack_times:.2f}\n")
    f.write(f"Collision rate: {cr:.2f}\n")
    f.write(f"Success rate: {sr:.2f}\n")
    f.write(f"Success attack rate: {asr:.2f}\n")
    f.write(f"Reward per attack: {mean_attack_reward:.2f} +/- {std_attack_reward:.2f}\n")
    f.write("-" * 50 + "\n")
