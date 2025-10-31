import gymnasium as gym
import numpy as np
from stable_baselines3 import SAC, PPO, TD3  # 或者您使用的其他算法
from gymnasium.wrappers import TimeLimit
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.utils import obs_as_tensor
from config import get_config
import Environment.environment
import os
import torch as th
# from perturbation import *
from fgsm import FGSM_v2
from PIL import Image
from policy import FniNet, IGCARLNet


# get parameters from config.py
parser = get_config()
args = parser.parse_args()

# 设置随机种子
np.random.seed(args.seed + 100)  # 设置 NumPy 随机种子
th.manual_seed(args.seed + 100)  # 设置 CPU 随机种子
if th.cuda.is_available():
    th.cuda.manual_seed(args.seed + 100)  # 设置 CUDA 随机种子
    th.cuda.manual_seed_all(args.seed + 100)  # 设置所有 GPU 随机种子
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
env = gym.make(args.env_name, attack=args.attack, adv_steps=args.adv_steps, eval=args.eval, use_gui=args.use_gui,
               render_mode=args.render_mode)
env.unwrapped.start()

if args.attack:
    if args.best_model:
        if args.eval_best_model:
            advmodel_path = "./logs/eval_adv/" + os.path.join(args.algo_adv, args.env_name, args.algo, str(args.train_eps) , str(args.seed), str(args.trained_step), 'eval_best_model/best_model')
        else:
            advmodel_path = "./logs/eval_adv/" + os.path.join(args.algo_adv, args.env_name, args.algo,str(args.train_eps) , str(args.seed), str(args.trained_step), 'best_model')
    else:#str(args.train_eps)
        advmodel_path = "./logs/eval_adv/" + os.path.join(args.algo_adv, args.env_name, args.algo, str(args.train_eps) , str(args.seed), str(args.trained_step), 'lunar')
    # 加载训练好的攻击者模型
    print("DEBUG adv model path: ", advmodel_path)

    if args.algo_adv == 'SAC':
        model = SAC.load(advmodel_path, device=device)
    elif args.algo_adv == 'PPO':
        #advmodel_path = "./logs/adv_eval/TrafficEnv2-v0/SAC/std/lunar"
        model = PPO.load(advmodel_path, device=device)
    else:
        #advmodel_path = "./logs/adv_eval/TrafficEnv2-v0/SAC/std/lunar"
        model = PPO.load(advmodel_path, device=device)


# 加载训练好的自动驾驶模型



# if args.base:#str(args.seed)
#     defense_model_path = "./logs/eval_def/" + os.path.join("base", args.algo, args.env_name, args.addition_msg, "1")
# else:
#     defense_model_path = "./logs/eval_def/" + os.path.join(args.algo, args.env_name, args.addition_msg,  str(args.train_eps), str(args.seed))
#
# # defense_model_path = os.path.join(args.path, args.env_name, args.algo)
# if args.best_model:
#     model_path = os.path.join(defense_model_path, 'best_model.zip')
#     # model_path = os.path.join(defense_model_path, 'best_model/best_model')
#     #model_path = os.path.join(args.path, args.env_name, args.algo,  'best_model/best_model')
# else:
#     #model_path = os.path.join(args.path, args.env_name, args.algo, args.addition_msg, 'lunar')
#     model_path = os.path.join(defense_model_path, 'lunar')
# trained_agent = SAC.load(model_path, device=device)

if args.algo == "IGCARL":
    prefix = "./logs/eval_def/" + os.path.join(args.algo, args.env_name)
    filename = f'{args.model_name}.pth'
    model_path_drl = os.path.join(prefix, filename)
    if not os.path.isfile(model_path_drl):
        raise FileNotFoundError(f"找不到模型文件：{model_path_drl}")
    trained_agent = IGCARLNet(state_dim=26, action_dim=1).to(device)
    trained_agent.load_state_dict(th.load(model_path_drl, map_location=device))
    trained_agent.eval()
elif args.algo == "RARL":
    defense_model_path = "./logs/eval_def/" + os.path.join(args.algo, args.env_name,  str(args.train_eps), str(args.seed), str(args.trained_step))
    if args.best_model:
        model_path = os.path.join(defense_model_path, 'best_model')
    elif args.eval_best_model:
        model_path = os.path.join(defense_model_path, 'eval_best_model', 'best_model')
    else:
        model_path = os.path.join(defense_model_path, 'lunar')

    print("DEBUG def model path: ", model_path)
    trained_agent = SAC.load(model_path, device=device)




# 进行验证
rewards = []
steps = []

maxSpeed = 15.0
ct = 0
sn = 0
success_attack_count = 0
speed_list = []
attack_count_list = []
for episode in range(args.train_step):
    obs, info = env.reset(options="random")
    # img = env.render()
    speed = 0
    episode_reward = 0
    episode_steps = 0
    attack_count = 0
    # save_dir = f'./render/{args.env_name}/{args.adv_algo}/{episode}'
    # # # 创建目录（如果不存在的话）
    # os.makedirs(save_dir, exist_ok=True)
    for _ in range(args.T_horizon):
        obs_tensor = obs_as_tensor(obs, device)
        if args.attack:
            speed_list.append(obs[-4])
            if args.algo in ('FNI', 'DARRL', 'IGCARL'):
                actions, std, _action = trained_agent(obs_tensor[:-2])
                actions = actions.detach().cpu().numpy()
            else:
                actions, _ = trained_agent.predict(obs_tensor[:-2].cpu(), deterministic=True)
                # actions, _ = trained_agent.predict(obs_tensor[:-2].cpu(), deterministic=True)
            actions_tensor = th.tensor(actions, device=obs_tensor.device)  # 确保在同一设备上
            obs_tensor[-1] = actions_tensor

            adv_actions, _ = model.predict(obs_tensor.cpu(), deterministic=True)

            adv_action_mask = (adv_actions[0] > 0) & (obs[-2] > 0)
            #print(adv_action_MAD,actions)
            if adv_action_mask or args.unlimited_attack:
                if args.attack_method == 'fgsm':
                    adv_state = FGSM_v2(adv_actions[1], victim_agent=trained_agent, last_state=obs_tensor[:-2].unsqueeze(0),
                                        epsilon=args.attack_eps, device=device)
                # elif args.attack_method == 'pgd':
                #     adv_state = PGD(adv_actions[1], trained_agent, obs_tensor[:-2].unsqueeze(0), device=device)

                if args.attack_method == 'direct':
                    action = adv_actions[1]
                else:
                    if args.algo in ('FNI', 'DARRL', 'IGCARL'):
                        adv_action_fromState, _, _ = trained_agent(adv_state)
                        action = adv_action_fromState.detach().cpu().numpy()
                    else:
                        adv_action_fromState, _ = trained_agent.predict(adv_state.cpu(), deterministic=True)
                        action = adv_action_fromState
                    print("DEBUG action before attack: ", actions, " adv action: ", adv_actions[1],
                          "action after attack: ", adv_action_fromState)
                attack_count += 1
            else:
                print("DEBUG adv_action_mask: ", adv_action_mask)
                action = actions
            #action = adv_action_FGSM[0]
            action = np.column_stack(( action, adv_action_mask))
            obs, reward, done, terminate, info = env.step(action)
            # 2025-10-30 wq 攻击并且成功碰撞
            if done and adv_action_mask:
                success_attack_count += 1

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
            print('step ', episode_steps, 'reward is ', r_def-c_def)


        else:
            speed_list.append(obs[-2])
            #actions = trained_agent.policy(obs_tensor.unsqueeze(0))
            #actions1 = trained_agent.policy(obs_tensor.unsqueeze(0), deterministic=True)
            if args.algo in ('FNI', 'DARRL', 'IGCARL'):
                actions, std, _action = trained_agent(obs_tensor)
                actions = actions.cpu().detach().numpy()
            else:
                actions,_ = trained_agent.predict(obs, deterministic=True)
            obs, reward, done, terminate, info = env.step(actions)

            print('step ', episode_steps, 'reward is ', reward)
        episode_reward += reward
        episode_steps += 1


        if done:
            ct += 1
            break
        # if args.use_gui:
        #     img = env.render()
        #     img = Image.fromarray(img)
        #     img.save(f'{save_dir}/{episode_steps}.jpg')

    attack_count_list.append(attack_count)
    # 如果需要，转成GIF
    # if args.to_gif:
    #     gif_generate(save_dir, args.duration)

    xa = info['x_position']
    ya = info['y_position']
    if args.attack:
        if args.unlimited_attack:
            attack_count_list.append(episode_steps)
    if args.env_name == 'TrafficEnv1-v0' or args.env_name == 'TrafficEnv3-v0' or args.env_name == 'TrafficEnv6-v0':
        if xa < -50.0 and ya > 4.0 and done is False:
            sn += 1
    elif args.env_name == 'TrafficEnv2-v0':
        if xa > 50.0 and ya > -5.0 and done is False:
            sn += 1
    elif args.env_name == 'TrafficEnv4-v0':
        if ya < -50.0 and done is False:
            sn += 1
    elif args.env_name == 'TrafficEnv7-v0':
        if done is False:
            sn += 1
    elif args.env_name == 'TrafficEnv8-v0':
        if ya == 10.0 and done is False:
            sn += 1
    rewards.append(episode_reward)
    steps.append(episode_steps)

env.close()

# 计算平均奖励和步数
mean_reward = np.mean(rewards)
std_reward = np.std(rewards)
mean_steps = np.mean(steps)
std_steps = np.std(steps)

# 计算碰撞率
cr = ct / args.train_step * 100
sr = sn / args.train_step * 100

# 计算平均速度
mean_speed = np.mean(speed_list)
std_speed = np.std(speed_list)


print(f"Mean reward: {mean_reward:.2f} +/- {std_reward:.2f}")
print(f"Mean steps: {mean_steps:.2f} +/- {std_steps:.2f}")
print(f"Mean speed: {mean_speed * maxSpeed:.2f} +/- {std_speed * maxSpeed:.2f}")
print(f"Collision rate: {cr:.2f}")
print(f"Success rate: {sr:.2f}")

if args.attack:
    # 计算平均攻击次数
    mean_attack_times = np.mean(attack_count_list)
    std_attack_times = np.std(attack_count_list)
    # 2025-10-30 wq 计算攻击后成功碰撞的概率
    asr = success_attack_count / sum(attack_count_list) * 100
    ep_asr = success_attack_count / args.train_step * 100

    print('attack lists ', attack_count_list, 'attack times ', len(attack_count_list))
    print(f"Mean attack times: {mean_attack_times:.2f} +/- {std_attack_times:.2f}")
    print(f"Success attack rate: {asr:.2f}")
    print(f"episode Success attack rate: {ep_asr:.2f}")


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
    f.write(f"Collision rate: {cr:.2f}\n")
    f.write(f"Success rate: {sr:.2f}\n")
    if args.attack:
        f.write(f"Mean attack times: {mean_attack_times:.2f} +/- {std_attack_times:.2f}\n")
        f.write(f"Success attack rate: {asr:.2f}\n")
        f.write(f"Episode Success attack rate: {ep_asr:.2f}\n")

    f.write("-" * 50 + "\n")


