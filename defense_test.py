import gymnasium as gym
import numpy as np
from stable_baselines3 import SAC, PPO, TD3  # 或者您使用的其他算法
from stable_baselines3.common.utils import obs_as_tensor
from config import get_config
import Environment.environment
import os
import torch as th
from fgsm import FGSM_v2
from policy import  IGCARLNet, FniNet
import glob
from SAC_lag.SAC_Agent_Continuous import SAC_Lag

# --- [新增 2] 复制 train.py 中的 Wrapper 类 ---
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

def _resolve_sac_lag_checkpoint(args, addition_msg):
    """
    根据 env/algo/seed(/addition_msg) 自动寻找最新的 sac_lag checkpoint。
    注意：这里稍微修改了入参，直接传入 addition_msg 变量
    """
    base_dir = os.path.join("logs", args.env_name, args.algo, str(args.trained_seed))
    search_dirs = []
    # defense_test.py 中 addition_msg 是局部变量，可能不为空
    if addition_msg and addition_msg != "base":
        search_dirs.append(os.path.join(base_dir, addition_msg))
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
temp_def_env = gym.make(args.env_name, attack=False) # attack=False 是关键！
env.unwrapped.start()

msg_parts = []
if args.action_diff:
    msg_parts.append("action_diff")
    if args.use_expert:
        msg_parts.append("expert")
    # 依赖于 expert 的技术，可以进行嵌套
    if args.use_kl:
        msg_parts.append("kl")
    elif args.use_lag:
        msg_parts.append("lag")
# 新增的 Buffer 技术
if args.use_DualBuffer:
    msg_parts.append("DualBuffer")
if not msg_parts:
    # 如果没有任何特殊技术，就是一个基础版本
    addition_msg = "base"
else:
    addition_msg = "_".join(msg_parts)

env_name = "TrafficEnv3-v0"

if args.attack:
    # 加载训练好的攻击者模型
    # adv_path = "./logs/eval_adv/" + os.path.join(args.algo_adv, args.env_name, args.algo, args.model_name, 'policy.pth')
    adv_path =  os.path.join("logs", env_name, args.algo, str(args.trained_seed), "adv", "lunar")
    print("DEBUG adv model path: ", adv_path)
    model = PPO.load(adv_path, device=device)


# 加载训练好的自动驾驶模型

if args.algo == "IGCARL":
    model_path_drl = os.path.join("logs", env_name, args.algo, "defender_v265.pth")
    if not os.path.isfile(model_path_drl):
        raise FileNotFoundError(f"找不到模型文件：{model_path_drl}")
    trained_agent = IGCARLNet(state_dim=26, action_dim=1).to(device)
    trained_agent.load_state_dict(th.load(model_path_drl, map_location=device))
    trained_agent.eval()
elif args.algo == "RARL":
    defense_model_path = os.path.join("logs", env_name, args.algo, str(args.trained_seed), "lunar")
    print("DEBUG def model path: ", defense_model_path)
    trained_agent  = SAC.load(defense_model_path, device=device)
elif args.algo == "SAC":
    sac_model_path = os.path.join("logs", env_name, args.algo, str(args.trained_seed), "lunar")
    print("DEBUG defense_test.py sac_model_path: ", sac_model_path)
    trained_agent = SAC.load(sac_model_path, device=device)
elif args.algo == "PPO":
    ppo_model_path = os.path.join("logs", env_name, args.algo, str(args.trained_seed), "lunar")
    print("DEBUG defense_test.py ppo_model_path: ", ppo_model_path)
    trained_agent = PPO.load(ppo_model_path, device=device)
elif args.algo == "TD3":
    td3_model_path = os.path.join("logs", env_name, args.algo, str(args.trained_seed), "lunar")
    print("DEBUG defense_test.py td3_model_path: ", td3_model_path)
    trained_agent = TD3.load(td3_model_path, device=device)
elif args.algo == 'DARRL':
    trained_agent = IGCARLNet(26, 1)
    score = f"policy2000_actor"
    model_path_drl = os.path.join("logs", env_name, args.algo, str(args.trained_seed), score) + '.pth'
    state_dict = th.load(model_path_drl, map_location=device)
    trained_agent.load_state_dict(state_dict)
    trained_agent.eval()
    trained_agent.to(device)  # 再次确保
elif args.algo == 'FNI':
    model_path = FniNet(args.state_dim, args.action_dim).to(device)
    if args.fni_model_path:
        ckpt_path = args.fni_model_path
        if not ckpt_path.endswith(".pth"):
            ckpt_path = f"{ckpt_path}.pth"
        if not os.path.isabs(ckpt_path):
            ckpt_path = os.path.join("logs", env_name, args.algo, ckpt_path)
    else:
        ckpt_path = os.path.join("logs", env_name, args.algo, "policy_v411.pth")
    ckpt_path = os.path.expanduser(ckpt_path)
    if not os.path.isfile(ckpt_path):
        raise FileNotFoundError(f"找不到模型文件：{ckpt_path}")
    state_dict = th.load(ckpt_path, map_location=device)
    model_path.load_state_dict(state_dict)
    model_path.eval()
    model_path.to(device)  # 再次确保
    trained_agent = model_path
# --- [新增 3] 添加 SAC_lag 加载分支 ---
elif args.algo == "SAC_lag":
    # 使用之前定义的 helper 函数查找模型路径
    # ckpt_path = _resolve_sac_lag_checkpoint(args, addition_msg)
    base_dir = os.path.join("logs", env_name, args.algo, str(args.trained_seed), "sac_lag_2000.pt")
    print("DEBUG def model path: ", base_dir)

    # 初始化 Agent 并加载权重
    lag_agent = SAC_Lag(state_dim=args.state_dim, action_dim=args.action_dim, device=device)
    lag_agent.load_actor(base_dir)

    # 使用 Wrapper 包装，使其具备 .predict 方法
    trained_agent = SACLagSB3Wrapper(lag_agent)

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
            if args.unlimited_attack:
                adv_action_mask = np.ones_like(adv_action_mask, dtype=bool)
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
            episode_reward += r_def - c_def
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
            episode_reward += reward
            print('No attack step ', episode_steps, 'reward is ', reward)

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
    if args.env_name == 'TrafficEnv1-v0' or args.env_name == 'TrafficEnv3-v0' or args.env_name == 'TrafficEnv3-v1' or args.env_name == 'TrafficEnv3-v2':
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
temp_def_env.close()
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
    total_attacks = sum(attack_count_list)
    if total_attacks == 0:
        mean_attack_times = 0
        std_attack_times = 0
        asr = 0
    else:
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
log_file = "eval_attack_log_101.txt"

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


