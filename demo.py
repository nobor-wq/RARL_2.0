import os
import time
import gym
from gym.wrappers import TimeLimit, Monitor
from stable_baselines3 import PPO, SAC
from stable_baselines3.common.vec_env import DummyVecEnv, SubprocVecEnv
from stable_baselines3.common.callbacks import CheckpointCallback
from stable_baselines3.common.utils import set_random_seed

# --------- PolicyWrapper -----------
class PolicyWrapper:
    """
    给环境提供统一的 agent 接口：agent.predict(obs, deterministic=True)
    我们会把 snapshot 的模型（只用于推理）或正在训练的模型都包装成这个接口注入 env
    """
    def __init__(self, model):
        self.model = model

    def predict(self, obs, deterministic=True):
        return self.model.predict(obs, deterministic=deterministic)

# --------- 快速 snapshot/load 帮手 ----------
def snapshot_model(model, path):
    """
    将 sb3 模型保存到 path（zip 文件路径）。
    """
    dirname = os.path.dirname(path)
    os.makedirs(dirname, exist_ok=True)
    model.save(path)

def load_model_copy(model_class, path, device="cpu"):
    """
    从 path 载入模型的一个独立拷贝（用于固定对手、仅用于predict）。
    model_class: 模型类，例如 PPO 或 SAC
    注意：这里不传 env，若你使用 VecNormalize 或有 stateful obs normalization,
    需要传入相同 env 或另外处理归一化。
    """
    # load 不指定 env（仅用于推理），如果需要可传 env=...
    return model_class.load(path, device=device)

# ---------- 你的模型创建函数（按需替换） ----------
def create_attacker_model(args, env, rollout_buffer_class, device, best_model_path=None):
    # 示例：PPO
    model = PPO("MlpPolicy", env, verbose=1, device=device)
    if best_model_path and os.path.exists(best_model_path):
        model = PPO.load(best_model_path, env=env, device=device)
    return model

def create_defender_model(args, env, rollout_buffer_class, device, best_model_path=None):
    # 示例：SAC
    model = SAC("MlpPolicy", env, verbose=1, device=device)
    if best_model_path and os.path.exists(best_model_path):
        model = SAC.load(best_model_path, env=env, device=device)
    return model

# ---------- 注入到 VecEnv 的方法（支持 SubprocVecEnv/DummyVecEnv） ----------
def set_attacker_in_vecenv(vec_env, attacker_wrapper):
    try:
        vec_env.env_method("set_attacker", attacker_wrapper)
    except Exception as e:
        # 兜底：直接访问 envs 列表
        for e in getattr(vec_env, "envs", []):
            if hasattr(e, "set_attacker"):
                e.set_attacker(attacker_wrapper)

def set_defender_in_vecenv(vec_env, defender_wrapper):
    try:
        vec_env.env_method("set_defender", defender_wrapper)
    except Exception as e:
        for e in getattr(vec_env, "envs", []):
            if hasattr(e, "set_defender"):
                e.set_defender(defender_wrapper)

# ---------- 主训练函数（交替迭代，固定对手为上一轮模型） ----------
def train_rarl_prev_opponent(args):
    eval_log_path = args.eval_log_path
    model_path = os.path.join(eval_log_path, 'model')
    os.makedirs(model_path, exist_ok=True)

    # checkpoint 回调（可分别为 attacker/defender）
    checkpoint_callback_attacker = CheckpointCallback(save_freq=args.save_freq, save_path=os.path.join(model_path, "attacker"))
    checkpoint_callback_defender = CheckpointCallback(save_freq=args.save_freq, save_path=os.path.join(model_path, "defender"))

    # rollout buffer mapping（若无则删除或替换）
    rollout_buffer_map = {
        (True, True): DecouplePaddingRolloutBuffer,
        (True, False): PaddingRolloutBuffer,
        (False, True): DecoupleRolloutBuffer,
        (False, False): RolloutBuffer
    }
    rollout_buffer_class = rollout_buffer_map[(args.padding, args.decouple)]

    # env 构造（保持你的原始逻辑）
    def make_env(seed, rank):
        def _init():
            env = gym.make(args.env_name, attack=args.attack, adv_steps=args.adv_steps)
            env = TimeLimit(env, max_episode_steps=args.T_horizon)
            env = Monitor(env)
            env.unwrapped.start()
            env.reset(seed=seed + rank)
            return env
        return _init

    num_envs = args.num_envs if hasattr(args, 'num_envs') else 1
    if num_envs > 1:
        env = SubprocVecEnv([make_env(args.seed, i) for i in range(num_envs)])
    else:
        env = DummyVecEnv([make_env(args.seed, 0)])
    eval_env = DummyVecEnv([make_env(args.seed + 1000, 0)])

    set_random_seed(args.seed)
    device = args.device if hasattr(args, "device") else "cpu"

    # ---------- 初始化 attacker^0 和 defender^0 ----------
    attacker = create_attacker_model(args, env, rollout_buffer_class, device, best_model_path=getattr(args, "attacker_init_path", None))
    defender = create_defender_model(args, env, rollout_buffer_class, device, best_model_path=getattr(args, "defender_init_path", None))

    # 保存初始快照（作为第0轮的上一轮模型）
    timestamp = int(time.time())
    attacker_prev_path = os.path.join(model_path, "snapshots", f"attacker_iter_0_{timestamp}.zip")
    defender_prev_path = os.path.join(model_path, "snapshots", f"defender_iter_0_{timestamp}.zip")
    snapshot_model(attacker, attacker_prev_path)
    snapshot_model(defender, defender_prev_path)

    # 迭代参数
    rarl_iters = getattr(args, "rarl_iters", 10)
    attacker_steps_per_iter = getattr(args, "attacker_steps_per_iter", 20000)
    defender_steps_per_iter = getattr(args, "defender_steps_per_iter", 20000)

    for it in range(1, rarl_iters + 1):
        print(f"\n=== RARL Iteration {it} : Train Attacker against Defender_prev (round {it-1}) ===")

        # --------- 加载上一轮的 defender（固定对手）并注入 env ---------
        # IMPORTANT: 这里加载的是 snapshot（只是用于predict的模型拷贝），以保证对手固定不变
        defender_prev_model = load_model_copy(SAC, defender_prev_path, device=device)  # 若你 defender 用非SAC，替换对应类
        defender_prev_wrapper = PolicyWrapper(defender_prev_model)
        set_defender_in_vecenv(env, defender_prev_wrapper)
        set_defender_in_vecenv(eval_env, defender_prev_wrapper)

        # --------- 训练 attacker（从上轮 attacker 参数继续训练） ---------
        # attacker 变量当前为上轮的 attacker 参数（attacker^{t-1}），在此基础上训练得到 attacker^{t}
        attacker.learn(total_timesteps=attacker_steps_per_iter, progress_bar=True, callback=[checkpoint_callback_attacker])
        # 保存本轮训练好的 attacker^{t}
        attacker_curr_path = os.path.join(model_path, "attacker", f"attacker_iter_{it}.zip")
        snapshot_model(attacker, attacker_curr_path)
        print(f"Saved attacker^{it} -> {attacker_curr_path}")

        # --------- NOTE: 不要更新 defender 变量！ defender 仍然是 defender^{t-1} ---------

        # --------- 现在准备训练 defender^{t}，但是按你的要求：固定对手应为上一轮的 attacker（attacker^{t-1})，因此我们必须使用 attacker_prev snapshot ---------
        print(f"\n=== RARL Iteration {it} : Train Defender against Attacker_prev (round {it-1}) ===")
        # 加载上一轮 attacker snapshot（attacker^{t-1}）
        attacker_prev_model = load_model_copy(PPO, attacker_prev_path, device=device)  # 若 attacker 非 PPO，替换类
        attacker_prev_wrapper = PolicyWrapper(attacker_prev_model)
        set_attacker_in_vecenv(env, attacker_prev_wrapper)
        set_attacker_in_vecenv(eval_env, attacker_prev_wrapper)

        # --------- 训练 defender（从 defender^{t-1} 的参数继续训练，得到 defender^{t}） ---------
        defender.learn(total_timesteps=defender_steps_per_iter, progress_bar=True, callback=[checkpoint_callback_defender])
        defender_curr_path = os.path.join(model_path, "defender", f"defender_iter_{it}.zip")
        snapshot_model(defender, defender_curr_path)
        print(f"Saved defender^{it} -> {defender_curr_path}")

        # --------- 迭代结束：更新 attacker_prev_path 与 defender_prev_path 为本轮结束后的模型快照（作为下一轮的上一轮模型） ---------
        # IMPORTANT: attacker_prev for next round should be attacker^{t}  OR attacker^{t}?
        # According to spec: "固定的对手是上一轮训练好的模型" -> next round's "previous" becomes current round's trained models
        attacker_prev_path = attacker_curr_path
        defender_prev_path = defender_curr_path

        # 你也可以在这里定期清理老 snapshot 文件，或保留全部用于回滚

    # 训练完全结束，保存最终模型
    final_attacker_path = os.path.join(model_path, "attacker", "attacker_final.zip")
    final_defender_path = os.path.join(model_path, "defender", "defender_final.zip")
    snapshot_model(attacker, final_attacker_path)
    snapshot_model(defender, final_defender_path)
    print(f"\nTraining finished. Final models:\n {final_attacker_path}\n {final_defender_path}")

# ============================
# 使用示例（替换 args）
# ============================
if __name__ == "__main__":
    class Args:
        env_name = "YourIntersectionEnv-v0"
        attack = True
        adv_steps = 5
        T_horizon = 200
        seed = 0
