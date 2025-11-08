import optuna
from config import get_config
from train import run_training  # 确保 train.py 中 run_training 的修改已完成

# --- 关键修改 1: 在全局范围解析命令行参数 ---
# 这样，无论函数如何调用，我们都有一份从命令行读取的、固定的基础配置
parser = get_config()
# parse_known_args() 是一个很好的实践，它会忽略它不认识的参数（如果有的话）
# 但在这里用 parse_args() 也可以
cli_args = parser.parse_args()


# --- 关键修改 2: 修改 objective 函数，让它接收这份基础配置 ---
def objective(trial: optuna.trial.Trial, base_args) -> float:
    """
    Optuna 的目标函数。它接收一份基础配置，然后用 trial 的建议值来覆盖它。
    """
    # 我们创建一个 args 的副本，这样每次试验都从一个干净的状态开始
    # .copy() 很重要，可以防止不同 trial 之间互相干扰
    args = base_args

    # --- 在这里建议你需要优化的超参数 ---
    # Optuna会为每次试验提供不同的值来覆盖 base_args 中的值
    args.lr_def = trial.suggest_categorical("lr_def", [1e-3, 1e-4, 1e-5])
    args.batch_size = trial.suggest_categorical("batch_size", [32, 64, 128])
    args.train_step = trial.suggest_categorical("train_step", [40, 60, 80])
    args.adv_steps = trial.suggest_int("adv_steps", 4, 6)
    args.loop_nums = trial.suggest_int("loop_nums", 5, 15)
    args.lagrangian_eps = trial.suggest_categorical("lagrangian_eps", [0.001, 0.01, 0.1, 1.0, 2.0])
    args.adv_sample_ratio = trial.suggest_categorical("adv_sample_ratio", [0.25, 0.5])
    # args.kl_coef = trial.suggest_categorical("kl_coef", [0.1, 1.0, 5.0, 10])


    # --- 固定参数 ---
    # 你在命令行中指定的参数（比如 --cuda_number 1）已经被 cli_args 捕获
    # 如果你还想在这里硬编码一些值，也可以，但从命令行传入更灵活


    # --- 运行训练并返回结果 ---
    try:
        # 调用我们重构的训练函数，传递最终配置好的 args 对象
        final_reward = run_training(args)
        print("DEBUG optimize.py final_reward:", final_reward)

        return final_reward
    except Exception as e:
        # 如果训练过程中出错 (例如梯度爆炸导致 NaN)，
        # 告诉 Optuna 这是一次失败的试验
        print(f"Trial failed with error: {e}")
        print("Failing args:", args) # 打印出导致失败的参数组合，便于调试

        return -1.0  # 返回一个非常差的值


if __name__ == "__main__":
    # --- 设置并启动 Study ---
    db_path = "/data/wq/RARL_AD_2.0-wq/rarl_optimization.db"
    db_connection_string = f"sqlite:///{db_path}"
    # 建议使用新名字，避免与之前不同参数结构的 study 冲突
    study_name = "sac-defender-tuning-v4"

    study = optuna.create_study(
        storage=db_connection_string,
        study_name=study_name,
        direction="maximize",
        load_if_exists=True
    )

    # --- 关键修改 3: 使用 lambda 函数来传递固定的 cli_args ---
    # study.optimize 需要一个只接受 trial 参数的函数。
    # 我们用 lambda 来创建一个这样的匿名函数，它会捕获当前的 cli_args 并传递给我们的 objective
    study.optimize(lambda trial: objective(trial, base_args=cli_args), n_trials=50)

    # --- 打印优化结果 (不变) ---
    print("Optimization finished.")
    print("Best trial:")
    trial = study.best_trial

    print(f"  Value: {trial.value}")
    print("  Params: ")
    for key, value in trial.params.items():
        print(f"    {key}: {value}")