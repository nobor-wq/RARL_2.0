# RARL_AD_2.0-wq 使用说明

本项目围绕对抗强化学习与安全约束强化学习，提供攻防训练、基线实验和评测脚本。核心算法包含 RARL/IGCARL（攻击者）、SAC/Lagrangian SAC（防御者）等，环境定义位于 `Environment/environment/env3` 及相关目录。


## 训练命令示例

#  Lagrangian 调试
```bash
parallel -j 3 python train.py --swanlab --decouple --train_step 1 --loop_nums 2 --adv_steps 5 \
  --attack_eps 0.05 --action_diff --use_DualBuffer --use_lag --adv_sample_ratio 0.5 \
  --batch_size 64 --use_cuda --cuda_number 1 --lag_eps 0.1 --eval_episode 2 --seed {1} ::: 3 4 5
```

基础 SAC-Lag 训练与 FNI 任务：
```bash
# 基线 SAC-lag
python base_train.py --algo SAC_lag --seed 3 --use_cuda --cuda_number 1 --train_step 2000 --swanlab

# FNI 任务
taskset -c 50-78 python run_short_dis_tasks.py --train_step 2000 --swanlab --seed 3
```

## 单独训练攻击者

# RARL 攻击者
```bash
python train.py --decouple --adv_test --train_step 500 --adv_steps 5 \
  --use_cuda --cuda_number 0 --attack_eps 0.05 --batch_size 64 --seed 18 --swanlab --algo {algo} \
 --trained_seed 8  (--unlimited_attack)
```

## 测试命令
# RARL 防御评估
```bash
python defense_test.py --train_step 200 --use_cuda --cuda_number 1 --algo RARL \
  --attack_eps 0.05 --adv_steps 5 --seed 11 --trained_seed 8
```

基线与不同算法评测：
```bash
parallel -j 3 python defense_test.py --train_step 5 --use_cuda --cuda_number 0 --algo PPO \
  --attack_eps 0.05 --adv_steps 5 --trained_seed 0 --attack --seed {1} ::: 15 20 25
parallel -j 3 python defense_test.py --train_step 200 --use_cuda --cuda_number 0 --algo SAC \
  --attack_eps 0.05 --adv_steps 5 --attack --trained_seed 8 --seed {1} ::: 15 20 25
parallel -j 3 python defense_test.py --train_step 200 --use_cuda --cuda_number 0 --algo SAC_lag \
  --attack_eps 0.05 --adv_steps 5 --attack --seed 20 --trained_seed {1} ::: 3 5 8
parallel -j 3 python defense_test.py --train_step 200 --use_cuda --cuda_number 0 --algo FNI \
  --attack_eps 0.05 --adv_steps 5 --attack --seed 20 --trained_seed {1} ::: 3 5 8
parallel -j 3 python defense_test.py --train_step 200 --use_cuda --cuda_number 0 --algo DARRL \
  --attack_eps 0.05 --adv_steps 5 --attack --seed 20 --trained_seed {1} ::: 0 3 8
parallel -j 3 python defense_test.py --train_step 200 --env_name TrafficEnv3-v0 --use_cuda --cuda_number 0 \
  --algo IGCARL --attack_eps 0.07 --adv_steps 5 --attack --seed 20 --trained_seed {1} ::: 1 5 11
```

## 额外提示
- 主要代码入口：`train.py`（攻防训练）、`base_train.py`（基线）、`defense_test.py`（评测）。
- 攻击者实现：`on_policy_algorithm.py`；防御者实现：`off_policy_algorithm.py`、`defensive_sac.py`。
- 若使用 GPU，请根据实际设备调整 `--use_cuda` 与 `--cuda_number`。训练过程中建议关注 GPU 显存与 SWANLab/Optuna 面板。
