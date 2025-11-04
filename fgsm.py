import torch
import torch.nn as nn
from policy import IGCARLNet


def FGSM_v2(adv_action, victim_agent, last_state, epsilon=0.1,
            device="cpu", num_iterations=50):
    alpha = epsilon / num_iterations
    device = torch.device(device)

    last_state = last_state.clone().detach()

    # 如果确实有 batch 维 (1, N)，才 squeeze
    if last_state.dim() == 2 and last_state.size(0) == 1:
        last_state = last_state.squeeze(0)

    low = torch.zeros_like(last_state, device=device)
    high = torch.zeros_like(last_state, device=device)

    # 距离和速度所在的维度索引
    dist_speed_dims = [0, 2, 3, 4, 6, 7, 8, 10, 11, 12, 14, 15, 16, 18, 19, 20, 22, 23, 24, 25]
    # 角度所在的维度索引
    angle_dims = [1, 5, 9, 13, 17, 21]

    high[dist_speed_dims] = 1.0
    low[angle_dims] = -0.5
    high[angle_dims] = 0.5

    # 2) 构造 epsilon 球的局部上下界（保持在 original_state ± epsilon 内）
    orig = last_state.clone().detach().to(device)
    clamp_min = torch.clamp(orig - epsilon, min=low)  # 不低于全局 low
    clamp_max = torch.clamp(orig + epsilon, max=high)  # 不高于全局 high

    last_state = last_state.to(last_state.device)
    if not isinstance(adv_action, torch.Tensor):
        adv_action = torch.tensor(adv_action, dtype=torch.float32, device=device)
    adv_action = adv_action.clone().detach().to(device)  # requires_grad=False
    loss = nn.MSELoss()

    for i in range(num_iterations):
        last_state.requires_grad = True

        if isinstance(victim_agent, IGCARLNet):
            outputs, _, _ = victim_agent(last_state)
            victim_agent.zero_grad()
        else:
            last_state_on_device = last_state.to(victim_agent.device)

            outputs = victim_agent.policy(last_state_on_device.unsqueeze(0), deterministic=True)
            if outputs[0].dim() > 1:
                outputs = outputs[0].squeeze(0)
            else:
                outputs = outputs[0]
            victim_agent.policy.zero_grad()

        # --- 调试点 1: 检查模型输出 ---
        if torch.isnan(outputs).any() or torch.isinf(outputs).any():
            print(f"\n--- DEBUG BREAKPOINT TRIGGERED (Iteration {i}) ---")
            print(">>> 原因: 在模型输出 'outputs' 中检测到 NaN 或 Inf。")
            print("导致问题的输入 'last_state' 是:")
            print(last_state)
            print("\n模型的输出 'outputs' 是:")
            print(outputs)
            print("--- 进入调试器。输入 'c' 继续, 'q' 退出。---")
            breakpoint()

        # cost = -loss(outputs, adv_action).to(device)
        # 确保 adv_action 和 outputs 在同一个设备上，以 outputs 的设备为准
        adv_action_on_device = adv_action.to(outputs.device)

        # 现在两个张量都在同一个设备上，可以安全地计算损失了
        cost = -loss(outputs, adv_action_on_device)
        # --- 调试点 2: 检查损失值 ---
        if torch.isnan(cost).any() or torch.isinf(cost).any():
            print(f"\n--- DEBUG BREAKPOINT TRIGGERED (Iteration {i}) ---")
            print(">>> 原因: 在损失值 'cost' 中检测到 NaN 或 Inf。")
            print("模型的输出 'outputs' 是:")
            print(outputs)
            print("\n目标的 'adv_action' 是:")
            print(adv_action)
            print("\n计算出的 'cost' 是:")
            print(cost)
            print("--- 进入调试器。输入 'c' 继续, 'q' 退出。---")
            breakpoint()

        cost.backward()

        # --- 调试点 3: 检查梯度 ---
        if last_state.grad is None or torch.isnan(last_state.grad).any() or torch.isinf(last_state.grad).any():
            print(f"\n--- DEBUG BREAKPOINT TRIGGERED (Iteration {i}) ---")
            print(">>> 原因: 在 'last_state.grad' 中检测到 NaN, Inf, 或 None。")
            print("'last_state.grad' 的值是:")
            print(last_state.grad)
            print("--- 进入调试器。输入 'c' 继续, 'q' 退出。---")
            breakpoint()
            # 如果梯度有问题，最好不要继续更新，可以选择跳出循环
            break

        # 原始的更新步骤
        last_state = torch.clamp(last_state + alpha * last_state.grad.sign(), min=clamp_min, max=clamp_max).detach_()

        # --- 调试点 4: 检查更新后的状态 ---
        if torch.isnan(last_state).any() or torch.isinf(last_state).any():
            print(f"\n--- DEBUG BREAKPOINT TRIGGERED (After Iteration {i} update) ---")
            print(">>> 原因: 更新后的 'last_state' 中检测到 NaN 或 Inf。")
            print("用于更新的梯度符号 last_state.grad.sign() 是:")
            # 注意：此时 .grad 可能已经不存在，所以我们只打印 last_state
            print("\n新的 'last_state' 是:")
            print(last_state)
            print("--- 进入调试器。输入 'c' 继续, 'q' 退出。---")
            breakpoint()
            break  # 状态已损坏，跳出循环

    # 在函数返回前，最后再做一次检查，确保万无一失
    if torch.isnan(last_state).any():
        print("警告: FGSM 函数最终生成的 state 包含 NaN。将返回原始 state 以防污染数据。")
        return orig

    return last_state