import torch
import torch.nn as nn
from policy import IGCARLNet, FniNet


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

        if isinstance(victim_agent, (IGCARLNet, FniNet)):
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
        # 确保 adv_action 和 outputs 在同一个设备上，以 outputs 的设备为准
        adv_action_on_device = adv_action.to(outputs.device)
        # 现在两个张量都在同一个设备上，可以安全地计算损失了
        cost = -loss(outputs, adv_action_on_device)
        # --- 调试点 2: 检查损失值 ---
        cost.backward()
        # 原始的更新步骤
        last_state = torch.clamp(last_state + alpha * last_state.grad.sign(), min=clamp_min, max=clamp_max).detach_()

    return last_state
