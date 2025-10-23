import numpy as np
import torch
import torch.nn as nn
from policy import FniNet
import torch.optim as optim
import matplotlib.pyplot as plt
import os


def FAB_FGSM_v2(victim_agent,adv_action, last_state,  epsilon=0.1,
                 device='cuda:0', T=50, delta_vanish=0.01):
    # print(last_state)
    m_t = torch.zeros_like(last_state)
    s_t = torch.zeros_like(last_state)
    beta1 = 0.9
    alpha = epsilon / np.sqrt(T + 1)
    #alpha=0.1
    gamma = 0
    weight_decay = 0.01
    optim_epsilon = 1e-7

    clamp_min = torch.max((last_state - epsilon), torch.zeros_like(last_state))
    clamp_max = torch.min((last_state + epsilon), torch.ones_like(last_state))

    last_state = last_state.to(device)

    if not isinstance(adv_action, torch.Tensor):
        adv_action = torch.tensor(adv_action, dtype=torch.float32, device=device)
    loss = nn.MSELoss()

    ori_state = last_state.data
    cost_list = []
    adv_state_list = []
    for t in range(T):
        last_state.requires_grad = True

        t = t + 1
        beta2 = 1 - 0.9 / t

        if isinstance(victim_agent, FniNet):
            outputs, _, _ = victim_agent(last_state)
            victim_agent.zero_grad()
        else:
            outputs = victim_agent.policy(last_state, deterministic=True)
            # print('action pred is', action_pred)
            if outputs[0].dim() > 1:
                outputs = outputs[0].squeeze(0)
            else:
                outputs = outputs[0]
            victim_agent.policy.zero_grad()

        cost = -loss(outputs, adv_action).to(device)
        cost_list.append(cost.item())
        cost.backward()

        grad = last_state.grad
        # Compute γ'
        gamma += np.sqrt(1 - beta2 ** t) / (1 - beta1 ** t)

        # Update m_t and s_t
        m_t = beta1 * m_t + (1 - beta1) * grad
        s_temp = s_t
        s_t = beta2 * s_t + (1 - beta2) * (grad - m_t) ** 2
        s_t = torch.max(s_t, s_temp)

        last_state = last_state * (1 - alpha * weight_decay)

        # Update m_t' and s_t'
        m_t_hat = m_t / (1 - beta1 ** t)
        s_t_hat = (s_t + optim_epsilon) / (1 - beta2 ** t) + delta_vanish / t

        # Update adversarial example
        step = alpha / gamma * torch.sign(m_t_hat / (s_t_hat + optim_epsilon))
        #step = alpha / t * torch.sign(m_t_hat / (s_t_hat + optim_epsilon))
        # print(step)
        last_state = last_state + step

        # Clip to ensure perturbation constraint
        last_state = torch.clamp(last_state, clamp_min, clamp_max).detach()

    return last_state


def PGD(adv_action, victim_agent, last_state, epsilon=0.1,alpha=0.0075,
                 device='cuda:0',num_iterations=50):
    #print(last_state)
    clamp_min = torch.max((last_state - epsilon), torch.zeros_like(last_state))
    clamp_max = torch.min((last_state + epsilon), torch.ones_like(last_state))

    last_state = last_state.to(last_state.device)
    last_state = last_state + torch.empty_like(last_state).uniform_(-epsilon, epsilon)
    last_state = torch.clamp(last_state,clamp_min,clamp_max)
    if not isinstance(adv_action, torch.Tensor):
        adv_action = torch.tensor(adv_action, dtype=torch.float32, device=device)
    loss = nn.MSELoss()

    ori_state = last_state.data
    cost_list = []
    adv_state_list = []
    for i in range(num_iterations):
        last_state.requires_grad = True

        if isinstance(victim_agent, FniNet):
            outputs,_, _ = victim_agent(last_state)
            victim_agent.zero_grad()
        else:
            outputs = victim_agent.policy(last_state.unsqueeze(0), deterministic=True)
            # print('action pred is', action_pred)
            if outputs[0].dim() > 1:
                outputs = outputs[0].squeeze(0)
            else:
                outputs = outputs[0]
            victim_agent.policy.zero_grad()

        #print('o,a',outputs,adv_action)
        cost = -loss(outputs, adv_action).to(device)
        cost_list.append(cost.item())
        cost.backward()

        adv_state = last_state + alpha * last_state.grad.sign()
        eta = torch.clamp(adv_state - ori_state, min=-epsilon, max=epsilon)
        last_state = torch.clamp(ori_state + eta, min=clamp_min, max=clamp_max).detach_()
    return last_state

def PGD_v3(adv_action, victim_agent, last_state, epsilon=0.1,alpha=0.0075,
                 device='cuda:0',num_iterations=50):
    #print(last_state)
    clamp_min = torch.max((last_state - epsilon), torch.zeros_like(last_state))
    clamp_max = torch.min((last_state + epsilon), torch.ones_like(last_state))

    last_state = last_state.to(last_state.device)
    last_state = last_state + torch.empty_like(last_state).uniform_(-epsilon, epsilon)
    last_state = torch.clamp(last_state,clamp_min,clamp_max)
    if not isinstance(adv_action, torch.Tensor):
        adv_action = torch.tensor(adv_action, dtype=torch.float32, device=device)
    loss = nn.MSELoss()

    ori_state = last_state.data
    cost_list = []
    adv_state_list = []
    for i in range(num_iterations):
        last_state.requires_grad = True

        if isinstance(victim_agent, FniNet):
            outputs,_, _ = victim_agent(last_state)
            victim_agent.zero_grad()
        else:
            outputs = victim_agent(last_state.unsqueeze(0), deterministic=True)
            # print('action pred is', action_pred)
            if outputs[0].dim() > 1:
                outputs = outputs[0].squeeze(0)
            else:
                outputs = outputs[0]
            victim_agent.zero_grad()

        #print('o,a',outputs,adv_action)
        cost = -loss(outputs, adv_action).to(device)
        cost_list.append(cost.item())
        cost.backward()

        adv_state = last_state + alpha * last_state.grad.sign()
        eta = torch.clamp(adv_state - ori_state, min=-epsilon, max=epsilon)
        last_state = torch.clamp(ori_state + eta, min=clamp_min, max=clamp_max).detach_()
    return last_state


def FGSM_v2(adv_action, victim_agent, last_state, epsilon=0.1,
                 device='cuda:0',num_iterations=50):
    alpha = epsilon/num_iterations
    #print(last_state)
    ori_state = last_state.detach()
    # print('ori state', ori_state)
    clamp_min = torch.max((ori_state - epsilon), torch.zeros_like(ori_state))
    clamp_max = torch.min((ori_state + epsilon), torch.ones_like(ori_state))

    last_state = ori_state.clone().detach().to(device)
    # print('last_state', last_state)

    if hasattr(victim_agent, 'policy'):
        policy = victim_agent.policy
    else:
        policy = victim_agent

    if not isinstance(adv_action, torch.Tensor):
        adv_action = torch.tensor(adv_action, dtype=torch.float32, device=device)
    adv_action = adv_action.to(device)

    loss_fn = nn.MSELoss(reduction='mean')

    for i in range(num_iterations):
        last_state.requires_grad_(True)

        if isinstance(victim_agent, FniNet):
            outputs,_, _ = victim_agent(last_state)
            policy.zero_grad()
        else:
            outputs = policy(last_state, deterministic=True)
            # print('action pred is', action_pred)
            if isinstance(outputs, tuple):
                outputs = outputs[0]  # usually (action, value)
            policy.zero_grad(set_to_none=True)

        #print('o,a',outputs,adv_action)
        # Compute loss
        cost = -loss_fn(outputs, adv_action)
        cost.backward()

        last_state = torch.clamp(last_state + alpha * last_state.grad.sign(),min=clamp_min, max=clamp_max).detach_()
    return last_state

def FGSM_v3(adv_action, victim_agent, last_state, epsilon=0.1,
                 device='cuda:0',num_iterations=50):
    alpha = epsilon/num_iterations
    print(last_state)
    clamp_min = torch.max((last_state - epsilon), torch.zeros_like(last_state))
    clamp_max = torch.min((last_state + epsilon), torch.ones_like(last_state))

    last_state = last_state.to(last_state.device)
    if not isinstance(adv_action, torch.Tensor):
        adv_action = torch.tensor(adv_action, dtype=torch.float32, device=device)
    loss = nn.MSELoss()

    ori_state = last_state.data
    cost_list = []
    adv_state_list = []
    for i in range(num_iterations):
        last_state.requires_grad = True
        if isinstance(victim_agent, FniNet):
            outputs,_, _ = victim_agent(last_state)
            victim_agent.zero_grad()
        else:
            outputs = victim_agent(last_state.unsqueeze(0), deterministic=True)
            if outputs[0].dim() > 1:
                outputs = outputs[0].squeeze(0)
            else:
                outputs = outputs[0]
            victim_agent.zero_grad()

        #print('o,a',outputs,adv_action)
        cost = -loss(outputs, adv_action).to(device)
        #print(cost)
        cost_list.append(cost.item())
        cost.backward()

        last_state = torch.clamp(last_state + alpha * last_state.grad.sign(),min=clamp_min, max=clamp_max).detach_()
    return last_state


def cw_attack_v2(victim_agent, last_state, adv_action, targeted=True, epsilon=0.1,c=1e-4, lr=0.01, iters=1000):

    if not isinstance(adv_action, torch.Tensor):
        adv_action = torch.tensor(adv_action, dtype=torch.float32, device=last_state.device)

    BINARY_SEARCH_STEPS = 1  # number of times to adjust the constant with binary search
    MAX_ITERATIONS = iters  # number of iterations to perform gradient descent

    # C的初始化边界
    lower_bound = 0
    upper_bound = 1e10

    # 若攻击成功 记录最好的l2 loss 以及 adv_state
    o_bestl2 = 1e10
    o_bestscore = -1
    o_bestattack = np.zeros(last_state.shape)

    #epsilon-ball around last_state
    clamp_min = torch.max((last_state - epsilon), torch.zeros_like(last_state))
    clamp_max = torch.min((last_state + epsilon), torch.ones_like(last_state))

    #用于将tanh空间转换时的状态范围约束在epsilon-ball内
    mul = (clamp_max-clamp_min)/2
    plus = (clamp_max+clamp_min)/2

    factor = 1e-6
    last_state = torch.clamp(last_state, -1 + factor, 1 - factor)

    # 计算 tanh 逆变换的 x 原始值
    def to_tanh_space(x):
        return mul*torch.tanh(x)+plus

    # 计算 x 的逆变换
    def from_tanh_space(x):
        return torch.atanh((x-plus)/mul*0.999999)

    # 原始输入图像的逆变换形式
    original_state = from_tanh_space(last_state)
    #print('l,o',last_state,original_state)
    loss_list=[]

    #外循环，用于寻找最优c值 , c值用来控制loss中loss1和loss2的权重
    for outer_step in range(BINARY_SEARCH_STEPS):
        print("best_l2={},confidence={}".format(o_bestl2,c))

        # 初始化扰动变量 w，它是优化目标，初始为原始图像
        w = torch.zeros_like(last_state, requires_grad=True).to(last_state.device)
        # 需要优化的目标是 perturbation w
        optimizer = optim.Adam([w], lr=lr)
        #内循环，用于优化以生成对抗状态
        for step in range(1,MAX_ITERATIONS+1):
            # 通过 tanh 变换生成对抗样本 x'
            adv_state = to_tanh_space(original_state + w)

            # 计算模型输出
            if isinstance(victim_agent, FniNet):
                outputs,_, _ = victim_agent(adv_state)
            else:
                outputs = victim_agent.policy(adv_state.unsqueeze(0), deterministic=True)
                if outputs[0].dim() > 1:
                    outputs = outputs[0].squeeze(0)
                else:
                    outputs = outputs[0]

            # 创建一个标签，用于 targeted 或 non-targeted 的攻击目标
            if targeted:
                loss1 = nn.MSELoss()(outputs,adv_action)
            else:
                # 非目标攻击
                if isinstance(victim_agent, (Actor, DarrlNet)):
                    ori_action, _, _ = victim_agent(last_state)
                else:
                    ori_action = victim_agent.policy(last_state, deterministic=True)
                    # print('action pred is', action_pred)
                    if ori_action[0].dim() > 1:
                        ori_action = ori_action[0].squeeze(0)
                    else:
                        ori_action = ori_action[0]
                loss1 = -nn.MSELoss()(outputs,ori_action)

            # 损失函数由两部分组成：1. 分类损失 2. L2 范数约束
            l2_loss = torch.norm(adv_state - last_state, p=2)
            loss = c * l2_loss + loss1.mean()
            loss_list.append(loss.item())

            # print out loss every 10%
            if step % (MAX_ITERATIONS // 10) == 0:
                print("iteration={} loss={} loss1={} loss2={} action={}".format(step, loss, loss1, l2_loss,outputs))

            l2 = l2_loss
            #攻击成功的情况 即成功误导agent做出指定action
            if (l2 < o_bestl2) and (outputs == adv_action):
                print("attack success l2={} target_action={}".format(l2, adv_action))
                o_bestl2 = l2
                o_bestscore = outputs
                o_bestattack = adv_state.data.cpu().numpy()

            # 反向传播和优化
            optimizer.zero_grad()
            loss.backward(retain_graph=True)
            optimizer.step()

        confidence_old = 1
        if (o_bestscore == adv_action) and o_bestscore != -1:
            # 攻击成功，减小c
            upper_bound = min(upper_bound, c)
            if upper_bound < 1e9:
                confidence_old = c
                c = (lower_bound + upper_bound) / 2
            else:
                lower_bound = max(lower_bound, c)
                confidence_old = c
                if upper_bound < 1e9:
                    c = (lower_bound + upper_bound) / 2
                else:
                    c *= 10
            # torch.sign
        print("outer_step={} confidence {}->{}".format(outer_step, confidence_old, c))

    # 返回优化后的对抗样本
    return o_bestattack

    # # 创建保存图像的目录（如果不存在）
    # save_dir = "lossForCW"
    # if not os.path.exists(save_dir):
    #     os.makedirs(save_dir)
    #
    # # 绘制 loss2 随迭代次数的变化并保存图像
    # plt.plot(range(len(loss_list)), loss_list, label='Loss over iterations')
    # plt.xlabel('Iteration')
    # plt.ylabel('Loss')
    # plt.title('Loss vs Iterations')
    # plt.legend()
    #
    # # 保存图像到指定文件夹
    # plt.savefig(os.path.join(save_dir, 'loss_iterations_iter{}.png'.format(iters)))


