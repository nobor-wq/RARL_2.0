from off_policy_algorithm import OffPolicyDefensiveAlgorithm, OffPolicyBaseAlgorithm
from stable_baselines3 import PPO, SAC, TD3
from policy import FniNet
import torch as th
import os
from fgsm import FGSM_v2
import numpy as np
from torch.nn import functional as F
from stable_baselines3.common.utils import polyak_update
from stable_baselines3.common.utils import obs_as_tensor
import torch.optim as optim

class DefensiveSAC(OffPolicyDefensiveAlgorithm, SAC):
    def __init__(self, custom_args, best_model_path,  *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.best_model = custom_args.best_model
        self.algo = custom_args.algo
        self.algo_adv = custom_args.algo_adv
        self.path_adv = custom_args.path_adv
        self.path_def = custom_args.path_def
        self.env_name = custom_args.env_name
        self.fni_model_path = custom_args.fni_model_path
        self.unlimited_attack = custom_args.unlimited_attack
        self.attack_method = custom_args.attack_method
        self.decouple = custom_args.decouple
        self.attack_eps = custom_args.attack_eps
        self.trained_adv = None
        self.trained_agent = None
        self.trained_expert = None
        self.action_diff = custom_args.action_diff
        self.use_expert = custom_args.use_expert
        self.kl_div =  custom_args.use_kl
        # 2025-11-01 wq TODO kl系数调整
        self.kl_coef = custom_args.kl_coef if hasattr(custom_args, 'kl_coef') else 0.1
        self.use_lagrangian = custom_args.use_lagrangian if hasattr(custom_args, 'use_lagrangian') else False
        if self.use_lagrangian:
            # 约束的阈值
            self.eps2 = custom_args.lagrangian_eps if hasattr(custom_args, 'lagrangian_eps') else 0.05
            # 拉格朗日乘子的学习率
            lam_lr = custom_args.lagrangian_lr if hasattr(custom_args, 'lagrangian_lr') else 1e-4

            # 我们优化 log(lambda) 来保证 lambda > 0
            self.log_lam2 = th.zeros(1, requires_grad=True, device=self.device)
            self.lam2_optimizer = optim.Adam([self.log_lam2], lr=lam_lr)


        # 2025-09-23 wq 加载敌手和防御者
        # if self.best_model:
        #     model_path_adv = os.path.join(self.path_adv, self.algo_adv, self.env_name, 'best_model/best_model')
        #     model_path_age = os.path.join(self.path_def, self.algo, self.env_name, 'best_model/best_model')
        # else:
        #     model_path_adv = os.path.join(self.path_adv, self.algo_adv, self.env_name, 'lunar')
        #     model_path_age = os.path.join(self.path_def, self.algo, self.env_name, 'lunar')
        # self.trained_adv = PPO.load(model_path_adv, device=self.device)
        # self.trained_agent = SAC.load(model_path_age, device=self.device)

        self.fni_flag = True if self.algo in ('FNI', 'DARRL') else False
        self.max_epi_reward = 0
        self.best_model_path = best_model_path
        self._last_infos = None


    def learn(
        self,
        total_timesteps,
        callback=None,
        log_interval=4,
        tb_log_name="SAC",
        reset_num_timesteps=True,
        progress_bar=False,
        trained_agent=None,
        trained_adv=None,
        trained_expert=None,
    ):

        self.trained_agent = trained_agent
        self.trained_adv = trained_adv
        self.trained_expert = trained_expert

        return super().learn(
            total_timesteps=total_timesteps,
            callback=callback,
            log_interval=log_interval,
            tb_log_name=tb_log_name,
            reset_num_timesteps=reset_num_timesteps,
            progress_bar=progress_bar,
        )

    # 2025-09-23 wq 训练
    def train(self, gradient_steps: int, batch_size: int = 64) -> None:
        # Switch to train mode (this affects batch norm / dropout)
        self.policy.set_training_mode(True)
        # Update optimizers learning rate
        optimizers = [self.actor.optimizer, self.critic.optimizer]
        if self.ent_coef_optimizer is not None:
            optimizers += [self.ent_coef_optimizer]

        # Update learning rate according to lr schedule
        self._update_learning_rate(optimizers)

        ent_coef_losses, ent_coefs = [], []
        actor_losses, critic_losses = [], []

        for gradient_step in range(gradient_steps):
            # Sample replay buffer
            replay_data = self.replay_buffer.sample(batch_size, env=self._vec_normalize_env)  # type: ignore[union-attr]

            obs = replay_data.observations  # th.Tensor, shape (B, ...)
            obs_eps = replay_data.observations_eps
            obs_is_per = replay_data.obs_is_perturbed

            obs_used = th.where(obs_is_per, obs_eps, obs)

            # We need to sample because `log_std` may have changed between two gradient steps
            if self.use_sde:
                self.actor.reset_noise()

            # Action by the current actor for the sampled state
            actions_pi, log_prob = self.actor.action_log_prob(obs_used)
            log_prob = log_prob.reshape(-1, 1)

            ent_coef_loss = None
            if self.ent_coef_optimizer is not None and self.log_ent_coef is not None:
                # Important: detach the variable from the graph
                # so we don't change it with other losses
                # see https://github.com/rail-berkeley/softlearning/issues/60
                ent_coef = th.exp(self.log_ent_coef.detach())
                ent_coef_loss = -(self.log_ent_coef * (log_prob + self.target_entropy).detach()).mean()
                ent_coef_losses.append(ent_coef_loss.item())
            else:
                ent_coef = self.ent_coef_tensor

            ent_coefs.append(ent_coef.item())

            # Optimize entropy coefficient, also called
            # entropy temperature or alpha in the paper
            if ent_coef_loss is not None and self.ent_coef_optimizer is not None:
                self.ent_coef_optimizer.zero_grad()
                ent_coef_loss.backward()
                self.ent_coef_optimizer.step()

            with th.no_grad():
                # Select action according to policy
                next_actions, next_log_prob = self.actor.action_log_prob(replay_data.next_observations)
                # Compute the next Q values: min over all critics targets
                next_q_values = th.cat(self.critic_target(replay_data.next_observations, next_actions), dim=1)
                next_q_values, _ = th.min(next_q_values, dim=1, keepdim=True)
                # add entropy term
                next_q_values = next_q_values - ent_coef * next_log_prob.reshape(-1, 1)
                # td error + entropy term
                target_q_values = replay_data.rewards + (1 - replay_data.dones) * self.gamma * next_q_values

            # Get current Q-values estimates for each critic network
            # using action from the replay buffer

            current_q_values = self.critic(obs_used, replay_data.actions)

            # Compute critic loss
            critic_loss = 0.5 * sum(F.mse_loss(current_q, target_q_values) for current_q in current_q_values)
            assert isinstance(critic_loss, th.Tensor)  # for type checker
            critic_losses.append(critic_loss.item())  # type: ignore[union-attr]

            # Optimize the critic
            self.critic.optimizer.zero_grad()
            critic_loss.backward()
            self.critic.optimizer.step()

            # Compute actor loss
            # Alternative: actor_loss = th.mean(log_prob - qf1_pi)
            # Min over all critic networks

            # 2025-09-23 wq 修改loss，添加动作
            # obs_np = replay_data.observations.detach().cpu().numpy().copy()
            # with th.no_grad():
            #     obs_tensor = obs_as_tensor(obs_np, self.device)
            #     # if self.fni_flag:
            #     #     actions, std, _action = self.trained_agent(obs_tensor)
            #     #     actions = actions.detach().cpu().numpy()
            #     # else:
            #     #     actions, _states = self.trained_agent.predict(obs_tensor.cpu(), deterministic=True)
            #
            #     actions, _states = self.trained_agent.predict(obs_tensor.cpu(), deterministic=True)
            #     actions_tensor = th.tensor(actions, device=self.device)
            #
            #     # 2025-10-22 wq 插入剩余攻击次数/总攻击次数
            #
            #     device = obs_tensor.device
            #     dtype = obs_tensor.dtype
            #     batch = obs_tensor.shape[0]
            #     ones_middle = th.ones((batch, 1), device=device, dtype=dtype)
            #     # ones_middle = ones_middle * 0.5  # 非原地
            #
            #     obs_adv = th.cat([obs_tensor, ones_middle, actions_tensor], dim=-1)  # -> (batch, 28)
            #     # obs_tensor[:, -1] = actions_tensor.squeeze(-1)
            #     adv_action, _ = self.trained_adv.predict(obs_adv.cpu(), deterministic=True)
            #
            # adv_action_mask = adv_action[:, 0] > 0
            # if adv_action_mask:
            #     adv_action_eps = adv_action[:, 1]
            #     state_eps = FGSM_v2(adv_action_eps, victim_agent=self.trained_agent, epsilon=self.attack_eps,
            #                         last_state=obs_np, device=self.device)
            #
            #     if isinstance(state_eps, th.Tensor):
            #         state_eps = state_eps.detach().cpu().numpy()

            # 2025-10-22 wq 这里修改loss，如果扰动了，就添加一个动作约束，是扰动前的动作和扰动后的动作

            q_values_pi = th.cat(self.critic(obs_used, actions_pi), dim=1)
            min_qf_pi, _ = th.min(q_values_pi, dim=1, keepdim=True)
            actor_loss_policy = (ent_coef * log_prob - min_qf_pi).mean()

            if self.action_diff:
                target_agent = self.trained_expert if self.use_expert else self.trained_agent
                if self.kl_div:
                    # 步骤 1: 获取专家在“干净”状态下的策略分布
                    with th.no_grad():
                        # 直接调用 actor 获取均值和 log_std
                        # 注意：不同的 SB3 版本，Actor 的返回值可能略有不同，但通常是 (mean, log_std)
                        # 我们使用 get_action_dist_params，这是许多版本都有的
                        mean_actions_expert, log_std_expert, _ = target_agent.policy.actor.get_action_dist_params(
                            obs)
                        # 手动构建正态分布对象
                        # SAC 通常使用 TanhSquashedGaussianDistribution，但为了计算 KL 散度，
                        # 我们先计算基础高斯分布的 KL 散度，这通常足够作为约束。
                        # 如果需要更精确，可以使用 SB3 提供的分布类。
                        # 这里我们使用标准正态分布，因为它简单且通常有效。
                        dist_expert = th.distributions.Normal(mean_actions_expert, th.exp(log_std_expert))

                    # 步骤 2: 获取当前防御模型在“可能被扰动”状态下的策略分布
                    # 使用同样的方法获取当前模型的分布参数
                    mean_actions_current, log_std_current, _ = self.policy.actor.get_action_dist_params(obs_used)
                    dist_current = th.distributions.Normal(mean_actions_current, th.exp(log_std_current))

                    # 步骤 3: 计算两个分布之间的KL散度
                    # sum(-1) 是为了对动作维度求和，得到每个样本的 KL 散度标量
                    kl_div = th.distributions.kl.kl_divergence(dist_expert, dist_current).sum(-1)

                    # 步骤 4: 使用掩码
                    mask = obs_is_per.to(device=kl_div.device, dtype=th.bool).view(-1)
                    masked_kl_div = kl_div[mask]

                    if masked_kl_div.numel() > 0:
                        kl_loss_masked = masked_kl_div.mean()
                    else:
                        kl_loss_masked = th.tensor(0.0, device=self.device)

                    # 步骤 5: 添加到总损失
                    actor_loss = actor_loss_policy + self.kl_coef * kl_loss_masked
                    print("DEBUG defensive_sac.py train actor_loss_policy: ", actor_loss_policy)
                    print("DEBUG defensive_sac.py train self.kl_coef * kl_loss_masked: ", self.kl_coef * kl_loss_masked)
                    print("DEBUG defensive_sac.py train actor_loss: ", actor_loss)

                    self.logger.record("train_def/self.kl_coef_kl_loss_masked", (self.kl_coef * kl_loss_masked).item())
                elif self.use_lagrangian:
                    # 步骤 1: 计算“安全”动作 (在干净状态 obs 下的动作)
                    # 我们不希望梯度流向这个目标，所以使用 no_grad
                    with th.no_grad():
                        # 使用 action_log_prob 获取采样动作，与 actions_pi 保持一致
                        actions_clean_np, _states = target_agent.predict(obs.cpu(), deterministic=True)
                        actions_clean = th.as_tensor(actions_clean_np, device=actions_pi.device, dtype=actions_pi.dtype)
                    # 步骤 2: 计算逐样本的均方误差损失 (约束的基础)
                    # actions_pi 是在扰动状态 obs_used 下的动作
                    policy_loss_per_sample = F.mse_loss(actions_pi, actions_clean, reduction='none').mean(dim=1).squeeze()

                    # 步骤 3: 准备拉格朗日乘子
                    # 我们在 actor loss 中使用 lambda 的值，但不通过 actor loss 更新 lambda
                    lam2 = self.log_lam2.exp().detach()

                    # 步骤 4: 计算 Actor Loss (Primal Objective)
                    # 只对被扰动的样本施加约束
                    mask = obs_is_per.view(-1)
                    # 将 policy_loss 应用于被扰动的样本，未扰动的损失为0
                    masked_policy_loss = policy_loss_per_sample[mask]

                    self.logger.record("train_def/masked_policy_loss", masked_policy_loss.mean().item())

                    # 将约束项加到 actor loss 中
                    # 注意：这里需要确保 actor_loss_policy 也只在 mask 上应用 lagrangian loss
                    # 一个简单的实现是先计算 mean，然后加起来
                    lagrangian_penalty = th.tensor(0.0, device=self.device)
                    if masked_policy_loss.numel() > 0:
                        lagrangian_penalty = (lam2 * masked_policy_loss).mean()

                    actor_loss = actor_loss_policy + lagrangian_penalty

                    # 步骤 5: 更新 Actor 参数 (Primal Update)
                    self.actor.optimizer.zero_grad()
                    actor_loss.backward()
                    self.actor.optimizer.step()

                    # 步骤 6: 更新 Lambda (Dual Update)
                    # 只有在存在被扰动样本时，才进行 lambda 更新
                    # 这是梯度上升，通过最小化负的目标函数实现
                    # 我们不希望梯度流回 actor，所以 detach g2
                    with th.no_grad():  # 这里不需要梯度
                        constraint_violation = policy_loss_per_sample - self.eps2
                        masked_violation = constraint_violation[mask]
                    if masked_violation.numel() > 0:
                        # 最小化 -log(lambda) * g，等价于最大化 log(lambda) * g
                        lambda_loss = -(self.log_lam2 * masked_violation).mean()

                        self.lam2_optimizer.zero_grad()
                        lambda_loss.backward()
                        self.lam2_optimizer.step()
                        self.logger.record("train_def/constraint_violation",
                                           masked_violation.mean().item())
                    else:
                        # 如果没有攻击样本，就记录一个 0
                        self.logger.record("train_def/constraint_violation", 0)

                    # 记录 lam2 的值，方便监控
                    self.logger.record("train_def/lambda2", self.log_lam2.exp().item())

                    # actor_loss 已经反向传播，这里我们只记录 item
                    actor_losses.append(actor_loss.item())
                else:
                    with th.no_grad():
                        actions_clean_np, _states = target_agent.predict(obs.cpu(), deterministic=True)
                    actions_clean = th.as_tensor(actions_clean_np, device=actions_pi.device, dtype=actions_pi.dtype)
                    per_elem_sq = (actions_clean - actions_pi) ** 2
                    print("DEBUG defensive_sac.py train per_elem_sq: ", per_elem_sq, " shape: ", per_elem_sq.shape)
                    per_sample_mse = per_elem_sq.mean(dim=1)
                    print("DEBUG defensive_sac.py train per_sample_mse: ", per_sample_mse, " shape: ", per_sample_mse.shape)
                    mask = obs_is_per.to(device=per_sample_mse.device)
                    print("DEBUG defensive_sac.py train mask: ", mask, "shape: ", mask.shape)
                    if mask.dtype.is_floating_point:
                        mask = mask > 0.5
                    else:
                        mask = mask.bool()
                    mask = mask.view(-1)
                    print("DEBUG defensive_sac.py train mask_1: ", mask, "shape_1: ", mask.shape)
                    # apply mask: false -> zero loss, true -> keep per-sample mse
                    mask_float = mask.to(dtype=per_sample_mse.dtype)
                    print("DEBUG defensive_sac.py train mask_float: ", mask_float, " shape: ", mask_float.shape)
                    masked_per_sample = per_sample_mse * mask_float  # (B,)
                    print("DEBUG defensive_sac.py train masked_per_sample: ", masked_per_sample, " shape: ", masked_per_sample.shape)
                    action_loss_masked = masked_per_sample.mean()
                    print("DEBUG defensive_sac.py train action_loss_masked: ", action_loss_masked, " shape: ", action_loss_masked.shape)
                    actor_loss = actor_loss_policy + action_loss_masked * 10
                    print("DEBUG defensive_sac.py train actor_loss_policy: ", actor_loss_policy, " shape: ", actor_loss_policy.shape)
                    print("DEBUG defensive_sac.py train actor_loss: ", actor_loss, " shape: ", actor_loss.shape)
                    self.logger.record("train_def/action_loss_masked_10", (action_loss_masked * 10).item())
            else:
                actor_loss = actor_loss_policy

            if not self.use_lagrangian:
                actor_losses.append(actor_loss.item())
                # Optimize the actor
                self.actor.optimizer.zero_grad()
                actor_loss.backward()
                self.actor.optimizer.step()

            # Update target networks
            if gradient_step % self.target_update_interval == 0:
                polyak_update(self.critic.parameters(), self.critic_target.parameters(), self.tau)
                # Copy running stats, see GH issue #996
                polyak_update(self.batch_norm_stats, self.batch_norm_stats_target, 1.0)

        self._n_updates += gradient_steps

        self.logger.record("train_def/n_updates", self._n_updates, exclude="tensorboard")
        self.logger.record("train_def/ent_coef", np.mean(ent_coefs))
        self.logger.record("train_def/actor_loss", np.mean(actor_losses))
        self.logger.record("train_def/critic_loss", np.mean(critic_losses))
        if len(ent_coef_losses) > 0:
            self.logger.record("train_def/ent_coef_loss", np.mean(ent_coef_losses))


class BaseSAC(OffPolicyBaseAlgorithm, SAC):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    # 2025-09-23 wq 训练
    def train(self, gradient_steps: int, batch_size: int = 64) -> None:
        # Switch to train mode (this affects batch norm / dropout)
        self.policy.set_training_mode(True)
        # Update optimizers learning rate
        optimizers = [self.actor.optimizer, self.critic.optimizer]
        if self.ent_coef_optimizer is not None:
            optimizers += [self.ent_coef_optimizer]

        # Update learning rate according to lr schedule
        self._update_learning_rate(optimizers)

        ent_coef_losses, ent_coefs = [], []
        actor_losses, critic_losses = [], []

        for gradient_step in range(gradient_steps):
            # Sample replay buffer
            replay_data = self.replay_buffer.sample(batch_size, env=self._vec_normalize_env)  # type: ignore[union-attr]

            # We need to sample because `log_std` may have changed between two gradient steps
            if self.use_sde:
                self.actor.reset_noise()

            # Action by the current actor for the sampled state
            actions_pi, log_prob = self.actor.action_log_prob(replay_data.observations)
            log_prob = log_prob.reshape(-1, 1)

            ent_coef_loss = None
            if self.ent_coef_optimizer is not None and self.log_ent_coef is not None:
                # Important: detach the variable from the graph
                # so we don't change it with other losses
                # see https://github.com/rail-berkeley/softlearning/issues/60
                ent_coef = th.exp(self.log_ent_coef.detach())
                ent_coef_loss = -(self.log_ent_coef * (log_prob + self.target_entropy).detach()).mean()
                ent_coef_losses.append(ent_coef_loss.item())
            else:
                ent_coef = self.ent_coef_tensor

            ent_coefs.append(ent_coef.item())

            # Optimize entropy coefficient, also called
            # entropy temperature or alpha in the paper
            if ent_coef_loss is not None and self.ent_coef_optimizer is not None:
                self.ent_coef_optimizer.zero_grad()
                ent_coef_loss.backward()
                self.ent_coef_optimizer.step()

            with th.no_grad():
                # Select action according to policy
                next_actions, next_log_prob = self.actor.action_log_prob(replay_data.next_observations)
                # Compute the next Q values: min over all critics targets
                next_q_values = th.cat(self.critic_target(replay_data.next_observations, next_actions), dim=1)
                next_q_values, _ = th.min(next_q_values, dim=1, keepdim=True)
                # add entropy term
                next_q_values = next_q_values - ent_coef * next_log_prob.reshape(-1, 1)
                # td error + entropy term
                target_q_values = replay_data.rewards + (1 - replay_data.dones) * self.gamma * next_q_values

            # Get current Q-values estimates for each critic network
            # using action from the replay buffer
            current_q_values = self.critic(replay_data.observations, replay_data.actions)

            # Compute critic loss
            critic_loss = 0.5 * sum(F.mse_loss(current_q, target_q_values) for current_q in current_q_values)
            assert isinstance(critic_loss, th.Tensor)  # for type checker
            critic_losses.append(critic_loss.item())  # type: ignore[union-attr]

            # Optimize the critic
            self.critic.optimizer.zero_grad()
            critic_loss.backward()
            self.critic.optimizer.step()

            # Compute actor loss
            # Alternative: actor_loss = th.mean(log_prob - qf1_pi)
            # Min over all critic networks
            # 2025-10-21 wq 添加动作差约束

            q_values_pi = th.cat(self.critic(replay_data.observations, actions_pi), dim=1)
            min_qf_pi, _ = th.min(q_values_pi, dim=1, keepdim=True)

            actor_loss = (ent_coef * log_prob - min_qf_pi).mean()
            actor_losses.append(actor_loss.item())

            # Optimize the actor
            self.actor.optimizer.zero_grad()
            actor_loss.backward()
            self.actor.optimizer.step()

            # Update target networks
            if gradient_step % self.target_update_interval == 0:
                polyak_update(self.critic.parameters(), self.critic_target.parameters(), self.tau)
                # Copy running stats, see GH issue #996
                polyak_update(self.batch_norm_stats, self.batch_norm_stats_target, 1.0)

        self._n_updates += gradient_steps

        self.logger.record("train_def/n_updates", self._n_updates, exclude="tensorboard")
        self.logger.record("train_def/ent_coef", np.mean(ent_coefs))
        self.logger.record("train_def/actor_loss", np.mean(actor_losses))
        self.logger.record("train_def/critic_loss", np.mean(critic_losses))
        if len(ent_coef_losses) > 0:
            self.logger.record("train_def/ent_coef_loss", np.mean(ent_coef_losses))




