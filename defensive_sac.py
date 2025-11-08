from off_policy_algorithm import OffPolicyDefensiveAlgorithm, OffPolicyBaseAlgorithm
from stable_baselines3 import SAC
import torch as th
import numpy as np
from torch.nn import functional as F
from stable_baselines3.common.utils import polyak_update
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
        self.kl_coef = custom_args.kl_coef if hasattr(custom_args, 'kl_coef') else 10.0
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

            obs = replay_data.observations
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
            q_values_pi = th.cat(self.critic(obs_used, actions_pi), dim=1)
            min_qf_pi, _ = th.min(q_values_pi, dim=1, keepdim=True)
            actor_loss_policy = (ent_coef * log_prob - min_qf_pi).mean()



            if self.action_diff:
                target_agent = self.trained_expert if self.use_expert else self.trained_agent
                if self.kl_div:
                    # 步骤 1: 获取专家在“干净”状态下的策略分布
                    with th.no_grad():
                        mean_actions_expert, log_std_expert, _ = target_agent.policy.actor.get_action_dist_params(obs)
                        dist_expert = th.distributions.Normal(mean_actions_expert, th.exp(log_std_expert))
                    mean_actions_current, log_std_current, _ = self.policy.actor.get_action_dist_params(obs_used)
                    dist_current = th.distributions.Normal(mean_actions_current, th.exp(log_std_current))

                    kl_div = th.distributions.kl.kl_divergence(dist_expert, dist_current).sum(-1)

                    mask = obs_is_per.to(device=kl_div.device, dtype=th.bool).view(-1)
                    masked_kl_div = kl_div[mask]

                    if masked_kl_div.numel() > 0:
                        kl_loss_masked = masked_kl_div.mean()
                    else:
                        kl_loss_masked = th.tensor(0.0, device=self.device)

                    actor_loss = actor_loss_policy + self.kl_coef * kl_loss_masked

                    self.logger.record("train_def/self.kl_coef_kl_loss_masked", (self.kl_coef * kl_loss_masked).item())
                elif self.use_lagrangian:
                    # 步骤 1: 计算“安全”动作 (在干净状态 obs 下的动作)
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

                    # 将约束项加到 actor loss 中
                    # 注意：这里需要确保 actor_loss_policy 也只在 mask 上应用 lagrangian loss
                    # 一个简单的实现是先计算 mean，然后加起来
                    lagrangian_penalty = th.tensor(0.0, device=self.device)
                    if masked_policy_loss.numel() > 0:
                        lagrangian_penalty = (lam2 * masked_policy_loss).mean()

                    actor_loss = actor_loss_policy + lagrangian_penalty

                    self.logger.record("train_def/lagrangian_penalty", lagrangian_penalty.item())

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
                        lambda_loss = -(self.log_lam2 * masked_violation.detach()).mean()

                        self.lam2_optimizer.zero_grad()
                        lambda_loss.backward()
                        self.lam2_optimizer.step()
                        self.log_lam2.data.clamp_(-5, 10)  # 裁剪 log-lambda 的值

                    # 记录 lam2 的值，方便监控
                    self.logger.record("train_def/lambda2", self.log_lam2.exp().item())

                    # actor_loss 已经反向传播，这里我们只记录 item
                    actor_losses.append(actor_loss.item())
                else:
                    with th.no_grad():
                        actions_clean_np, _states = target_agent.predict(obs.cpu(), deterministic=True)
                    actions_clean = th.as_tensor(actions_clean_np, device=actions_pi.device, dtype=actions_pi.dtype)
                    per_elem_sq = (actions_clean - actions_pi) ** 2
                    per_sample_mse = per_elem_sq.mean(dim=1)
                    mask = obs_is_per.to(device=per_sample_mse.device)
                    if mask.dtype.is_floating_point:
                        mask = mask > 0.5
                    else:
                        mask = mask.bool()
                    mask = mask.view(-1)
                    mask_float = mask.to(dtype=per_sample_mse.dtype)
                    masked_per_sample = per_sample_mse * mask_float  # (B,)
                    action_loss_masked = masked_per_sample.mean()
                    actor_loss = actor_loss_policy + action_loss_masked * 10
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

        self.logger.record("train_pre_def/n_updates", self._n_updates, exclude="tensorboard")
        self.logger.record("train_pre_def/ent_coef", np.mean(ent_coefs))
        self.logger.record("train_pre_def/actor_loss", np.mean(actor_losses))
        self.logger.record("train_pre_def/critic_loss", np.mean(critic_losses))
        if len(ent_coef_losses) > 0:
            self.logger.record("train_pre_def/ent_coef_loss", np.mean(ent_coef_losses))




