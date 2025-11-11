from on_policy_algorithm import OnPolicyAdversarialAlgorithm
from stable_baselines3.common.utils import explained_variance
from stable_baselines3 import PPO, SAC
import torch as th
import numpy as np
from gymnasium import spaces

class AdversarialPPO(OnPolicyAdversarialAlgorithm, PPO):
    def __init__(self, custom_args, best_model_path, age_model_path, *args, **kwargs):
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

        if custom_args.algo == "RARL":
            self.trained_agent = SAC.load(age_model_path, device=self.device)



        self.base_cost = custom_args.base_cost

        # # Instantiate the agent
        # if self.best_model:
        #     model_path = os.path.join(self.path_def, self.algo, self.env_name, 'best_model/best_model')
        # else:
        #     model_path = os.path.join(self.path_def, self.algo, self.env_name, 'lunar')
        # if self.algo == 'PPO':
        #     self.trained_agent = PPO.load(model_path, device=self.device)
        # elif self.algo == 'SAC':
        #     self.trained_agent = SAC.load(model_path, device=self.device)
        # elif self.algo == 'TD3':
        #     self.trained_agent = TD3.load(model_path, device=self.device)
        # elif self.algo == 'RA_SAC':
        #     self.trained_agent = SAC.load(model_path, device=self.device)
        # elif self.algo in ('FNI', 'DARRL'):
        #     if self.env_name == 'TrafficEnv5-v0':
        #         state_dim = 29
        #         action_dim = 2
        #     else:
        #         state_dim = 26
        #         action_dim = 1
        #     if self.algo == 'FNI':
        #         # 创建一个新的 Actor 实例
        #         self.trained_agent = FniNet(state_dim, action_dim)
        #         self.trained_agent.load_state_dict(
        #             th.load(os.path.join(self.path, self.env_name, self.algo, self.fni_model_path) + '.pth',
        #                     weights_only=True))
        #         self.trained_agent.to(self.device)
        #         self.trained_agent.eval()
        #     elif self.algo == 'DARRL':
        #         self.trained_agent = FniNet(state_dim, action_dim)
        #         self.trained_agent.load_state_dict(
        #             th.load(os.path.join(self.path, self.env_name, self.algo, self.fni_model_path) + '.pth',
        #                     weights_only=True))
        #         self.trained_agent.to(self.device)
        #         self.trained_agent.eval()
        self.max_epi_reward = np.inf
        self.best_model_path = best_model_path

        # Get customized parameters
        self.fni_flag = True if self.algo in ('FNI', 'DARRL', 'IGCARL') else False



    def learn(self, total_timesteps, callback=None, log_interval=1,
              tb_log_name='PPO', reset_num_timesteps=True, progress_bar=False):
        return super().learn(
            total_timesteps=total_timesteps,
            callback=callback,
            log_interval=log_interval,
            tb_log_name=tb_log_name,
            reset_num_timesteps=reset_num_timesteps,
            progress_bar=progress_bar,
        )


class AdversarialDecouplePPO(AdversarialPPO):

    def learn(self, total_timesteps, callback=None, log_interval=1,
              tb_log_name='PPO', reset_num_timesteps=True, progress_bar=False,  *args, **kwargs):

        return super().learn(
            total_timesteps=total_timesteps,
            callback=callback,
            log_interval=log_interval,
            tb_log_name=tb_log_name,
            reset_num_timesteps=reset_num_timesteps,
            progress_bar=progress_bar,
            *args, **kwargs
        )

    def _setup_model(self):
        super()._setup_model()
        # Use the custom rollout buffer if provided, otherwise, use the default one

    def train(self):
        """
        Update policy using the currently gathered rollout buffer.
        divide loss to swi_loss and lur_loss, similar to the setting of 'Attacking Deep Reinforcement Learning with  Decoupled Adversarial Policy'
        """
        # Switch to train mode (this affects batch norm / dropout)
        self.policy.set_training_mode(True)
        # Update optimizer learning rate
        self._update_learning_rate(self.policy.optimizer)
        # Compute current clip range
        clip_range = self.clip_range(self._current_progress_remaining)  # type: ignore[operator]
        # Optional: clip range for the value function
        if self.clip_range_vf is not None:
            clip_range_vf = self.clip_range_vf(self._current_progress_remaining)  # type: ignore[operator]

        entropy_losses = []
        pg_losses, value_losses = [], []
        prob_losses = []
        clip_fractions = []

        continue_training = True
        # train for n_epochs epochs
        for epoch in range(self.n_epochs):
            approx_kl_divs = []
            # Do a complete pass on the rollout buffer
            for rollout_data in self.rollout_buffer.get(self.batch_size):
                actions = rollout_data.actions
                if isinstance(self.action_space, spaces.Discrete):
                    # Convert discrete action from float to long
                    actions = rollout_data.actions.long().flatten()

                values, log_prob, entropy = self.policy.evaluate_actions(rollout_data.observations, actions)
                values = values.flatten()
                # Normalize advantage
                advantages = rollout_data.advantages
                # Normalization does not make sense if mini batchsize == 1, see GH issue #325
                if self.normalize_advantage and len(advantages) > 1:
                    advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

                # 分离 a1 和 a2 的动作
                actions_a1 = actions[:, 0]

                # print('values', values, 'log prob', log_prob, 'entropy', entropy)
                # print('actions are ', actions)
                # print('actions', actions)
                distribution = self.policy.get_distribution(rollout_data.observations)
                log_prob_joint = distribution.distribution.log_prob(actions)
                # print('log prob joint ', log_prob_joint)
                # print('old log prob ', rollout_data.old_log_prob)
                # print('log_prob', log_prob)
                # 分离 log_prob
                log_prob_a1 = log_prob_joint[:, 0]  # [batch_size]
                log_prob_a2 = log_prob_joint[:, 1]  # [batch_size]
                # ratio between old and new policy for a1 and a2
                ratio_a1 = th.exp(log_prob_a1 - rollout_data.old_log_prob[:, 0])
                ratio_a2 = th.exp(log_prob_a2 - rollout_data.old_log_prob[:, 1])

                # clipped surrogate loss for a1 and a2
                policy_loss_1_a1 = advantages * ratio_a1
                policy_loss_2_a1 = advantages * th.clamp(ratio_a1, 1 - clip_range, 1 + clip_range)
                policy_loss_a1 = -th.min(policy_loss_1_a1, policy_loss_2_a1).mean()

                policy_loss_1_a2 = advantages * ratio_a2
                policy_loss_2_a2 = advantages * th.clamp(ratio_a2, 1 - clip_range, 1 + clip_range)
                surrogate_a2 = th.min(policy_loss_1_a2, policy_loss_2_a2)
                # ---------------------

                mask = (actions_a1 >= 0).float().to(actions.device)
                masked_surrogate_a2 = surrogate_a2 * mask
                mask_count = mask.sum()
                if mask_count.item() > 0:
                    policy_loss_a2 = -masked_surrogate_a2.sum() / mask_count
                    clip_fraction_a2 = (((th.abs(ratio_a2 - 1) > clip_range).float() * mask).sum() / mask_count).item()
                else:
                    policy_loss_a2 = th.zeros(1, device=actions.device, dtype=policy_loss_a1.dtype).squeeze()
                    clip_fraction_a2 = 0.0

                pg_losses.append([policy_loss_a1.item(), policy_loss_a2.item()])
                clip_fraction_a1 = th.mean((th.abs(ratio_a1 - 1) > clip_range).float()).item()
                clip_fractions.append((clip_fraction_a1 + clip_fraction_a2) / 2)

                policy_loss = policy_loss_a1 + policy_loss_a2

                if self.clip_range_vf is None:
                    # No clipping
                    values_pred = values
                else:
                    # Clip the difference between old and new value
                    # NOTE: this depends on the reward scaling
                    values_pred = rollout_data.old_values + th.clamp(
                        values - rollout_data.old_values, -clip_range_vf, clip_range_vf
                    )
                # Value loss using the TD(gae_lambda) target
                value_loss = th.nn.functional.mse_loss(rollout_data.returns, values_pred)
                value_losses.append(value_loss.item())

                # Entropy loss favor exploration
                if entropy is None:
                    # Approximate entropy when no analytical form
                    entropy_loss = -th.mean(-log_prob)
                else:
                    entropy_loss = -th.mean(entropy)

                entropy_losses.append(entropy_loss.item())

                loss = (policy_loss + self.ent_coef * entropy_loss + self.vf_coef * value_loss)

                # Calculate approximate form of reverse KL Divergence for early stopping
                # see issue #417: https://github.com/DLR-RM/stable-baselines3/issues/417
                # and discussion in PR #419: https://github.com/DLR-RM/stable-baselines3/pull/419
                # and Schulman blog: http://joschu.net/blog/kl-approx.html
                with th.no_grad():
                    log_ratio = log_prob_joint - rollout_data.old_log_prob
                    approx_kl_div = th.mean((th.exp(log_ratio) - 1) - log_ratio).cpu().numpy()
                    approx_kl_divs.append(approx_kl_div)

                if self.target_kl is not None and approx_kl_div > 1.5 * self.target_kl:
                    continue_training = False
                    if self.verbose >= 1:
                        print(f"Early stopping at step {epoch} due to reaching max kl: {approx_kl_div:.2f}")
                    break

                # Optimization step
                self.policy.optimizer.zero_grad()
                loss.backward()
                # Clip grad norm
                th.nn.utils.clip_grad_norm_(self.policy.parameters(), self.max_grad_norm)
                self.policy.optimizer.step()

            self._n_updates += 1
            if not continue_training:
                break

        explained_var = explained_variance(self.rollout_buffer.values.flatten(), self.rollout_buffer.returns.flatten())

        # Logs
        self.logger.record("train_adv/entropy_loss", np.mean(entropy_losses))
        self.logger.record("train_adv/policy_gradient_loss", np.mean(pg_losses))
        self.logger.record("train_adv/value_loss", np.mean(value_losses))
        self.logger.record("train_adv/approx_kl", np.mean(approx_kl_divs))
        self.logger.record("train_adv/clip_fraction", np.mean(clip_fractions))
        self.logger.record("train_adv/loss", loss.item())
        self.logger.record("train_adv/explained_variance", explained_var)
        if hasattr(self.policy, "log_std"):
            self.logger.record("train_adv/std", th.exp(self.policy.log_std).mean().item())

        self.logger.record("train_adv/n_updates", self._n_updates, exclude="tensorboard")
        self.logger.record("train_adv/clip_range", clip_range)
        if self.clip_range_vf is not None:
            self.logger.record("train_adv/clip_range_vf", clip_range_vf)