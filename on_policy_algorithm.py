import torch as th
from stable_baselines3.common.on_policy_algorithm import OnPolicyAlgorithm
from stable_baselines3.common.utils import obs_as_tensor, safe_mean
import time
import sys
from fgsm import FGSM_v2
from gymnasium import spaces
import numpy as np
from buffer import PaddingRolloutBuffer, DecouplePaddingRolloutBuffer

class OnPolicyAdversarialAlgorithm(OnPolicyAlgorithm):
    """
    Rewrite the collect_rollouts class of OnPolicyAlgorithm to support adversarial training.
    """
    def collect_rollouts(self, env, callback, rollout_buffer, n_rollout_steps):
        """
        Collect experiences using the current policy and fill a ``RolloutBuffer``.
        The term rollout here refers to the model-free notion and should not
        be used with the concept of rollout used in model-based RL or planning.
        :param env: The training environment
        :param callback: Callback that will be called at each step
            (and at the beginning and end of the rollout)
        :param rollout_buffer: Buffer to fill with rollouts
        :param n_rollout_steps: Number of experiences to collect per environment
        :return: True if function returned with at least `n_rollout_steps`
            collected, False if callback terminated rollout prematurely.
        """
        assert self._last_obs is not None, "No previous observation was provided"
        # Switch to eval mode (this affects batch norm / dropout)
        self.policy.set_training_mode(False)

        n_steps = 0
        rollout_buffer.reset()
        # Sample new weights for the state dependent exploration
        if self.use_sde:
            self.policy.reset_noise(env.num_envs)

        callback.on_rollout_start()

        while n_steps < n_rollout_steps:
            if self.use_sde and self.sde_sample_freq > 0 and n_steps % self.sde_sample_freq == 0:
                # Sample a new noise matrix
                self.policy.reset_noise(env.num_envs)

            with th.no_grad():
                # Get adv action
                # Convert to pytorch tensor or to TensorDict
                obs_tensor = obs_as_tensor(self._last_obs, self.device)
                if self.fni_flag:
                    actions, std, _action = self.trained_agent(obs_tensor[:, :-2])
                    actions = actions.detach().cpu().numpy()
                else:
                    actions, _states = self.trained_agent.predict(obs_tensor[:, :-2].cpu(), deterministic=True)
                actions_tensor = th.tensor(actions, device=self.device)
                obs_tensor[:, -1] = actions_tensor.squeeze(-1)
                adv_actions, adv_values, adv_log_probs = self.policy(obs_tensor)
                if self.decouple:
                    # print('adv actions ', adv_actions)
                    distribution = self.policy.get_distribution(obs_tensor)
                    adv_log_probs = distribution.distribution.log_prob(adv_actions)

            adv_actions = adv_actions.cpu().numpy()
            # Rescale and perform action
            clipped_adv_actions = adv_actions
            if isinstance(self.action_space, spaces.Box):
                if self.policy.squash_output:
                    # Unscale the actions to match env bounds
                    # if they were previously squashed (scaled in [-1, 1])
                    clipped_adv_actions = self.policy.unscale_action(clipped_adv_actions)
                else:
                    # Otherwise, clip the actions to avoid out of bound error
                    # as we are sampling from an unbounded Gaussian distribution
                    clipped_adv_actions = np.clip(adv_actions, self.action_space.low, self.action_space.high)

            # Get adv action mask
            if self.unlimited_attack:
                adv_action_mask = np.ones_like(clipped_adv_actions)
            else:
                adv_action_mask = (clipped_adv_actions[:, 0] > 0) & (obs_tensor[:, -2].cpu().numpy() > 0)


            if adv_action_mask:
                self.logger.record("adv_action_record/action_def_old", actions_tensor.item())

            # Generate perturbation into observations to get adv_obs
            final_actions = self.attack_process(obs_tensor, adv_action_mask, clipped_adv_actions, actions)

            new_obs, rewards, dones, infos = env.step(final_actions)

            if isinstance(infos, dict):
                info1 = infos
            elif isinstance(infos, (list, tuple)) and len(infos) > 0:
                info1 = infos[0]
            else:
                raise ValueError(f"Invalid infos format: {type(infos)}")

            curr_step = int(info1.get("step", None))

            if curr_step is None:
                raise ValueError(f"Invalid step format: {curr_step}")
            # 2025-10-21 wq 对敌手的reward添加惩罚
            attack_start = 12
            attack_end = 18

            if not adv_action_mask:
                if curr_step >= attack_start:
                    re_c = (curr_step - attack_start) / (attack_end -  attack_start)
                    rewards -= re_c

            if adv_action_mask:
                if self.base_cost != 0:
                    base_cost = self.base_cost
                    k = 1.0
                    att_remain = new_obs[:, -2]
                    # attack_cost = base_cost * (1.0 + k * (1.0 - att_remain))
                    attack_cost = 1.0 - att_remain
                    rewards = rewards - attack_cost
            # breakpoint()

            # Get next origin action according next state and insert into next_obs
            with th.no_grad():
                next_obs_tensor = obs_as_tensor(new_obs, self.device)
                if self.fni_flag:
                    actions, std, _action = self.trained_agent(next_obs_tensor[:, :-2])
                    actions = actions.cpu().numpy()
                else:
                    actions, _states = self.trained_agent.predict(next_obs_tensor[:, :-2].cpu(), deterministic=True)
                actions_tensor = th.tensor(actions, device=self.device)

                # Update next_obs_tensor
                next_obs_tensor[:, -1] = actions_tensor.squeeze(-1)
                new_obs = next_obs_tensor.detach().cpu().numpy()

            self.num_timesteps += env.num_envs

            # Give access to local variables
            callback.update_locals(locals())
            if not callback.on_step():
                return False

            self._update_info_buffer(infos, dones)


            n_steps += 1

            if isinstance(self.action_space, spaces.Discrete):
                # Reshape in case of discrete action
                adv_actions = adv_actions.reshape(-1, 1)

            # Handle timeout by bootstrapping with value function
            # see GitHub issue #633
            for idx, done in enumerate(dones):
                if (
                        done
                        and infos[idx].get("terminal_observation") is not None
                        and infos[idx].get("TimeLimit.truncated", False)
                ):
                    terminal_obs = self.policy.obs_to_tensor(infos[idx]["terminal_observation"])[0]
                    with th.no_grad():
                        terminal_value = self.policy.predict_values(terminal_obs)[0]  # type: ignore[arg-type]
                    rewards[idx] += self.gamma * terminal_value

            # TODO: whether to add trajectory padding method
            # check rollout_buffer type
            # is_padding_buffer = isinstance(rollout_buffer, (PaddingRolloutBuffer, DecouplePaddingRolloutBuffer))
            # if is_padding_buffer:
            #     # 遍历每个环境的 info，处理每条轨迹
            #     for env_idx, info in enumerate(infos):
            #         if info.get('flag', False):  # 如果这个环境提前终止（攻击成功）
            #             rollout_buffer.log_collisions(env_idx=env_idx)  # 支持按环境记录碰撞
            #             rollout_buffer.return_pos(env_idx=env_idx)  # 返回当前环境的n_steps，用于后续对齐处理

            # print('last obs is ', self._last_obs)
            # print('obs tensor is ', obs_tensor.cpu().numpy())
            rollout_buffer.add(
                obs_tensor.cpu().numpy(),  # type: ignore[arg-type]
                adv_actions,
                rewards,
                self._last_episode_starts,  # type: ignore[arg-type]
                adv_values,
                adv_log_probs,
            )
            self._last_obs = new_obs  # type: ignore[assignment]
            self._last_episode_starts = dones
            # print('last episode start is ', self._last_episode_starts)

        with th.no_grad():
            # Compute value for the last timestep
            values = self.policy.predict_values(obs_as_tensor(new_obs, self.device))  # type: ignore[arg-type]

        rollout_buffer.compute_returns_and_advantage(last_values=values, dones=dones)

        callback.update_locals(locals())

        callback.on_rollout_end()

        return True

    def attack_process(self, obs_tensor, adv_action_mask, clipped_adv_actions, actions):
        if adv_action_mask.any():
            attack_idx = np.where(adv_action_mask)[0]

            selected_states = obs_tensor[attack_idx, :-2]
            print('selected_states shape:', selected_states.shape)
            selected_adv_actions = clipped_adv_actions[attack_idx, 1]
            print('selected_adv_actions shape:', selected_adv_actions)

            if self.attack_method == 'fgsm':
                adv_state = FGSM_v2(selected_adv_actions, victim_agent=self.trained_agent,
                                    last_state=selected_states, epsilon=self.attack_eps, device=self.device)
            # elif self.attack_method == 'pgd':
            #     adv_state = PGD(selected_adv_actions, self.trained_agent, selected_states, epsilon=self.attack_eps,
            #                     device=self.device)
            # elif self.attack_method == 'cw':
            #     adv_state = cw_attack_v2(self.trained_agent, selected_states, selected_adv_actions,
            #                              epsilon=self.attack_eps)

            if self.attack_method == 'direct':
                final_action = actions.copy()
                final_action[attack_idx] = selected_adv_actions.detach().cpu().numpy() if th.is_tensor(
                    selected_adv_actions) else selected_adv_actions
            else:
                if self.fni_flag:
                    adv_action_fromState, _, _ = self.trained_agent(adv_state)
                    adv_action = adv_action_fromState.detach().cpu().numpy()
                else:
                    adv_action_fromState, _ = self.trained_agent.predict(adv_state.cpu(), deterministic=True)
                    adv_action = adv_action_fromState
            self.logger.record("adv_action_record/action_adv", selected_adv_actions.item())
            self.logger.record("adv_action_record/action_def_new", adv_action_fromState.item())
            # print('clip',clipped_adv_actions,'adv_actions:', adv_actions, 'adv_final_action', action, 'actions:', actions, 'remain attack times ', obs_tensor[:, -2].cpu().numpy())
            final_action = actions.copy()
            final_action[attack_idx] = adv_action
        else:
            final_action = actions.copy()
        # Concat final_action with adv_action_mask
        output_action = np.column_stack((final_action, adv_action_mask.astype(np.float32)))

        return output_action

    def _dump_logs(self, iteration):
        super()._dump_logs(iteration)
        if safe_mean([ep_info["r"] for ep_info in self.ep_info_buffer]) <= self.max_epi_reward:
            self.save(self.best_model_path)
            self.max_epi_reward = safe_mean([ep_info["r"] for ep_info in self.ep_info_buffer])

    def learn(
            self,
            total_timesteps: int,
            callback = None,
            log_interval: int = 1,
            tb_log_name: str = "OnPolicyAlgorithm",
            reset_num_timesteps: bool = True,
            progress_bar: bool = False,
    ) :
        iteration = 0

        total_timesteps, callback = self._setup_learn(
            total_timesteps,
            callback,
            reset_num_timesteps,
            tb_log_name,
            progress_bar,
        )

        callback.on_training_start(locals(), globals())

        assert self.env is not None

        while self.num_timesteps < total_timesteps:
            continue_training = self.collect_rollouts(self.env, callback, self.rollout_buffer, n_rollout_steps=self.n_steps)

            if not continue_training:
                break

            iteration += 1
            self._update_current_progress_remaining(self.num_timesteps, total_timesteps)

            # Display training infos
            if log_interval is not None and iteration % log_interval == 0:
                assert self.ep_info_buffer is not None
                time_elapsed = max((time.time_ns() - self.start_time) / 1e9, sys.float_info.epsilon)
                fps = int((self.num_timesteps - self._num_timesteps_at_start) / time_elapsed)
                self.logger.record("time_adv/iterations", iteration, exclude="tensorboard")
                if len(self.ep_info_buffer) > 0 and len(self.ep_info_buffer[0]) > 0:
                    self.logger.record("rollout_adv/ep_rew_mean", safe_mean([ep_info["r"] for ep_info in self.ep_info_buffer]))
                    self.logger.record("rollout_adv/ep_len_mean", safe_mean([ep_info["l"] for ep_info in self.ep_info_buffer]))
                self.logger.record("time_adv/fps", fps)
                self.logger.record("time_adv/time_elapsed", int(time_elapsed), exclude="tensorboard")
                self.logger.record("time_adv/total_timesteps", self.num_timesteps, exclude="tensorboard")
                self.logger.dump(step=self.num_timesteps)
                self._dump_logs(iteration)

            self.train()

        callback.on_training_end()

        return self




class OnPolicyDefenseAlgorithm(OnPolicyAlgorithm):
    """
    Rewrite the collect_rollouts class of OnPolicyAlgorithm to support adversarial training.
    """
    def collect_rollouts(self, env, callback, rollout_buffer, n_rollout_steps):
        """
        Collect experiences using the current policy and fill a ``RolloutBuffer``.
        The term rollout here refers to the model-free notion and should not
        be used with the concept of rollout used in model-based RL or planning.
        :param env: The training environment
        :param callback: Callback that will be called at each step
            (and at the beginning and end of the rollout)
        :param rollout_buffer: Buffer to fill with rollouts
        :param n_rollout_steps: Number of experiences to collect per environment
        :return: True if function returned with at least `n_rollout_steps`
            collected, False if callback terminated rollout prematurely.
        """
        assert self._last_obs is not None, "No previous observation was provided"
        # Switch to eval mode (this affects batch norm / dropout)
        self.policy.set_training_mode(False)

        n_steps = 0
        rollout_buffer.reset()
        # Sample new weights for the state dependent exploration
        if self.use_sde:
            self.policy.reset_noise(env.num_envs)

        callback.on_rollout_start()

        while n_steps < n_rollout_steps:
            if self.use_sde and self.sde_sample_freq > 0 and n_steps % self.sde_sample_freq == 0:
                # Sample a new noise matrix
                self.policy.reset_noise(env.num_envs)

            with th.no_grad():
                # Get adv action
                # Convert to pytorch tensor or to TensorDict
                obs_tensor = obs_as_tensor(self._last_obs, self.device)
                # 获取维度1——剩余攻击次数
                remain_attacks = env.get_remain_attack_times()
                # TODO: 验证使用当前轮agent和上一轮固定的agent的区别
                # 获取维度2——agent原始动作维度
                if self.current_agent_tag:
                    origin_actions, _states = self.policy.predict(obs_tensor.cpu(), deterministic=True)
                else:
                    origin_actions, _states = self.last_agent.predict(obs_tensor.cpu(), deterministic=True)
                # print('*******origin_actions', origin_actions)
                remain_attack_tensor = th.tensor(remain_attacks, device=self.device, dtype=obs_tensor.dtype).unsqueeze(
                    -1)  # (n_envs, 1)
                agent_action_tensor = th.tensor(origin_actions, device=self.device, dtype=obs_tensor.dtype)  # (n_envs, 1)
                # 拼接成敌手用的obs (n_envs, 28)
                # print("obs_tensor shape:", obs_tensor.shape)
                # print("remain_attack_tensor shape:", remain_attack_tensor.shape)
                # print("agent_action_tensor shape:", agent_action_tensor.shape)

                adv_obs_tensor = th.cat([obs_tensor, remain_attack_tensor, agent_action_tensor], dim=-1)
            adv_actions, _adv_states = self.trained_adv_agent.predict(adv_obs_tensor.cpu(), deterministic=True)

            # adv_actions = adv_actions.cpu().numpy()
            # # Rescale and perform action
            # clipped_adv_actions = adv_actions
            # if isinstance(self.action_space, spaces.Box):
            #     if self.policy.squash_output:
            #         # Unscale the actions to match env bounds
            #         # if they were previously squashed (scaled in [-1, 1])
            #         clipped_adv_actions = self.policy.unscale_action(clipped_adv_actions)
            #     else:
            #         # Otherwise, clip the actions to avoid out of bound error
            #         # as we are sampling from an unbounded Gaussian distribution
            #         clipped_adv_actions = np.clip(adv_actions, self.action_space.low, self.action_space.high)

            # Get adv action mask
            adv_action_mask = (adv_actions[:, 0] > 0) & (remain_attack_tensor.squeeze(-1).cpu().numpy() > 0)

            # Generate perturbation into observations to get adv_obs
            final_adv_states = self.attack_process(obs_tensor, adv_action_mask, adv_actions)
            with th.no_grad():
                actions, action_value, action_log_probs = self.policy(final_adv_states)
            # print('actions is', actions)
            actions = actions.cpu().numpy()

            # Rescale and perform action
            clipped_actions = actions

            if isinstance(self.action_space, spaces.Box):
                if self.policy.squash_output:
                    # Unscale the actions to match env bounds
                    # if they were previously squashed (scaled in [-1, 1])
                    clipped_actions = self.policy.unscale_action(clipped_actions)
                else:
                    # Otherwise, clip the actions to avoid out of bound error
                    # as we are sampling from an unbounded Gaussian distribution
                    clipped_actions = np.clip(actions, self.action_space.low, self.action_space.high)
            # print('clipped_actions is', clipped_actions)
            new_obs, rewards, dones, infos = env.step(clipped_actions)

            self.num_timesteps += env.num_envs

            # Give access to local variables
            callback.update_locals(locals())
            if not callback.on_step():
                return False

            self._update_info_buffer(infos, dones)


            n_steps += 1

            if isinstance(self.action_space, spaces.Discrete):
                # Reshape in case of discrete action
                actions = actions.reshape(-1, 1)

            # Handle timeout by bootstrapping with value function
            # see GitHub issue #633
            for idx, done in enumerate(dones):
                if (
                        done
                        and infos[idx].get("terminal_observation") is not None
                        and infos[idx].get("TimeLimit.truncated", False)
                ):
                    terminal_obs = self.policy.obs_to_tensor(infos[idx]["terminal_observation"])[0]
                    with th.no_grad():
                        terminal_value = self.policy.predict_values(terminal_obs)[0]  # type: ignore[arg-type]
                    rewards[idx] += self.gamma * terminal_value

            # TODO: whether to add trajectory padding method
            # check rollout_buffer type
            # is_padding_buffer = isinstance(rollout_buffer, (PaddingRolloutBuffer, DecouplePaddingRolloutBuffer))
            # if is_padding_buffer:
            #     # 遍历每个环境的 info，处理每条轨迹
            #     for env_idx, info in enumerate(infos):
            #         if info.get('flag', False):  # 如果这个环境提前终止（攻击成功）
            #             rollout_buffer.log_collisions(env_idx=env_idx)  # 支持按环境记录碰撞
            #             rollout_buffer.return_pos(env_idx=env_idx)  # 返回当前环境的n_steps，用于后续对齐处理

            # print('last obs is ', self._last_obs)
            # print('obs tensor is ', obs_tensor.cpu().numpy())
            rollout_buffer.add(
                final_adv_states.cpu().numpy(),
                # self._last_obs,  # type: ignore[arg-type]
                actions,
                rewards,
                self._last_episode_starts,  # type: ignore[arg-type]
                action_value,
                action_log_probs,
            )
            self._last_obs = new_obs  # type: ignore[assignment]
            self._last_episode_starts = dones
            # print('last episode start is ', self._last_episode_starts)

        with th.no_grad():
            # Compute value for the last timestep
            values = self.policy.predict_values(obs_as_tensor(new_obs, self.device))  # type: ignore[arg-type]

        rollout_buffer.compute_returns_and_advantage(last_values=values, dones=dones)

        callback.update_locals(locals())

        callback.on_rollout_end()

        return True

    def attack_process(self, obs_tensor, adv_action_mask, clipped_adv_actions):
        final_obs_tensor = obs_tensor.clone()
        if adv_action_mask.any():
            attack_idx = np.where(adv_action_mask)[0]

            selected_states = obs_tensor[attack_idx]
            print('selected_states shape:', selected_states.shape)
            selected_adv_actions = clipped_adv_actions[attack_idx, 1]
            print('selected_adv_actions shape:', selected_adv_actions)

            if self.current_agent_tag:
                victim_agent = self.policy
            else:
                victim_agent = self.last_agent.policy
            if self.attack_method == 'fgsm':
                adv_state = FGSM_v2(selected_adv_actions, victim_agent=victim_agent, last_state=selected_states,
                                    epsilon=self.attack_eps, device=self.device)
            # elif self.attack_method == 'pgd':
            #     adv_state = PGD(selected_adv_actions, victim_agent, selected_states, epsilon=self.attack_eps,
            #                     device=self.device)
            # elif self.attack_method == 'cw':
            #     adv_state = cw_attack_v2(victim_agent, selected_states, selected_adv_actions, epsilon=self.attack_eps)
            final_obs_tensor[attack_idx] = adv_state
        return final_obs_tensor

    def _dump_logs(self, iteration):
        super()._dump_logs(iteration)
        if safe_mean([ep_info["r"] for ep_info in self.ep_info_buffer]) >= self.max_epi_reward:
            self.save(self.best_model_path)
            self.max_epi_reward = safe_mean([ep_info["r"] for ep_info in self.ep_info_buffer])