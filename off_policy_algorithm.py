
from typing import Any, Dict, List, Optional, Tuple, Type, TypeVar, Union
from fgsm import FGSM_v2
import numpy as np
import torch as th
import time
import sys
from gymnasium import spaces
from stable_baselines3.common.base_class import BaseAlgorithm
from stable_baselines3.common.buffers import DictReplayBuffer, ReplayBuffer
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.noise import ActionNoise, VectorizedActionNoise
from stable_baselines3.common.policies import BasePolicy
from stable_baselines3.common.save_util import load_from_pkl, save_to_pkl
from stable_baselines3.common.type_aliases import GymEnv, MaybeCallback, RolloutReturn, Schedule, TrainFreq, TrainFrequencyUnit
from stable_baselines3.common.utils import safe_mean, should_collect_more_steps, obs_as_tensor
from stable_baselines3.common.vec_env import VecEnv
from stable_baselines3.her.her_replay_buffer import HerReplayBuffer
from stable_baselines3.common.off_policy_algorithm import OffPolicyAlgorithm
SelfOffPolicyAlgorithm = TypeVar("SelfOffPolicyAlgorithm", bound="OffPolicyAlgorithm")
from copy import deepcopy

class OffPolicyDefensiveAlgorithm(OffPolicyAlgorithm):

    # TODO:这里需要重写一下父文件的_dump_logs
    def _dump_logs(self):
        assert self.ep_info_buffer is not None
        assert self.ep_success_buffer is not None

        time_elapsed = max((time.time_ns() - self.start_time) / 1e9, sys.float_info.epsilon)
        fps = int((self.num_timesteps - self._num_timesteps_at_start) / time_elapsed)
        self.logger.record("time_def/episodes", self._episode_num, exclude="tensorboard")
        if len(self.ep_info_buffer) > 0 and len(self.ep_info_buffer[0]) > 0:
            self.logger.record("rollout_def/ep_rew_mean", safe_mean([ep_info["r"] for ep_info in self.ep_info_buffer]))
            self.logger.record("rollout_def/ep_len_mean", safe_mean([ep_info["l"] for ep_info in self.ep_info_buffer]))
        self.logger.record("time_def/fps", fps)
        self.logger.record("time_def/time_elapsed", int(time_elapsed), exclude="tensorboard")
        self.logger.record("time_def/total_timesteps", self.num_timesteps, exclude="tensorboard")
        if self.use_sde:
            self.logger.record("train_def/std", (self.actor.get_std()).mean().item())

        if len(self.ep_success_buffer) > 0:
            self.logger.record("rollout_def/success_rate", safe_mean(self.ep_success_buffer))
        # Pass the number of timesteps for tensorboard
        self.logger.dump(step=self.num_timesteps)
        if safe_mean([ep_info["r"] for ep_info in self.ep_info_buffer]) >= self.max_epi_reward:
            self.save(self.best_model_path)
            self.max_epi_reward = safe_mean([ep_info["r"] for ep_info in self.ep_info_buffer])


    def _sample_action(
        self,
        learning_starts: int,
        action_noise: Optional[ActionNoise] = None,
        n_envs: int = 1,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Sample an action according to the exploration policy.
        This is either done by sampling the probability distribution of the policy,
        or sampling a random action (from a uniform distribution over the action space)
        or by adding noise to the deterministic output.

        :param action_noise: Action noise that will be used for exploration
            Required for deterministic policy (e.g. TD3). This can also be used
            in addition to the stochastic policy for SAC.
        :param learning_starts: Number of steps before learning for the warm-up phase.
        :param n_envs:
        :return: action to take in the environment
            and scaled action that will be stored in the replay buffer.
            The two differs when the action space is not normalized (bounds are not [-1, 1]).
        """
        # Select action randomly or according to policy
        # 2025-10-06 wq flag判断下面的if else

        state_eps = None
        adv_action_mask = False

        if self.num_timesteps < learning_starts and not (self.use_sde and self.use_sde_at_warmup):
            # Warmup phase
            unscaled_action = np.array([self.action_space.sample() for _ in range(n_envs)])

        else:
            # Note: when using continuous actions,
            # we assume that the policy uses tanh to scale the action
            # We use non-deterministic action in the case of SAC, for TD3, it does not matter
            assert self._last_obs is not None, "self._last_obs was not set"
            # 2025-09-26 wq 添加攻击
            # if isinstance(self._last_obs, np.ndarray):
            #     last_obs_copy = self._last_obs.copy()
            # else:
            #     last_obs_copy = deepcopy(self._last_obs)
            # 拷贝观测（兼容 numpy 和 tensor）
            if isinstance(self._last_obs, np.ndarray):
                last_obs_copy = self._last_obs.copy()
            elif isinstance(self._last_obs, th.Tensor):
                last_obs_copy = self._last_obs.clone().detach()
            else:
                last_obs_copy = deepcopy(self._last_obs)

            #  统一转换为 torch.Tensor
            last_obs_copy = th.as_tensor(last_obs_copy, device=self.device, dtype=th.float32)

            with th.no_grad():
                obs_tensor = obs_as_tensor(self._last_obs, self.device)
                if self.fni_flag:
                    actions, std, _action = self.trained_agent(obs_tensor)
                    actions = actions.detach().cpu().numpy()
                else:
                    actions, _states = self.trained_agent.predict(obs_tensor.cpu(), deterministic=True)

                actions_tensor = th.tensor(actions, device=self.device)

                # infos 可能是 dict 或 list
                if isinstance(self._last_infos, dict):
                    info0 = self._last_infos
                elif isinstance(self._last_infos, (list, tuple)) and len(self._last_infos) > 0:
                    info0 = self._last_infos[0]
                else:
                    raise ValueError(f"Invalid infos format: {type(self._last_infos)}")
                # 如果没有 attReStep 键则报错
                if 'attReStep' not in info0:
                    raise KeyError(f"'attReStep' key not found in info: {info0}")
                attReStep = float(info0['attReStep'])
                # attReStep = self._last_infos[0].get('attReStep', 0.0)
                attReStep_tensor= th.tensor(attReStep, device=self.device).reshape(1,1)

                obs_adv = th.cat([obs_tensor, attReStep_tensor, actions_tensor], dim=-1)  # -> (batch, 28)
                # obs_tensor[:, -1] = actions_tensor.squeeze(-1)
                # TODO:这里是否固定攻击者动作的输出
                adv_action, _ = self.trained_adv.predict(obs_adv.cpu(), deterministic=True)

            adv_action_mask = (adv_action[:, 0] > 0) & (obs_adv[:, -2].cpu().numpy() > 0)
            if adv_action_mask:
                adv_action_eps = adv_action[:, 1]
                state_eps = FGSM_v2(adv_action_eps, victim_agent=self.trained_agent, epsilon=self.attack_eps,
                                   last_state=last_obs_copy, device=self.device)

                if isinstance(state_eps, th.Tensor):
                    state_eps = state_eps.detach().cpu().numpy()
                unscaled_action, _ = self.predict(state_eps, deterministic=False)

            else:
                unscaled_action, _ = self.predict(self._last_obs, deterministic=False)

        # Rescale the action from [low, high] to [-1, 1]
        if isinstance(self.action_space, spaces.Box):
            scaled_action = self.policy.scale_action(unscaled_action)

            # Add noise to the action (improve exploration)
            if action_noise is not None:
                scaled_action = np.clip(scaled_action + action_noise(), -1, 1)

            # We store the scaled action in the buffer
            buffer_action = scaled_action
            action = self.policy.unscale_action(scaled_action)
        else:
            # Discrete case, no need to normalize or clip
            buffer_action = unscaled_action
            action = buffer_action
        if adv_action_mask:
            self.logger.record("def_action_record/action_def_old", actions_tensor.item())
            self.logger.record("def_action_record/action_adv", adv_action_eps.item())
            self.logger.record("def_action_record/action_def_new", buffer_action.item())

        return adv_action_mask, buffer_action, state_eps


    def _store_transition(
        self,
        replay_buffer: ReplayBuffer,
        buffer_action: np.ndarray,
        new_obs: Union[np.ndarray, Dict[str, np.ndarray]],
        reward: np.ndarray,
        dones: np.ndarray,
        infos: List[Dict[str, Any]],
        obs_eps: Optional[np.ndarray] = None,
        obs_is_per: Union[np.ndarray, bool] = False,
    ) -> None:
        """
        Store transition in the replay buffer.
        We store the normalized action and the unnormalized observation.
        It also handles terminal observations (because VecEnv resets automatically).

        :param replay_buffer: Replay buffer object where to store the transition.
        :param buffer_action: normalized action
        :param new_obs: next observation in the current episode
            or first observation of the episode (when dones is True)
        :param reward: reward for the current transition
        :param dones: Termination signal
        :param infos: List of additional information about the transition.
            It may contain the terminal observations and information about timeout.
        """
        # Store only the unnormalized version
        if self._vec_normalize_env is not None:
            new_obs_ = self._vec_normalize_env.get_original_obs()
            reward_ = self._vec_normalize_env.get_original_reward()
        else:
            # Avoid changing the original ones
            self._last_original_obs, new_obs_, reward_ = self._last_obs, new_obs, reward

        # Avoid modification by reference
        next_obs = deepcopy(new_obs_)
        # As the VecEnv resets automatically, new_obs is already the
        # first observation of the next episode
        for i, done in enumerate(dones):
            if done and infos[i].get("terminal_observation") is not None:
                if isinstance(next_obs, dict):
                    next_obs_ = infos[i]["terminal_observation"]
                    # VecNormalize normalizes the terminal observation
                    if self._vec_normalize_env is not None:
                        next_obs_ = self._vec_normalize_env.unnormalize_obs(next_obs_)
                    # Replace next obs for the correct envs
                    for key in next_obs.keys():
                        next_obs[key][i] = next_obs_[key]
                else:
                    next_obs[i] = infos[i]["terminal_observation"]
                    # VecNormalize normalizes the terminal observation
                    if self._vec_normalize_env is not None:
                        next_obs[i] = self._vec_normalize_env.unnormalize_obs(next_obs[i, :])

        replay_buffer.add(
            self._last_original_obs,  # type: ignore[arg-type]
            next_obs,  # type: ignore[arg-type]
            buffer_action,
            reward_,
            dones,
            infos,
            obs_eps,
            obs_is_per,
        )

        self._last_obs = new_obs
        # Save the unnormalized observation
        if self._vec_normalize_env is not None:
            self._last_original_obs = new_obs_


    def collect_rollouts(
        self,
        env: VecEnv,
        callback: BaseCallback,
        train_freq: TrainFreq,
        replay_buffer: ReplayBuffer,
        action_noise: Optional[ActionNoise] = None,
        learning_starts: int = 0,
        log_interval: Optional[int] = None,
    ) -> RolloutReturn:
        """
        Collect experiences and store them into a ``ReplayBuffer``.

        :param env: The training environment
        :param callback: Callback that will be called at each step
            (and at the beginning and end of the rollout)
        :param train_freq: How much experience to collect
            by doing rollouts of current policy.
            Either ``TrainFreq(<n>, TrainFrequencyUnit.STEP)``
            or ``TrainFreq(<n>, TrainFrequencyUnit.EPISODE)``
            with ``<n>`` being an integer greater than 0.
        :param action_noise: Action noise that will be used for exploration
            Required for deterministic policy (e.g. TD3). This can also be used
            in addition to the stochastic policy for SAC.
        :param learning_starts: Number of steps before learning for the warm-up phase.
        :param replay_buffer:
        :param log_interval: Log data every ``log_interval`` episodes
        :return:
        """
        # Switch to eval mode (this affects batch norm / dropout)
        self.policy.set_training_mode(False)

        num_collected_steps, num_collected_episodes = 0, 0

        assert isinstance(env, VecEnv), "You must pass a VecEnv"
        assert train_freq.frequency > 0, "Should at least collect one step or episode."

        if env.num_envs > 1:
            assert train_freq.unit == TrainFrequencyUnit.STEP, "You must use only one env when doing episodic training."

        if self.use_sde:
            self.actor.reset_noise(env.num_envs)

        callback.on_rollout_start()
        continue_training = True
        while should_collect_more_steps(train_freq, num_collected_steps, num_collected_episodes):
            if self.use_sde and self.sde_sample_freq > 0 and num_collected_steps % self.sde_sample_freq == 0:
                # Sample a new noise matrix
                self.actor.reset_noise(env.num_envs)

            # Select action randomly or according to policy
            adv_action_mask, buffer_actions, obs_eps = self._sample_action(learning_starts, action_noise, env.num_envs)
            # Rescale and perform action
            env.set_attr('adv_action_mask', adv_action_mask)
            new_obs, rewards, dones, infos = env.step(buffer_actions)

            self.num_timesteps += env.num_envs
            num_collected_steps += 1

            # Give access to local variables
            callback.update_locals(locals())
            # Only stop training if return value is False, not when it is None.
            if not callback.on_step():
                return RolloutReturn(num_collected_steps * env.num_envs, num_collected_episodes, continue_training=False)

            # Retrieve reward and episode length if using Monitor wrapper
            self._update_info_buffer(infos, dones)

            # Store data in replay buffer (normalized action and unnormalized observation)

            self._store_transition(replay_buffer, buffer_actions, new_obs, rewards, dones, infos, obs_eps, adv_action_mask)  # type: ignore[arg-type]

            self._last_infos = infos

            self._update_current_progress_remaining(self.num_timesteps, self._total_timesteps)

            # For DQN, check if the target network should be updated
            # and update the exploration schedule
            # For SAC/TD3, the update is dones as the same time as the gradient update
            # see https://github.com/hill-a/stable-baselines/issues/900
            self._on_step()

            for idx, done in enumerate(dones):
                if done:
                    # Update stats
                    num_collected_episodes += 1
                    self._episode_num += 1

                    if action_noise is not None:
                        kwargs = dict(indices=[idx]) if env.num_envs > 1 else {}
                        action_noise.reset(**kwargs)

                    # Log training infos
                    if log_interval is not None and self._episode_num % log_interval == 0:
                        self._dump_logs()
        callback.on_rollout_end()

        return RolloutReturn(num_collected_steps * env.num_envs, num_collected_episodes, continue_training)


class OffPolicyBaseAlgorithm(OffPolicyAlgorithm):

    def _dump_logs(self):
        assert self.ep_info_buffer is not None
        assert self.ep_success_buffer is not None

        time_elapsed = max((time.time_ns() - self.start_time) / 1e9, sys.float_info.epsilon)
        fps = int((self.num_timesteps - self._num_timesteps_at_start) / time_elapsed)
        self.logger.record("time_def/episodes", self._episode_num, exclude="tensorboard")
        if len(self.ep_info_buffer) > 0 and len(self.ep_info_buffer[0]) > 0:
            self.logger.record("rollout_def/ep_rew_mean", safe_mean([ep_info["r"] for ep_info in self.ep_info_buffer]))
            self.logger.record("rollout_def/ep_len_mean", safe_mean([ep_info["l"] for ep_info in self.ep_info_buffer]))
        self.logger.record("time_def/fps", fps)
        self.logger.record("time_def/time_elapsed", int(time_elapsed), exclude="tensorboard")
        self.logger.record("time_def/total_timesteps", self.num_timesteps, exclude="tensorboard")
        if self.use_sde:
            self.logger.record("train_def/std", (self.actor.get_std()).mean().item())

        if len(self.ep_success_buffer) > 0:
            self.logger.record("rollout_def/success_rate", safe_mean(self.ep_success_buffer))
        # Pass the number of timesteps for tensorboard
        self.logger.dump(step=self.num_timesteps)

