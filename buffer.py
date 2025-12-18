from stable_baselines3.common.type_aliases import RolloutBufferSamples
from stable_baselines3.common.buffers import RolloutBuffer, ReplayBuffer
from gymnasium import spaces
import numpy as np
from typing import NamedTuple, Optional, Union
import torch as th
import warnings

try:
    # Check memory used by replay buffer when possible
    import psutil
except ImportError:
    psutil = None

class NewRolloutBufferSamples(NamedTuple):
    observations: th.Tensor
    actions: th.Tensor
    old_values: th.Tensor
    old_log_prob: th.Tensor
    advantages: th.Tensor
    returns: th.Tensor


class ReplayBufferSamples(NamedTuple):
    observations: th.Tensor
    actions: th.Tensor
    next_observations: th.Tensor
    dones: th.Tensor
    rewards: th.Tensor
    observations_eps: th.Tensor
    obs_is_perturbed: th.Tensor


class DecoupleRolloutBuffer(RolloutBuffer):
    """
    Rollout buffer with only decouple policy update
    """

    def __init__(self, buffer_size, observation_space, action_space, device, gamma=0.99, gae_lambda=1, n_envs=1):
        super().__init__(buffer_size, observation_space, action_space, device, gamma, gae_lambda, n_envs)

    def add(self, obs, action, reward, episode_start, value, log_prob):
        if len(log_prob.shape) == 0:
            # Reshape 0-d tensor to avoid error
            log_prob = log_prob.reshape(-1, self.action_dim)

        # Reshape needed when using multiple envs with discrete observations
        # as numpy cannot broadcast (n_discrete,) to (n_discrete, 1)
        if isinstance(self.observation_space, spaces.Discrete):
            obs = obs.reshape((self.n_envs, *self.obs_shape))

        # Reshape to handle multi-dim and discrete action spaces, see GH #970 #1392
        action = action.reshape((self.n_envs, self.action_dim))

        self.observations[self.pos] = np.array(obs)
        self.actions[self.pos] = np.array(action)
        self.rewards[self.pos] = np.array(reward)
        self.episode_starts[self.pos] = np.array(episode_start)
        self.values[self.pos] = value.clone().cpu().numpy().flatten()
        self.log_probs[self.pos] = np.array(log_prob.clone().cpu())
        self.pos += 1
        if self.pos == self.buffer_size:
            self.full = True

    def return_pos(self):
        return self.pos

    def reset(self):
        super().reset()
        self.current_episode_length = 0
        self.log_probs = np.zeros((self.buffer_size, self.n_envs, self.action_dim), dtype=np.float32)

    def get(self, batch_size):
        assert self.full, ""
        indices = np.random.permutation(self.buffer_size * self.n_envs)
        # Prepare the data
        if not self.generator_ready:
            _tensor_names = [
                "observations",
                "actions",
                "values",
                "log_probs",
                "advantages",
                "returns"
            ]

            for tensor in _tensor_names:
                self.__dict__[tensor] = self.swap_and_flatten(self.__dict__[tensor])
            self.generator_ready = True

        # Return everything, don't create minibatches
        if batch_size is None:
            batch_size = self.buffer_size * self.n_envs

        start_idx = 0
        while start_idx < self.buffer_size * self.n_envs:
            yield self._get_samples(indices[start_idx : start_idx + batch_size])
            start_idx += batch_size

    def _get_samples(self, batch_inds, env=None):
        data = (
            self.observations[batch_inds],
            self.actions[batch_inds],
            self.values[batch_inds].flatten(),
            self.log_probs[batch_inds],
            self.advantages[batch_inds].flatten(),
            self.returns[batch_inds].flatten(),
        )
        return NewRolloutBufferSamples(*tuple(map(self.to_torch, data)))



class DualReplayBufferDefender(ReplayBuffer):
    """
    一个完全基于 PyTorch Tensor 的、在指定设备上运行的双重回放缓冲区。
    """

    def __init__(
            self,
            buffer_size: int,
            observation_space: spaces.Space,
            action_space: spaces.Space,
            device: Union[th.device, str],
            n_envs: int = 1,
            optimize_memory_usage: bool = False,
            adv_sample_ratio: float = 0.5,
            handle_timeout_termination: bool = True,
    ):
        super().__init__(buffer_size, observation_space, action_space, device, n_envs=n_envs)

        self.adv_sample_ratio = adv_sample_ratio



        adv_buffer_size = int(buffer_size * adv_sample_ratio)
        normal_buffer_size = buffer_size - adv_buffer_size

        self.normal_buffer_size = max(normal_buffer_size // n_envs, 1)
        self.adv_buffer_size = max(adv_buffer_size // n_envs, 1)

        # --- 为正常样本创建独立的Tensor ---
        self.normal_observations = th.zeros((self.normal_buffer_size, self.n_envs, *self.obs_shape), device=self.device, dtype=th.float32)
        self.normal_next_observations = th.zeros_like(self.normal_observations)
        self.normal_actions = th.zeros((self.normal_buffer_size, self.n_envs, self.action_dim), device=self.device, dtype=th.float32)
        self.normal_rewards = th.zeros((self.normal_buffer_size, self.n_envs), device=self.device, dtype=th.float32)
        self.normal_dones = th.zeros((self.normal_buffer_size, self.n_envs), device=self.device, dtype=th.float32)
        self.normal_timeouts = th.zeros((self.normal_buffer_size, self.n_envs), device=self.device, dtype=th.float32)
        # --- 为对抗样本创建独立的Tensor ---
        self.adv_observations = th.zeros((self.adv_buffer_size, self.n_envs, *self.obs_shape), device=self.device, dtype=th.float32)
        self.adv_observations_eps = th.zeros_like(self.adv_observations)
        self.adv_next_observations = th.zeros_like(self.adv_observations)
        self.adv_actions = th.zeros((self.adv_buffer_size, self.n_envs, self.action_dim), device=self.device, dtype=th.float32)
        self.adv_rewards = th.zeros((self.adv_buffer_size, self.n_envs), device=self.device, dtype=th.float32)
        self.adv_dones = th.zeros((self.adv_buffer_size, self.n_envs), device=self.device, dtype=th.float32)
        self.adv_timeouts = th.zeros((self.adv_buffer_size, self.n_envs), device=self.device, dtype=th.float32)
        self.normal_pos = 0
        self.normal_full = False
        self.adv_pos = 0
        self.adv_full = False

    def add(
            self,
            obs: np.ndarray,
            next_obs: np.ndarray,
            action: np.ndarray,
            reward: np.ndarray,
            done: np.ndarray,
            infos: list,
            obs_eps: Optional[np.ndarray] = None,
            obs_is_per: Union[np.ndarray, bool] = False,
    ) -> None:
        # 数据在进入时从 numpy 转换为 tensor，这是唯一一次转换！
        obs = self.to_torch(obs)
        next_obs = self.to_torch(next_obs)
        action = self.to_torch(action)
        reward = self.to_torch(reward)
        done = self.to_torch(done)

        # --- 核心改动：根据标志位将数据路由到正确的缓冲区 ---
        if np.any(obs_is_per):
            if obs_eps is None:
                raise ValueError("obs_eps cannot be None for a perturbed observation.")
            else:
                obs_eps = self.to_torch(obs_eps)

            self.adv_observations[self.adv_pos] = obs
            self.adv_observations_eps[self.adv_pos] = obs_eps
            self.adv_next_observations[self.adv_pos] = next_obs
            self.adv_actions[self.adv_pos] = action
            self.adv_rewards[self.adv_pos] = reward
            self.adv_dones[self.adv_pos] = done

            if self.handle_timeout_termination:
                timeouts = th.tensor([info.get("TimeLimit.truncated", False) for info in infos], device=self.device, dtype=th.float32)
                self.adv_timeouts[self.adv_pos] = timeouts


            self.adv_pos += 1
            if self.adv_pos == self.adv_buffer_size:
                self.adv_full = True
                self.adv_pos = 0
        else:
            self.normal_observations[self.normal_pos] = obs
            self.normal_next_observations[self.normal_pos] = next_obs
            self.normal_actions[self.normal_pos] = action
            self.normal_rewards[self.normal_pos] = reward
            self.normal_dones[self.normal_pos] = done
            if self.handle_timeout_termination:
                timeouts = th.tensor([info.get("TimeLimit.truncated", False) for info in infos], device=self.device, dtype=th.float32)
                self.normal_timeouts[self.normal_pos] = timeouts

            self.normal_pos += 1
            if self.normal_pos == self.normal_buffer_size:
                self.normal_full = True
                self.normal_pos = 0

    # In buffer.py, inside the DualReplayBufferDefender class

    def sample(self, batch_size: int, env=None) -> ReplayBufferSamples:
        """
        从完全基于PyTorch Tensor的双重回放缓冲区中采样元素。
        整个过程都在 self.device (例如GPU) 上完成。
        """
        # --- 核心改动：从两个缓冲区进行平衡采样 ---
        adv_samples_to_take = int(batch_size * self.adv_sample_ratio)
        normal_samples_to_take = batch_size - adv_samples_to_take

        normal_size = self.normal_buffer_size if self.normal_full else self.normal_pos
        adv_size = self.adv_buffer_size if self.adv_full else self.adv_pos

        # --- 根据可用数据调整采样数量 ---
        if adv_size == 0:
            normal_samples_to_take = batch_size
            adv_samples_to_take = 0
        elif adv_size < adv_samples_to_take:
            adv_samples_to_take = adv_size
            normal_samples_to_take = batch_size - adv_samples_to_take

        # 确保正常样本足够
        if normal_size < normal_samples_to_take:
            normal_samples_to_take = normal_size
            # 如果需要，可以重新平衡 adv_samples_to_take，但这通常不是必需的

        # --- 从各自的缓冲区采样索引 (使用PyTorch) ---
        if normal_samples_to_take > 0:
            normal_batch_inds = th.randint(0, normal_size, (normal_samples_to_take,), device=self.device)
        else:
            normal_batch_inds = th.tensor([], dtype=th.long, device=self.device)

        if adv_samples_to_take > 0:
            adv_batch_inds = th.randint(0, adv_size, (adv_samples_to_take,), device=self.device)
        else:
            adv_batch_inds = th.tensor([], dtype=th.long, device=self.device)
        # print("DEBUG buffer.py sample: normal_batch_inds =", normal_batch_inds, "adv_batch_inds =", adv_batch_inds)
        # print("DEBUG buffer.py sample: normal_size =", normal_size, "adv_size =", adv_size)

        return self._get_samples(normal_batch_inds, adv_batch_inds, env=env)

    def _get_samples(self, normal_batch_inds: th.Tensor, adv_batch_inds: th.Tensor, env=None) -> ReplayBufferSamples:
        """
        直接从底层的PyTorch Tensor存储中获取数据。
        """
        # --- 随机采样环境索引 (使用PyTorch) ---
        if self.n_envs > 1:
            normal_env_indices = th.randint(0, self.n_envs, (len(normal_batch_inds),), device=self.device)
            adv_env_indices = th.randint(0, self.n_envs, (len(adv_batch_inds),), device=self.device)
        else:
            normal_env_indices = th.zeros(len(normal_batch_inds), dtype=th.long, device=self.device)
            adv_env_indices = th.zeros(len(adv_batch_inds), dtype=th.long, device=self.device)

        # --- 从正常缓冲区获取数据 ---
        if len(normal_batch_inds) > 0:
            # 直接从Tensor中索引
            normal_obs = self.normal_observations[normal_batch_inds, normal_env_indices, :]
            normal_next_obs = self.normal_next_observations[normal_batch_inds, normal_env_indices, :]
            normal_actions = self.normal_actions[normal_batch_inds, normal_env_indices, :]

            normal_dones_raw = self.normal_dones[normal_batch_inds, normal_env_indices]
            if self.handle_timeout_termination:
                normal_timeouts_sample = self.normal_timeouts[normal_batch_inds, normal_env_indices]
                normal_dones = (normal_dones_raw * (1 - normal_timeouts_sample)).reshape(-1, 1)
            else:
                normal_dones = normal_dones_raw.reshape(-1, 1)

            # normal_dones = self.normal_dones[normal_batch_inds, normal_env_indices].reshape(-1, 1)
            normal_rewards = self.normal_rewards[normal_batch_inds, normal_env_indices].reshape(-1, 1)
            # 正常样本没有扰动，obs_eps就是obs
            normal_obs_eps = normal_obs
            # 扰动标志为False
            normal_is_per = th.zeros_like(normal_dones, dtype=th.bool)
        else:
            # 创建正确形状和类型的空Tensor
            empty_obs_shape = (0, *self.obs_shape)
            empty_action_shape = (0, self.action_dim)
            normal_obs = th.zeros(empty_obs_shape, device=self.device, dtype=th.float32)
            normal_next_obs = th.zeros_like(normal_obs)
            normal_actions = th.zeros(empty_action_shape, device=self.device, dtype=th.float32)
            normal_dones = th.zeros((0, 1), device=self.device, dtype=th.float32)
            normal_rewards = th.zeros_like(normal_dones)
            normal_obs_eps = th.zeros_like(normal_obs)
            normal_is_per = th.zeros((0, 1), device=self.device, dtype=th.bool)

        # --- 从对抗缓冲区获取数据 ---
        if len(adv_batch_inds) > 0:
            adv_obs = self.adv_observations[adv_batch_inds, adv_env_indices, :]
            adv_next_obs = self.adv_next_observations[adv_batch_inds, adv_env_indices, :]
            adv_actions = self.adv_actions[adv_batch_inds, adv_env_indices, :]
            adv_dones_raw = self.adv_dones[adv_batch_inds, adv_env_indices]
            if self.handle_timeout_termination:
                adv_timeouts_sample = self.adv_timeouts[adv_batch_inds, adv_env_indices]
                adv_dones = (adv_dones_raw * (1 - adv_timeouts_sample)).reshape(-1, 1)
            else:
                adv_dones = adv_dones_raw.reshape(-1, 1)

            # adv_dones = self.adv_dones[adv_batch_inds, adv_env_indices].reshape(-1, 1)
            adv_rewards = self.adv_rewards[adv_batch_inds, adv_env_indices].reshape(-1, 1)
            adv_obs_eps = self.adv_observations_eps[adv_batch_inds, adv_env_indices, :]
            # 扰动标志为True
            adv_is_per = th.ones_like(adv_dones, dtype=th.bool)
        else:
            empty_obs_shape = (0, *self.obs_shape)
            empty_action_shape = (0, self.action_dim)
            adv_obs = th.zeros(empty_obs_shape, device=self.device, dtype=th.float32)
            adv_next_obs = th.zeros_like(adv_obs)
            adv_actions = th.zeros(empty_action_shape, device=self.device, dtype=th.float32)
            adv_dones = th.zeros((0, 1), device=self.device, dtype=th.float32)
            adv_rewards = th.zeros_like(adv_dones)
            adv_obs_eps = th.zeros_like(adv_obs)
            adv_is_per = th.zeros((0, 1), device=self.device, dtype=th.bool)

        # --- 合并正常和对抗数据 (使用PyTorch) ---
        obs = th.cat((normal_obs, adv_obs), dim=0)
        next_obs = th.cat((normal_next_obs, adv_next_obs), dim=0)
        actions = th.cat((normal_actions, adv_actions), dim=0)
        dones = th.cat((normal_dones, adv_dones), dim=0)
        rewards = th.cat((normal_rewards, adv_rewards), dim=0)
        obs_eps = th.cat((normal_obs_eps, adv_obs_eps), dim=0)
        obs_is_per = th.cat((normal_is_per, adv_is_per), dim=0)

        # 2025-11-15 wq 打乱
        batch_size = obs.shape[0]
        # 创建一个从 0 到 batch_size-1 的随机排列索引
        shuffled_indices = th.randperm(batch_size, device=self.device)

        # 使用这个随机索引来重新排列所有的数据
        obs = obs[shuffled_indices]
        next_obs = next_obs[shuffled_indices]
        actions = actions[shuffled_indices]
        dones = dones[shuffled_indices]
        rewards = rewards[shuffled_indices]
        obs_eps = obs_eps[shuffled_indices]
        obs_is_per = obs_is_per[shuffled_indices]


        return ReplayBufferSamples(
            observations=obs,
            actions=actions,
            next_observations=next_obs,
            dones=dones,
            rewards=rewards,
            observations_eps=obs_eps,
            obs_is_perturbed=obs_is_per,
        )




class ReplayBufferDefender(ReplayBuffer):

    def __init__(
        self,
        buffer_size,
        observation_space,
        action_space,
        device,
        n_envs: int = 1,
        optimize_memory_usage: bool = False,
        handle_timeout_termination: bool = True,
    ):
        super().__init__(buffer_size, observation_space, action_space, device, n_envs=n_envs)


        # Adjust buffer size
        self.buffer_size = max(buffer_size // n_envs, 1)

        # Check that the replay buffer can fit into the memory
        if psutil is not None:
            mem_available = psutil.virtual_memory().available

        # there is a bug if both optimize_memory_usage and handle_timeout_termination are true
        # see https://github.com/DLR-RM/stable-baselines3/issues/934
        if optimize_memory_usage and handle_timeout_termination:
            raise ValueError(
                "ReplayBuffer does not support optimize_memory_usage = True "
                "and handle_timeout_termination = True simultaneously."
            )
        self.optimize_memory_usage = optimize_memory_usage

        self.observations = np.zeros((self.buffer_size, self.n_envs, *self.obs_shape), dtype=observation_space.dtype)

        if not optimize_memory_usage:
            # When optimizing memory, `observations` contains also the next observation
            self.next_observations = np.zeros((self.buffer_size, self.n_envs, *self.obs_shape), dtype=observation_space.dtype)

        self.actions = np.zeros(
            (self.buffer_size, self.n_envs, self.action_dim), dtype=self._maybe_cast_dtype(action_space.dtype)
        )

        self.rewards = np.zeros((self.buffer_size, self.n_envs), dtype=np.float32)
        self.dones = np.zeros((self.buffer_size, self.n_envs), dtype=np.float32)
        # Handle timeouts termination properly if needed
        # see https://github.com/DLR-RM/stable-baselines3/issues/284
        self.handle_timeout_termination = handle_timeout_termination
        self.timeouts = np.zeros((self.buffer_size, self.n_envs), dtype=np.float32)

        if psutil is not None:
            total_memory_usage: float = (
                self.observations.nbytes + self.actions.nbytes + self.rewards.nbytes + self.dones.nbytes
            )

            if not optimize_memory_usage:
                total_memory_usage += self.next_observations.nbytes

            if total_memory_usage > mem_available:
                # Convert to GB
                total_memory_usage /= 1e9
                mem_available /= 1e9
                warnings.warn(
                    "This system does not have apparently enough memory to store the complete "
                    f"replay buffer {total_memory_usage:.2f}GB > {mem_available:.2f}GB"
                )
        self.observations_eps = np.zeros_like(self.observations)
        self.obs_is_perturbed = np.zeros((self.buffer_size, self.n_envs), dtype=bool)

    def add(
        self,
        obs: np.ndarray,
        next_obs: np.ndarray,
        action: np.ndarray,
        reward: np.ndarray,
        done: np.ndarray,
        infos,
        obs_eps: Optional[np.ndarray] = None,
        obs_is_per: Union[np.ndarray, bool] = False,
    ) -> None:
        # Reshape needed when using multiple envs with discrete observations
        # as numpy cannot broadcast (n_discrete,) to (n_discrete, 1)
        if isinstance(self.observation_space, spaces.Discrete):
            obs = obs.reshape((self.n_envs, *self.obs_shape))
            next_obs = next_obs.reshape((self.n_envs, *self.obs_shape))
            if obs_eps is not None:
                obs_eps = obs_eps.reshape((self.n_envs, *self.obs_shape))

        # Reshape to handle multi-dim and discrete action spaces, see GH #970 #1392
        action = action.reshape((self.n_envs, self.action_dim))

        if obs_eps is None:
            obs_eps = np.array(obs, copy=True)


        # Copy to avoid modification by reference
        self.observations[self.pos] = np.array(obs)
        self.observations_eps[self.pos] = np.array(obs_eps)

        if self.optimize_memory_usage:
            self.observations[(self.pos + 1) % self.buffer_size] = np.array(next_obs)
        else:
            self.next_observations[self.pos] = np.array(next_obs)

        self.actions[self.pos] = np.array(action)
        self.rewards[self.pos] = np.array(reward)
        self.dones[self.pos] = np.array(done)
        self.obs_is_perturbed[self.pos] = np.array(obs_is_per)

        if self.handle_timeout_termination:
            self.timeouts[self.pos] = np.array([info.get("TimeLimit.truncated", False) for info in infos])

        self.pos += 1
        if self.pos == self.buffer_size:
            self.full = True
            self.pos = 0

    def sample(self, batch_size: int, env = None) -> ReplayBufferSamples:
        """
        Sample elements from the replay buffer.
        Custom sampling when using memory efficient variant,
        as we should not sample the element with index `self.pos`
        See https://github.com/DLR-RM/stable-baselines3/pull/28#issuecomment-637559274

        :param batch_size: Number of element to sample
        :param env: associated gym VecEnv
            to normalize the observations/rewards when sampling
        :return:
        """
        if not self.optimize_memory_usage:
            return super().sample(batch_size=batch_size, env=env)
        # Do not sample the element with index `self.pos` as the transitions is invalid
        # (we use only one array to store `obs` and `next_obs`)
        if self.full:
            batch_inds = (np.random.randint(1, self.buffer_size, size=batch_size) + self.pos) % self.buffer_size
        else:
            batch_inds = np.random.randint(0, self.pos, size=batch_size)
        return self._get_samples(batch_inds, env=env)

    def _get_samples(self, batch_inds: np.ndarray, env = None) -> ReplayBufferSamples:
        # Sample randomly the env idx
        env_indices = np.random.randint(0, high=self.n_envs, size=(len(batch_inds),))

        if self.optimize_memory_usage:
            next_obs = self._normalize_obs(self.observations[(batch_inds + 1) % self.buffer_size, env_indices, :], env)
        else:
            next_obs = self._normalize_obs(self.next_observations[batch_inds, env_indices, :], env)

        data = (
            self._normalize_obs(self.observations[batch_inds, env_indices, :], env),
            self.actions[batch_inds, env_indices, :],
            next_obs,
            # Only use dones that are not due to timeouts
            # deactivated by default (timeouts is initialized as an array of False)
            (self.dones[batch_inds, env_indices] * (1 - self.timeouts[batch_inds, env_indices])).reshape(-1, 1),
            self._normalize_reward(self.rewards[batch_inds, env_indices].reshape(-1, 1), env),
            self._normalize_obs(self.observations_eps[batch_inds, env_indices, :], env),
            self.obs_is_perturbed[batch_inds, env_indices].reshape(-1, 1),
        )
        return ReplayBufferSamples(*tuple(map(self.to_torch, data)))

    @staticmethod
    def _maybe_cast_dtype(dtype: np.typing.DTypeLike) -> np.typing.DTypeLike:
        """
        Cast `np.float64` action datatype to `np.float32`,
        keep the others dtype unchanged.
        See GH#1572 for more information.

        :param dtype: The original action space dtype
        :return: ``np.float32`` if the dtype was float64,
            the original dtype otherwise.
        """
        if dtype == np.float64:
            return np.float32
        return dtype








class PaddingRolloutBuffer(RolloutBuffer):
    """
    Rollout buffer with only trajectory padding
    """

    def __init__(self, buffer_size, observation_space, action_space, device, gamma=0.99, gae_lambda=1, n_envs=1):
        super().__init__(buffer_size, observation_space, action_space, device, gamma, gae_lambda, n_envs)
        self.max_steps = 30  # Assuming buffer_size as max_steps
        self.current_episode_length = 1
        self.flag = False

    def add(self, obs, action, reward, episode_start, value, log_prob):
        if len(log_prob.shape) == 0:
            # Reshape 0-d tensor to avoid error
            log_prob = log_prob.reshape(-1, 1)

        # Reshape needed when using multiple envs with discrete observations
        # as numpy cannot broadcast (n_discrete,) to (n_discrete, 1)
        if isinstance(self.observation_space, spaces.Discrete):
            obs = obs.reshape((self.n_envs, *self.obs_shape))

        # Reshape to handle multi-dim and discrete action spaces, see GH #970 #1392
        action = action.reshape((self.n_envs, self.action_dim))

        self.observations[self.pos] = np.array(obs)
        self.actions[self.pos] = np.array(action)
        self.rewards[self.pos] = np.array(reward)
        self.episode_starts[self.pos] = np.array(episode_start)
        self.values[self.pos] = value.clone().cpu().numpy().flatten()
        self.log_probs[self.pos] = log_prob.clone().cpu().numpy()
        self.pos += 1
        if self.pos == self.buffer_size:
            self.full = True
        # print('episode_starts is ', self.episode_starts[self.pos-1])
        if self.episode_starts[self.pos - 1][0]:
            self.current_episode_length = 1
        else:
            self.current_episode_length += 1
        # Check if the episode has ended
        if self.flag:
            print('*****************************padding***************************************')
            # If episode length < max_steps, repeat the trajectory
            if self.current_episode_length < self.max_steps and self.pos >= self.current_episode_length:
                N = int(np.ceil(self.max_steps / self.current_episode_length))
                remaining_space = self.buffer_size - self.pos
                print('remaining_space', remaining_space)
                N = min(N, remaining_space // self.current_episode_length)
                print('pos is ', self.pos)
                print('padding times is ', N)
                print('current episode length is ', self.current_episode_length)
                # Create slices for the last episode's data
                obs_slice = self.observations[self.pos - self.current_episode_length: self.pos]
                action_slice = self.actions[self.pos - self.current_episode_length: self.pos]
                rewards_slice = self.rewards[self.pos - self.current_episode_length: self.pos]
                episode_start_slice = self.episode_starts[self.pos - self.current_episode_length: self.pos]
                values_slice = self.values[self.pos - self.current_episode_length: self.pos]
                log_probs_slice = self.log_probs[self.pos - self.current_episode_length: self.pos]

                # Repeat the last episode's data N times along the second dimension
                for i in range(N):
                    self.observations[self.pos:self.pos + self.current_episode_length] = np.tile(obs_slice, 1)
                    self.actions[self.pos:self.pos + self.current_episode_length] = np.tile(action_slice, 1)
                    self.rewards[self.pos:self.pos + self.current_episode_length] = np.tile(rewards_slice, 1)
                    self.episode_starts[self.pos:self.pos + self.current_episode_length] = np.tile(episode_start_slice,
                                                                                                   1)  # 1D, so no change here
                    self.values[self.pos:self.pos + self.current_episode_length] = np.tile(values_slice, 1)
                    self.log_probs[self.pos:self.pos + self.current_episode_length] = np.tile(log_probs_slice, 1)
                    self.pos += self.current_episode_length
                if self.pos == self.buffer_size:
                    self.full = True
        self.flag = False

    def log_collisions(self):
        self.flag = True

    def return_pos(self):
        return self.pos


class DecouplePaddingRolloutBuffer(RolloutBuffer):
    """
    Rollout buffer with trajectory padding and decouple policy update
    """

    def __init__(self, buffer_size, observation_space, action_space, device, gamma=0.99, gae_lambda=1, n_envs=1):
        super().__init__(buffer_size, observation_space, action_space, device, gamma, gae_lambda, n_envs)
        self.max_steps = 30  # Assuming buffer_size as max_steps
        self.current_episode_length = 1
        self.flag = False

    def add(self, obs, action, reward, episode_start, value, log_prob):
        if len(log_prob.shape) == 0:
            # Reshape 0-d tensor to avoid error
            log_prob = log_prob.reshape(-1, self.action_dim)

        # Reshape needed when using multiple envs with discrete observations
        # as numpy cannot broadcast (n_discrete,) to (n_discrete, 1)
        if isinstance(self.observation_space, spaces.Discrete):
            obs = obs.reshape((self.n_envs, *self.obs_shape))

        # Reshape to handle multi-dim and discrete action spaces, see GH #970 #1392
        action = action.reshape((self.n_envs, self.action_dim))

        self.observations[self.pos] = np.array(obs)
        self.actions[self.pos] = np.array(action)
        self.rewards[self.pos] = np.array(reward)
        self.episode_starts[self.pos] = np.array(episode_start)
        self.values[self.pos] = value.clone().cpu().numpy().flatten()
        self.log_probs[self.pos] = np.array(log_prob.clone().cpu())
        self.pos += 1
        if self.pos == self.buffer_size:
            self.full = True
        # print('episode_starts is ', self.episode_starts[self.pos-1])
        if self.episode_starts[self.pos - 1][0]:
            self.current_episode_length = 1
        else:
            self.current_episode_length += 1
        # Check if the episode has ended
        if self.flag:
            # print('*****************************padding***************************************')
            # If episode length < max_steps, repeat the trajectory
            if self.current_episode_length < self.max_steps and self.pos >= self.current_episode_length:
                N = int(np.ceil(self.max_steps / self.current_episode_length))
                remaining_space = self.buffer_size - self.pos
                N = min(N, remaining_space // self.current_episode_length)
                # print('pos is ', self.pos)
                # print('padding times is ', N)
                # print('current episode length is ', self.current_episode_length)
                # Create slices for the last episode's data
                obs_slice = self.observations[self.pos - self.current_episode_length: self.pos]
                action_slice = self.actions[self.pos - self.current_episode_length: self.pos]
                rewards_slice = self.rewards[self.pos - self.current_episode_length: self.pos]
                episode_start_slice = self.episode_starts[self.pos - self.current_episode_length: self.pos]
                values_slice = self.values[self.pos - self.current_episode_length: self.pos]
                log_probs_slice = self.log_probs[self.pos - self.current_episode_length: self.pos]
                # print('obs is ', self.observations)
                # print('obs slice ', obs_slice)
                # print('obs slice is ', obs_slice)

                # Repeat the last episode's data N times along the second dimension
                for i in range(N):
                    self.observations[self.pos:self.pos + self.current_episode_length] = np.tile(obs_slice, 1)
                    self.actions[self.pos:self.pos + self.current_episode_length] = np.tile(action_slice, 1)
                    self.rewards[self.pos:self.pos + self.current_episode_length] = np.tile(rewards_slice, 1)
                    self.episode_starts[self.pos:self.pos + self.current_episode_length] = np.tile(episode_start_slice,
                                                                                                   1)  # 1D, so no change here
                    self.values[self.pos:self.pos + self.current_episode_length] = np.tile(values_slice, 1)
                    self.log_probs[self.pos:self.pos + self.current_episode_length] = np.tile(log_probs_slice, 1)
                    self.pos += self.current_episode_length
                if self.pos == self.buffer_size:
                    self.full = True

                # print('obs 1 N is ', self.observations[self.pos - self.current_episode_length * 1: self.pos - self.current_episode_length * 0])
                # print('obs 2 N is ', self.observations[self.pos - self.current_episode_length * 2: self.pos - self.current_episode_length * 1])
                # print('obs 3 N is ', self.observations[self.pos - self.current_episode_length * 3: self.pos - self.current_episode_length * 2])
                # print('actions 1 N ', self.actions[self.pos - self.current_episode_length * 1: self.pos - self.current_episode_length * 0])
                # print('actions 2 N ', self.actions[self.pos - self.current_episode_length * 2: self.pos - self.current_episode_length * 1])
                # print('actions 3 N ', self.actions[self.pos - self.current_episode_length * 3: self.pos - self.current_episode_length * 2])
                # print('rewards 1 N ', self.rewards[self.pos - self.current_episode_length * 1: self.pos - self.current_episode_length * 0])
                # print('rewards 2 N ', self.rewards[self.pos - self.current_episode_length * 2: self.pos - self.current_episode_length * 1])
                # print('rewards 3 N ', self.rewards[self.pos - self.current_episode_length * 3: self.pos - self.current_episode_length * 2])
                # print('episode_starts 1 N ', self.episode_starts[self.pos - self.current_episode_length * 1: self.pos - self.current_episode_length * 0])
                # print('episode_starts 2 N ', self.episode_starts[self.pos - self.current_episode_length * 2: self.pos - self.current_episode_length * 1])
                # print('episode_starts 3 N ', self.episode_starts[self.pos - self.current_episode_length * 3: self.pos - self.current_episode_length * 2])
                # print('values 1 N ', self.values[self.pos - self.current_episode_length * 1: self.pos - self.current_episode_length * 0])
                # print('values 2 N ', self.values[self.pos - self.current_episode_length * 2: self.pos - self.current_episode_length * 1])
                # print('values 3 N ', self.values[self.pos - self.current_episode_length * 3: self.pos - self.current_episode_length * 2])
                # print('log probs 1 N ', self.log_probs[self.pos - self.current_episode_length * 1: self.pos - self.current_episode_length * 0])
                # print('log probs 2 N ', self.log_probs[self.pos - self.current_episode_length * 2: self.pos - self.current_episode_length * 1])
                # print('log probs 3 N ', self.log_probs[self.pos - self.current_episode_length * 3: self.pos - self.current_episode_length * 2])
        self.flag = False

    def log_collisions(self):
        self.flag = True

    def return_pos(self):
        return self.pos

    def reset(self):
        super().reset()
        self.current_episode_length = 0
        self.log_probs = np.zeros((self.buffer_size, self.n_envs, self.action_dim), dtype=np.float32)

    def _get_samples(self, batch_inds, env=None):
        data = (
            self.observations[batch_inds],
            self.actions[batch_inds],
            self.values[batch_inds].flatten(),
            self.log_probs[batch_inds],
            self.advantages[batch_inds].flatten(),
            self.returns[batch_inds].flatten(),
        )
        return RolloutBufferSamples(*tuple(map(self.to_torch, data)))
