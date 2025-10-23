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
