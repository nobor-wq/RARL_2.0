import warnings
import numpy as np
import torch as th
from stable_baselines3.common.vec_env import DummyVecEnv, VecEnv, VecMonitor, is_vecenv_wrapped
from stable_baselines3.common.utils import obs_as_tensor
from fgsm import FGSM_v2
from policy import FniNet, IGCARLNet




def evaluate_policy_adv(
    model,
    trained_agent,
    env,
    n_eval_episodes = 10,
    deterministic = True,
    render = False,
    callback = None,
    reward_threshold = None,
    return_episode_rewards = False,
    warn = True,
    unlimited_attack = False,
    attack_method = 'fgsm',
    attack_eps = 0.01
):


    episode_rewards = 0
    episode_lengths = 0
    success_count = 0
    attack_count = 0
    attack_success_count = 0

    for i in range(n_eval_episodes):
        observations = env.reset()

        states = None
        xa = ya = None

        for j in range(30):
            with th.no_grad():
                # Convert to pytorch tensor or to TensorDict
                obs_tensor = obs_as_tensor(observations, model.device)
                if obs_tensor.dim() == 1:
                    obs_tensor = obs_tensor.unsqueeze(0)  # (1, obs_dim)
                if isinstance(trained_agent, (FniNet, IGCARLNet)):
                    actions, std, _action = trained_agent(obs_tensor[:, :-2])
                    actions = actions.detach().cpu().numpy()
                else:
                    actions, _states = trained_agent.predict(obs_tensor[:, :-2].cpu(), deterministic=True)

            actions_tensor = th.tensor(actions, device=obs_tensor.device)  # 确保在同一设备上
            obs_tensor[:, -1] = actions_tensor.squeeze(-1)

            adv_actions, states = model.predict(
                obs_tensor.cpu(),  # type: ignore[arg-type]
                state=states,
                deterministic=deterministic,
            )

            # Get adv action mask
            if unlimited_attack:
                adv_action_mask = np.ones_like(adv_actions)
            else:
                adv_action_mask = (adv_actions[:, 0] > 0) & (obs_tensor[:, -2].cpu().numpy() > 0)

                if adv_action_mask:
                    attack_count += 1

            final_actions = attack_process(obs_tensor, adv_action_mask, adv_actions, actions, attack_method, trained_agent, obs_tensor.device, attack_eps)

            new_observations, rewards, dones, infos = env.step(final_actions)
            print("DEBUG i: ", i, " j: ", j, " done: ", dones, "adv_mask: ", adv_action_mask)
            # infos 可能是 dict 或 list
            if isinstance(infos, dict):
                info0 = infos
            elif isinstance(infos, (list, tuple)) and len(infos) > 0:
                info0 = infos[0]
            else:
                raise ValueError(f"Invalid infos format: {type(infos)}")

            # 如果没有 attReStep 键则报错
            if 'reward' not in info0:
                raise KeyError(f"'reward' key not found in info: {info0}")

            r_def = float(info0['reward'])
            c_def = float(info0['cost'])


            xa = info0.get("x_position", None)
            ya = info0.get("y_position", None)
            rc_def = r_def - c_def
            episode_rewards += rc_def
            episode_lengths += 1
            if dones:
                if adv_action_mask:
                    attack_success_count += 1
                    print("DEBUG attack success: ", attack_success_count)
                break
            observations = new_observations
        if xa is not None and ya is not None:
            if xa < -50.0 and ya > 4.0 and not dones:  # 还没结束时达成目标
                success_count += 1
                print("DEBUG success: ", success_count)

            # if render:
            #     env.render()

    mean_reward = episode_rewards / n_eval_episodes
    success_rate = success_count / n_eval_episodes
    mean_length = episode_lengths / n_eval_episodes
    mean_attack = attack_count / n_eval_episodes
    mean_attack_success = attack_success_count / n_eval_episodes
    # std_reward = np.std(episode_rewards)
    # if reward_threshold is not None:
    #     assert mean_reward > reward_threshold, "Mean reward below threshold: " f"{mean_reward:.2f} < {reward_threshold:.2f}"
    # swanlab.log({"mean_reward": mean_reward, "std_reward": std_reward})
    # if return_episode_rewards:
    return mean_reward, mean_length, success_rate, mean_attack, mean_attack_success

        # return mean_reward, std_reward


def evaluate_policy_def(
        model,
        trained_agent,
        trained_adv,
        env,
        n_eval_episodes=10,
        deterministic=True,
        render=False,
        callback=None,
        reward_threshold=None,
        return_episode_rewards=False,
        warn=True,
        unlimited_attack=False,
        attack_method='fgsm',
        attack_eps=0.01
):
    """
    Runs policy for ``n_eval_episodes`` episodes and returns average reward.
    If a vector env is passed in, this divides the episodes to evaluate onto the
    different elements of the vector env. This static division of work is done to
    remove bias. See https://github.com/DLR-RM/stable-baselines3/issues/402 for more
    details and discussion.

    .. note::
        If environment has not been wrapped with ``Monitor`` wrapper, reward and
        episode lengths are counted as it appears with ``env.step`` calls. If
        the environment contains wrappers that modify rewards or episode lengths
        (e.g. reward scaling, early episode reset), these will affect the evaluation
        results as well. You can avoid this by wrapping environment with ``Monitor``
        wrapper before anything else.

    :param model: The RL agent you want to evaluate. This can be any object
        that implements a `predict` method, such as an RL algorithm (``BaseAlgorithm``)
        or policy (``BasePolicy``).
    :param env: The gym environment or ``VecEnv`` environment.
    :param n_eval_episodes: Number of episode to evaluate the agent
    :param deterministic: Whether to use deterministic or stochastic actions
    :param render: Whether to render the environment or not
    :param callback: callback function to do additional checks,
        called after each step. Gets locals() and globals() passed as parameters.
    :param reward_threshold: Minimum expected reward per episode,
        this will raise an error if the performance is not met
    :param return_episode_rewards: If True, a list of rewards and episode lengths
        per episode will be returned instead of the mean.
    :param warn: If True (default), warns user about lack of a Monitor wrapper in the
        evaluation environment.
    :return: Mean reward per episode, std of reward per episode.
        Returns ([float], [int]) when ``return_episode_rewards`` is True, first
        list containing per-episode rewards and second containing per-episode lengths
        (in number of steps).
    """

    episode_rewards = 0
    episode_lengths = 0
    success_count = 0
    attack_count = 0
    attack_success_count = 0

    for i in range(n_eval_episodes):

        observations = env.reset()

        states = None

        for j in range(30):
            with th.no_grad():
                # Convert to pytorch tensor or to TensorDict
                obs_tensor = obs_as_tensor(observations, model.device)
                if obs_tensor.dim() == 1:
                    obs_tensor = obs_tensor.unsqueeze(0)  # (1, obs_dim)

                if isinstance(trained_agent, FniNet):
                    actions, std, _action = trained_agent(obs_tensor[:, :-2])
                    actions = actions.detach().cpu().numpy()
                else:
                    actions, _states = trained_agent.predict(obs_tensor[:, :-2].cpu(), deterministic=True)

            actions_tensor = th.tensor(actions, device=obs_tensor.device)  # 确保在同一设备上
            obs_tensor[:, -1] = actions_tensor.squeeze(-1)

            adv_action, _ = trained_adv.predict(obs_tensor.cpu(), deterministic=True)

            adv_action_mask = (adv_action[:, 0] > 0) & (obs_tensor[:, -2].cpu().numpy() > 0)

            obs_tensor = obs_tensor[:, :-2]

            if adv_action_mask:
                adv_action_eps = adv_action[:, 1]
                attack_count += 1
                state_eps = FGSM_v2(adv_action_eps, victim_agent=trained_agent,
                                    last_state=obs_tensor, epsilon=attack_eps, device=obs_tensor.device)
                obs_tensor = state_eps

            if isinstance(obs_tensor, th.Tensor):
                obs_tensor = obs_tensor.detach().cpu().numpy()

            final_actions, states = model.predict(
                obs_tensor,  # type: ignore[arg-type]
                state=states,
                deterministic=deterministic,
            )

            output_action = np.column_stack((final_actions, adv_action_mask.astype(np.float32)))

            new_observations, rewards, dones, infos = env.step(output_action)
            print("DEBUG dones: ", dones)

            episode_lengths += 1

            if isinstance(infos, dict):
                info1 = infos
            elif isinstance(infos, (list, tuple)) and len(infos) > 0:
                info1 = infos[0]
            else:
                raise ValueError(f"Invalid infos format: {type(infos)}")

            xa = info1.get("x_position", None)
            ya = info1.get("y_position", None)

            re = info1.get("reward", None)
            co = info1.get("cost", None)
            ep_re = re - co
            episode_rewards += ep_re

            if dones:
                if adv_action_mask:
                    attack_success_count += 1
                    print("DEBUG attack success: ", attack_success_count)
                break

            observations = new_observations


        if xa is not None and ya is not None:
            if xa < -50.0 and ya > 4.0 and not dones:  # 还没结束时达成目标
                success_count += 1
                print("DEBUG success: ", success_count)

            # if render:
            #     env.render()

    mean_reward = episode_rewards / n_eval_episodes
    mean_length = episode_lengths / n_eval_episodes
    # std_reward = np.std(episode_rewards)
    success_rate = success_count / n_eval_episodes
    mean_attack = attack_count / n_eval_episodes
    mean_attack_success = attack_success_count / n_eval_episodes
    if reward_threshold is not None:
        assert mean_reward > reward_threshold, "Mean reward below threshold: " f"{mean_reward:.2f} < {reward_threshold:.2f}"
    # swanlab.log({"mean_reward": mean_reward, "std_reward": std_reward})
    # if return_episode_rewards:
    return mean_reward, mean_length, success_rate, mean_attack, mean_attack_success


#
# def evaluate_policy_def(
#     model,
#     trained_agent,
#     trained_adv,
#     env,
#     n_eval_episodes = 10,
#     deterministic = True,
#     render = False,
#     callback = None,
#     reward_threshold = None,
#     return_episode_rewards = False,
#     warn = True,
#     unlimited_attack = False,
#     attack_method = 'fgsm',
#     attack_eps = 0.01
# ):
#     """
#     Runs policy for ``n_eval_episodes`` episodes and returns average reward.
#     If a vector env is passed in, this divides the episodes to evaluate onto the
#     different elements of the vector env. This static division of work is done to
#     remove bias. See https://github.com/DLR-RM/stable-baselines3/issues/402 for more
#     details and discussion.
#
#     .. note::
#         If environment has not been wrapped with ``Monitor`` wrapper, reward and
#         episode lengths are counted as it appears with ``env.step`` calls. If
#         the environment contains wrappers that modify rewards or episode lengths
#         (e.g. reward scaling, early episode reset), these will affect the evaluation
#         results as well. You can avoid this by wrapping environment with ``Monitor``
#         wrapper before anything else.
#
#     :param model: The RL agent you want to evaluate. This can be any object
#         that implements a `predict` method, such as an RL algorithm (``BaseAlgorithm``)
#         or policy (``BasePolicy``).
#     :param env: The gym environment or ``VecEnv`` environment.
#     :param n_eval_episodes: Number of episode to evaluate the agent
#     :param deterministic: Whether to use deterministic or stochastic actions
#     :param render: Whether to render the environment or not
#     :param callback: callback function to do additional checks,
#         called after each step. Gets locals() and globals() passed as parameters.
#     :param reward_threshold: Minimum expected reward per episode,
#         this will raise an error if the performance is not met
#     :param return_episode_rewards: If True, a list of rewards and episode lengths
#         per episode will be returned instead of the mean.
#     :param warn: If True (default), warns user about lack of a Monitor wrapper in the
#         evaluation environment.
#     :return: Mean reward per episode, std of reward per episode.
#         Returns ([float], [int]) when ``return_episode_rewards`` is True, first
#         list containing per-episode rewards and second containing per-episode lengths
#         (in number of steps).
#     """
#
#     episode_rewards = 0
#     episode_lengths = 0
#     success_count = 0
#     attack_count = 0
#
#
#     for i in range(n_eval_episodes):
#
#         observations = env.reset()
#
#         states = None
#
#         print("DEBUG: obs: ", observations)
#
#         for j in range(30):
#
#             print("DEBUG i: ", i, " j: ", j)
#
#             with th.no_grad():
#                 # Convert to pytorch tensor or to TensorDict
#                 obs_tensor = obs_as_tensor(observations, model.device)
#
#                 if isinstance(trained_agent, FniNet):
#                     actions, std, _action = trained_agent(obs_tensor)
#                     actions = actions.detach().cpu().numpy()
#                 else:
#                     actions, _states = trained_agent.predict(obs_tensor.cpu(), deterministic=True)
#
#             actions_tensor = th.tensor(actions, device=obs_tensor.device)  # 确保在同一设备上
#
#             # infos 可能是 dict 或 list
#             if isinstance(infos_adv, dict):
#                 info0 = infos_adv
#             elif isinstance(infos_adv, (list, tuple)) and len(infos_adv) > 0:
#                 info0 = infos_adv[0]
#             else:
#                 raise ValueError(f"Invalid infos format: {type(infos_adv)}")
#
#             # 如果没有 attReStep 键则报错
#             if 'attReStep' not in info0:
#                 raise KeyError(f"'attReStep' key not found in info: {info0}")
#
#             attReStep = float(info0['attReStep'])
#
#
#             attReStep_tensor = th.tensor(attReStep, device=obs_tensor.device).reshape(1, 1)
#
#             obs_adv = th.cat([obs_tensor, attReStep_tensor, actions_tensor], dim=-1)  # -> (batch, 28)
#
#             adv_action, _ = trained_adv.predict(obs_adv.cpu(), deterministic=True)
#
#
#             adv_action_mask = (adv_action[:, 0] > 0) & (obs_adv[:, -2].cpu().numpy() > 0)
#
#             if adv_action_mask:
#                 adv_action_eps = adv_action[:, 1]
#                 attack_count += 1
#                 print("DEBUG: attack count: ", attack_count)
#                 state_eps = FGSM_v2(adv_action_eps, victim_agent=trained_agent,
#                                     last_state=obs_tensor, epsilon=attack_eps, device=obs_tensor.device)
#                 obs_tensor = state_eps
#             if isinstance(obs_tensor, th.Tensor):
#                 obs_tensor = obs_tensor.detach().cpu().numpy()
#
#             final_actions, states = model.predict(
#                 obs_tensor,  # type: ignore[arg-type]
#                 state=states,
#                 deterministic=deterministic,
#             )
#             combined_action = np.array([final_actions, adv_action_mask], dtype=np.float32)
#             print("DEBUG com_action: ", combined_action)
#             new_observations, rewards, dones, infos = env.step(combined_action)
#             episode_rewards += rewards
#             episode_lengths += 1
#
#             if isinstance(infos, dict):
#                 info1 = infos
#             elif isinstance(infos, (list, tuple)) and len(infos) > 0:
#                 info1 = infos[0]
#             else:
#                 raise ValueError(f"Invalid infos format: {type(infos)}")
#
#             xa = info1.get("x_position", None)
#             ya = info1.get("y_position", None)
#
#             cost = float(info1['cost'])
#             done_c = bool(cost)
#             if done_c:
#                 break
#
#             observations = new_observations
#             infos_adv = infos
#
#         if xa is not None and ya is not None:
#             if xa < -50.0 and ya > 4.0 and not done_c:  # 还没结束时达成目标
#                 success_count += 1
#
#             # if render:
#             #     env.render()
#
#     mean_reward = episode_rewards / n_eval_episodes
#     mean_length = episode_lengths / n_eval_episodes
#     # std_reward = np.std(episode_rewards)
#     success_rate = success_count / n_eval_episodes
#     mean_attack =  attack_count / n_eval_episodes
#     if reward_threshold is not None:
#         assert mean_reward > reward_threshold, "Mean reward below threshold: " f"{mean_reward:.2f} < {reward_threshold:.2f}"
#     # swanlab.log({"mean_reward": mean_reward, "std_reward": std_reward})
#     # if return_episode_rewards:
#     return mean_reward, mean_length, success_rate, mean_attack

    # return mean_reward, std_reward


#
# def evaluate_policy_adv(
#     model,
#     trained_agent,
#     env,
#     n_eval_episodes = 10,
#     deterministic = True,
#     render = False,
#     callback = None,
#     reward_threshold = None,
#     return_episode_rewards = False,
#     warn = True,
#     unlimited_attack = False,
#     attack_method = 'fgsm',
#     attack_eps = 0.01
# ):
#     """
#     Runs policy for ``n_eval_episodes`` episodes and returns average reward.
#     If a vector env is passed in, this divides the episodes to evaluate onto the
#     different elements of the vector env. This static division of work is done to
#     remove bias. See https://github.com/DLR-RM/stable-baselines3/issues/402 for more
#     details and discussion.
#
#     .. note::
#         If environment has not been wrapped with ``Monitor`` wrapper, reward and
#         episode lengths are counted as it appears with ``env.step`` calls. If
#         the environment contains wrappers that modify rewards or episode lengths
#         (e.g. reward scaling, early episode reset), these will affect the evaluation
#         results as well. You can avoid this by wrapping environment with ``Monitor``
#         wrapper before anything else.
#
#     :param model: The RL agent you want to evaluate. This can be any object
#         that implements a `predict` method, such as an RL algorithm (``BaseAlgorithm``)
#         or policy (``BasePolicy``).
#     :param env: The gym environment or ``VecEnv`` environment.
#     :param n_eval_episodes: Number of episode to evaluate the agent
#     :param deterministic: Whether to use deterministic or stochastic actions
#     :param render: Whether to render the environment or not
#     :param callback: callback function to do additional checks,
#         called after each step. Gets locals() and globals() passed as parameters.
#     :param reward_threshold: Minimum expected reward per episode,
#         this will raise an error if the performance is not met
#     :param return_episode_rewards: If True, a list of rewards and episode lengths
#         per episode will be returned instead of the mean.
#     :param warn: If True (default), warns user about lack of a Monitor wrapper in the
#         evaluation environment.
#     :return: Mean reward per episode, std of reward per episode.
#         Returns ([float], [int]) when ``return_episode_rewards`` is True, first
#         list containing per-episode rewards and second containing per-episode lengths
#         (in number of steps).
#     """
#     is_monitor_wrapped = False
#     # Avoid circular import
#     from stable_baselines3.common.monitor import Monitor
#
#     if not isinstance(env, VecEnv):
#         env = DummyVecEnv([lambda: env])  # type: ignore[list-item, return-value]
#
#     is_monitor_wrapped = is_vecenv_wrapped(env, VecMonitor) or env.env_is_wrapped(Monitor)[0]
#
#     if not is_monitor_wrapped and warn:
#         warnings.warn(
#             "Evaluation environment is not wrapped with a ``Monitor`` wrapper. "
#             "This may result in reporting modified episode lengths and rewards, if other wrappers happen to modify these. "
#             "Consider wrapping environment first with ``Monitor`` wrapper.",
#             UserWarning,
#         )
#
#     n_envs = env.num_envs
#     episode_rewards = []
#     episode_lengths = []
#     success_count = 0
#
#     episode_counts = np.zeros(n_envs, dtype="int")
#     # Divides episodes among different sub environments in the vector as evenly as possible
#     episode_count_targets = np.array([(n_eval_episodes + i) // n_envs for i in range(n_envs)], dtype="int")
#
#     current_rewards = np.zeros(n_envs)
#     current_lengths = np.zeros(n_envs, dtype="int")
#     observations = env.reset()
#     states = None
#     episode_starts = np.ones((env.num_envs,), dtype=bool)
#     while (episode_counts < episode_count_targets).any():
#         with th.no_grad():
#             # Convert to pytorch tensor or to TensorDict
#             obs_tensor = obs_as_tensor(observations, model.device)
#             if isinstance(trained_agent, FniNet):
#                 actions, std, _action = trained_agent(obs_tensor[:, :-2])
#                 actions = actions.detach().cpu().numpy()
#             else:
#                 actions, _states = trained_agent.predict(obs_tensor[:, :-2].cpu(), deterministic=True)
#
#         actions_tensor = th.tensor(actions, device=obs_tensor.device)  # 确保在同一设备上
#         obs_tensor[:, -1] = actions_tensor.squeeze(-1)
#
#         adv_actions, states = model.predict(
#             obs_tensor.cpu(),  # type: ignore[arg-type]
#             state=states,
#             episode_start=episode_starts,
#             deterministic=deterministic,
#         )
#
#         # Get adv action mask
#         if unlimited_attack:
#             adv_action_mask = np.ones_like(adv_actions)
#         else:
#             adv_action_mask = (adv_actions[:, 0] > 0) & (obs_tensor[:, -2].cpu().numpy() > 0)
#
#         final_actions = attack_process(obs_tensor, adv_action_mask, adv_actions, actions, attack_method, trained_agent, obs_tensor.device, attack_eps)
#
#         new_observations, rewards, dones, infos = env.step(final_actions)
#         # infos 可能是 dict 或 list
#         if isinstance(infos, dict):
#             info0 = infos
#         elif isinstance(infos, (list, tuple)) and len(infos) > 0:
#             info0 = infos[0]
#         else:
#             raise ValueError(f"Invalid infos format: {type(infos)}")
#
#         # 如果没有 attReStep 键则报错
#         if 'reward' not in info0:
#             raise KeyError(f"'reward' key not found in info: {info0}")
#
#
#         r_def = float(info0['reward'])
#         c_def = float(info0['cost'])
#
#         xa = info0.get("x_position", None)
#         ya = info0.get("y_position", None)
#
#
#
#         rc_def = r_def - c_def
#         print("DEBUG: re: ", rc_def)
#         current_rewards += rc_def
#         current_lengths += 1
#         for i in range(n_envs):
#             if episode_counts[i] < episode_count_targets[i]:
#                 # unpack values so that the callback can access the local variables
#                 reward = rewards[i]
#                 done = dones[i]
#                 info = infos[i]
#                 episode_starts[i] = done
#
#                 if callback is not None:
#                     callback(locals(), globals())
#
#                 if dones[i]:
#                     # if is_monitor_wrapped:
#                     #     # Atari wrapper can send a "done" signal when
#                     #     # the agent loses a life, but it does not correspond
#                     #     # to the true end of episode
#                     #     if "episode" in info.keys():
#                     #         # Do not trust "done" with episode endings.
#                     #         # Monitor wrapper includes "episode" key in info if environment
#                     #         # has been wrapped with it. Use those rewards instead.
#                     #         episode_rewards.append(info["episode"]["r"])
#                     #         episode_lengths.append(info["episode"]["l"])
#                     #         # Only increment at the real end of an episode
#                     #         episode_counts[i] += 1
#                     #         print("DEBUG: get episode")
#                     # else:
#                     episode_rewards.append(current_rewards[i])
#                     episode_lengths.append(current_lengths[i])
#                     episode_counts[i] += 1
#
#                     if xa is not None and ya is not None:
#                         if xa < -50.0 and ya > 4.0 and not any(dones[i]):  # 还没结束时达成目标
#                             success_count += 1
#
#                     current_rewards[i] = 0
#                     current_lengths[i] = 0
#
#         observations = new_observations
#
#         if render:
#             env.render()
#
#     mean_reward = np.mean(episode_rewards)
#     success_rate = success_count / n_eval_episodes
#     std_reward = np.std(episode_rewards)
#     if reward_threshold is not None:
#         assert mean_reward > reward_threshold, "Mean reward below threshold: " f"{mean_reward:.2f} < {reward_threshold:.2f}"
#     # swanlab.log({"mean_reward": mean_reward, "std_reward": std_reward})
#     if return_episode_rewards:
#         return episode_rewards, episode_lengths, success_rate
#
#     return mean_reward, std_reward


# def evaluate_policy_def(
#     model,
#     trained_agent,
#     trained_adv,
#     env,
#     n_eval_episodes = 10,
#     deterministic = True,
#     render = False,
#     callback = None,
#     reward_threshold = None,
#     return_episode_rewards = False,
#     warn = True,
#     unlimited_attack = False,
#     attack_method = 'fgsm',
#     attack_eps = 0.01
# ):
#     """
#     Runs policy for ``n_eval_episodes`` episodes and returns average reward.
#     If a vector env is passed in, this divides the episodes to evaluate onto the
#     different elements of the vector env. This static division of work is done to
#     remove bias. See https://github.com/DLR-RM/stable-baselines3/issues/402 for more
#     details and discussion.
#
#     .. note::
#         If environment has not been wrapped with ``Monitor`` wrapper, reward and
#         episode lengths are counted as it appears with ``env.step`` calls. If
#         the environment contains wrappers that modify rewards or episode lengths
#         (e.g. reward scaling, early episode reset), these will affect the evaluation
#         results as well. You can avoid this by wrapping environment with ``Monitor``
#         wrapper before anything else.
#
#     :param model: The RL agent you want to evaluate. This can be any object
#         that implements a `predict` method, such as an RL algorithm (``BaseAlgorithm``)
#         or policy (``BasePolicy``).
#     :param env: The gym environment or ``VecEnv`` environment.
#     :param n_eval_episodes: Number of episode to evaluate the agent
#     :param deterministic: Whether to use deterministic or stochastic actions
#     :param render: Whether to render the environment or not
#     :param callback: callback function to do additional checks,
#         called after each step. Gets locals() and globals() passed as parameters.
#     :param reward_threshold: Minimum expected reward per episode,
#         this will raise an error if the performance is not met
#     :param return_episode_rewards: If True, a list of rewards and episode lengths
#         per episode will be returned instead of the mean.
#     :param warn: If True (default), warns user about lack of a Monitor wrapper in the
#         evaluation environment.
#     :return: Mean reward per episode, std of reward per episode.
#         Returns ([float], [int]) when ``return_episode_rewards`` is True, first
#         list containing per-episode rewards and second containing per-episode lengths
#         (in number of steps).
#     """
#     is_monitor_wrapped = False
#     # Avoid circular import
#     from stable_baselines3.common.monitor import Monitor
#
#     if not isinstance(env, VecEnv):
#         env = DummyVecEnv([lambda: env])  # type: ignore[list-item, return-value]
#
#     is_monitor_wrapped = is_vecenv_wrapped(env, VecMonitor) or env.env_is_wrapped(Monitor)[0]
#
#     if not is_monitor_wrapped and warn:
#         warnings.warn(
#             "Evaluation environment is not wrapped with a ``Monitor`` wrapper. "
#             "This may result in reporting modified episode lengths and rewards, if other wrappers happen to modify these. "
#             "Consider wrapping environment first with ``Monitor`` wrapper.",
#             UserWarning,
#         )
#
#     n_envs = env.num_envs
#     episode_rewards = []
#     episode_lengths = []
#
#     episode_counts = np.zeros(n_envs, dtype="int")
#     # Divides episodes among different sub environments in the vector as evenly as possible
#     episode_count_targets = np.array([(n_eval_episodes + i) // n_envs for i in range(n_envs)], dtype="int")
#
#     current_rewards = np.zeros(n_envs)
#     current_lengths = np.zeros(n_envs, dtype="int")
#     observations = env.reset()
#     infos_adv = env.reset_infos[0]
#     success_count = 0
#
#     states = None
#     episode_starts = np.ones((env.num_envs,), dtype=bool)
#     while (episode_counts < episode_count_targets).any():
#         with th.no_grad():
#             # Convert to pytorch tensor or to TensorDict
#             obs_tensor = obs_as_tensor(observations, model.device)
#
#             if isinstance(trained_agent, FniNet):
#                 actions, std, _action = trained_agent(obs_tensor)
#                 actions = actions.detach().cpu().numpy()
#             else:
#                 actions, _states = trained_agent.predict(obs_tensor.cpu(), deterministic=True)
#
#         actions_tensor = th.tensor(actions, device=obs_tensor.device)  # 确保在同一设备上
#
#         # infos 可能是 dict 或 list
#         if isinstance(infos_adv, dict):
#             info0 = infos_adv
#         elif isinstance(infos_adv, (list, tuple)) and len(infos_adv) > 0:
#             info0 = infos_adv[0]
#         else:
#             raise ValueError(f"Invalid infos format: {type(infos_adv)}")
#
#         # 如果没有 attReStep 键则报错
#         if 'attReStep' not in info0:
#             raise KeyError(f"'attReStep' key not found in info: {info0}")
#
#         attReStep = float(info0['attReStep'])
#
#         attReStep_tensor = th.tensor(attReStep, device=obs_tensor.device).reshape(1, 1)
#
#         obs_adv = th.cat([obs_tensor, attReStep_tensor, actions_tensor], dim=-1)  # -> (batch, 28)
#         print("DEBUG: adv_obs shape: ", obs_adv.shape)
#
#         adv_action, _ = trained_adv.predict(obs_adv.cpu(), deterministic=True)
#
#
#         adv_action_mask = (adv_action[:, 0] > 0) & (obs_adv[:, -2].cpu().numpy() > 0)
#
#         if adv_action_mask:
#             adv_action_eps = adv_action[:, 1]
#             print("DEBUG: adv_action_eps: ", adv_action_eps)
#             state_eps = FGSM_v2(adv_action_eps, victim_agent=trained_agent,
#                                 last_state=obs_tensor, epsilon=attack_eps, device=obs_tensor.device)
#             obs_tensor = state_eps
#         if isinstance(obs_tensor, th.Tensor):
#             obs_tensor = obs_tensor.detach().cpu().numpy()
#
#         final_actions, states = model.predict(
#             obs_tensor,  # type: ignore[arg-type]
#             state=states,
#             episode_start=episode_starts,
#             deterministic=deterministic,
#         )
#         combined_action = np.array([final_actions, adv_action_mask], dtype=np.float32)
#
#         new_observations, rewards, dones, infos = env.step(combined_action)
#         current_rewards += rewards
#         current_lengths += 1
#
#         if isinstance(infos, dict):
#             info0 = infos
#         elif isinstance(infos, (list, tuple)) and len(infos) > 0:
#             info0 = infos[0]
#         else:
#             raise ValueError(f"Invalid infos format: {type(infos)}")
#
#         xa = info0.get("x_position", None)
#         ya = info0.get("y_position", None)
#
#
#
#         for i in range(n_envs):
#             if episode_counts[i] < episode_count_targets[i]:
#                 # unpack values so that the callback can access the local variables
#                 reward = rewards[i]
#                 done = dones[i]
#                 info = infos[i]
#                 episode_starts[i] = done
#
#                 if callback is not None:
#                     callback(locals(), globals())
#
#                 if dones[i]:
#                     if is_monitor_wrapped:
#                         # Atari wrapper can send a "done" signal when
#                         # the agent loses a life, but it does not correspond
#                         # to the true end of episode
#                         if "episode" in info.keys():
#                             # Do not trust "done" with episode endings.
#                             # Monitor wrapper includes "episode" key in info if environment
#                             # has been wrapped with it. Use those rewards instead.
#                             episode_rewards.append(info["episode"]["r"])
#                             episode_lengths.append(info["episode"]["l"])
#                             # Only increment at the real end of an episode
#                             episode_counts[i] += 1
#                     else:
#                         episode_rewards.append(current_rewards[i])
#                         episode_lengths.append(current_lengths[i])
#                         episode_counts[i] += 1
#
#                     if xa is not None and ya is not None:
#                         if xa < -50.0 and ya > 4.0 and not any(dones[i]):  # 还没结束时达成目标
#                             success_count += 1
#                     current_rewards[i] = 0
#                     current_lengths[i] = 0
#
#         observations = new_observations
#         infos_adv = infos
#
#         if render:
#             env.render()
#
#     mean_reward = np.mean(episode_rewards)
#     std_reward = np.std(episode_rewards)
#     success_rate = success_count / n_eval_episodes
#     if reward_threshold is not None:
#         assert mean_reward > reward_threshold, "Mean reward below threshold: " f"{mean_reward:.2f} < {reward_threshold:.2f}"
#     # swanlab.log({"mean_reward": mean_reward, "std_reward": std_reward})
#     if return_episode_rewards:
#         return episode_rewards, episode_lengths, success_rate
#
#     return mean_reward, std_reward

def attack_process(obs_tensor, adv_action_mask, clipped_adv_actions, actions, attack_method, trained_agent, device, attack_eps):
    if adv_action_mask.any():
        attack_idx = np.where(adv_action_mask)[0]

        selected_states = obs_tensor[attack_idx, :-2]
        print('selected_states shape:', selected_states.shape)
        selected_adv_actions = clipped_adv_actions[attack_idx, 1]
        print('selected_adv_actions shape:', selected_adv_actions)

        if attack_method == 'fgsm':
            adv_state = FGSM_v2(selected_adv_actions, victim_agent=trained_agent,
                                last_state=selected_states, device=device, epsilon=attack_eps)
        # elif attack_method == 'pgd':
        #     adv_state = PGD(selected_adv_actions, trained_agent, selected_states, device=device)
        # elif attack_method == 'cw':
        #     adv_state = cw_attack_v2(trained_agent, selected_states, selected_adv_actions)

        if attack_method == 'direct':
            final_action = actions.copy()
            final_action[attack_idx] = selected_adv_actions.detach().cpu().numpy() if th.is_tensor(
                selected_adv_actions) else selected_adv_actions
        else:
            if isinstance(trained_agent, (FniNet, IGCARLNet)):
                adv_action_fromState, _, _ = trained_agent(adv_state)
                adv_action = adv_action_fromState.detach().cpu().numpy()
            else:
                adv_action_fromState, _ = trained_agent.predict(adv_state.cpu(), deterministic=True)
                adv_action = adv_action_fromState
        # print('clip',clipped_adv_actions,'adv_actions:', adv_actions, 'adv_final_action', action, 'actions:', actions, 'remain attack times ', obs_tensor[:, -2].cpu().numpy())
        final_action = actions.copy()
        final_action[attack_idx] = adv_action
    else:
        final_action = actions.copy()
    # Concat final_action with adv_action_mask
    output_action = np.column_stack((final_action, adv_action_mask.astype(np.float32) ))

    return output_action
