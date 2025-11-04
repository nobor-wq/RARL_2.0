import numpy as np
import torch as th
from stable_baselines3.common.utils import obs_as_tensor
from fgsm import FGSM_v2
from policy import FniNet, IGCARLNet
from stable_baselines3.common.vec_env import DummyVecEnv, VecEnv, VecMonitor, is_vecenv_wrapped




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
                if info0.get('TimeLimit.truncated', False):
                    # 如果这是一个纯粹的超时，我们再检查位置是否正确。
                    if xa is not None and ya is not None and (xa < -50.0 and ya > 4.0):
                        success_count += 1
                break
            observations = new_observations

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
                if info1.get('TimeLimit.truncated', False):
                    # 如果这是一个纯粹的超时，我们再检查位置是否正确。
                    if xa is not None and ya is not None and (xa < -50.0 and ya > 4.0):
                        success_count += 1
                break

            observations = new_observations

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


def attack_process(obs_tensor, adv_action_mask, clipped_adv_actions, actions, attack_method, trained_agent, device, attack_eps):
    if adv_action_mask.any():
        attack_idx = np.where(adv_action_mask)[0]

        selected_states = obs_tensor[attack_idx, :-2]
        selected_adv_actions = clipped_adv_actions[attack_idx, 1]
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

        final_action = actions.copy()
        final_action[attack_idx] = adv_action
    else:
        final_action = actions.copy()
    # Concat final_action with adv_action_mask
    output_action = np.column_stack((final_action, adv_action_mask.astype(np.float32) ))

    return output_action
