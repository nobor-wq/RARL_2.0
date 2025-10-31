from stable_baselines3.common.callbacks import EvalCallback, EventCallback
from stable_baselines3.common.vec_env import DummyVecEnv, VecEnv, sync_envs_normalization
from evaluation import evaluate_policy_adv, evaluate_policy_def
import numpy as np
import os
from stable_baselines3.common.monitor import Monitor


class CustomEvalCallback_adv(EvalCallback):
    def __init__(
            self,
            eval_env,
            trained_agent,
            callback_on_new_best = None,
            callback_after_eval = None,
            n_eval_episodes = 10,
            eval_freq = 10000,
            log_path = None,
            best_model_save_path = None,
            deterministic = True,
            render = False,
            verbose = 1,
            unlimited_attack = False,
            attack_method = 'fgsm',
            attack_eps = 0.01,
    ):
        super().__init__(eval_env, n_eval_episodes=n_eval_episodes, eval_freq=eval_freq,
                         best_model_save_path=best_model_save_path, deterministic=deterministic, render=render, verbose=verbose)

        self.callback_on_new_best = callback_on_new_best
        if self.callback_on_new_best is not None:
            # Give access to the parent
            self.callback_on_new_best.parent = self
        self.trained_agent = trained_agent
        self.n_eval_episodes = n_eval_episodes
        self.eval_freq = eval_freq
        self.best_mean_reward = np.inf
        self.last_mean_reward = np.inf
        self.deterministic = deterministic
        self.render = render
        self.unlimited_attack = unlimited_attack
        self.attack_method = attack_method
        self.attack_eps = attack_eps
        # Convert to VecEnv for consistency
        if not isinstance(eval_env, VecEnv):
            eval_env = DummyVecEnv([lambda: eval_env])  # type: ignore[list-item, return-value]
        # if not isinstance(eval_env, Monitor):
        #     eval_env = Monitor(eval_env)
        self.eval_env = eval_env
        self.best_model_save_path = best_model_save_path
        # Logs will be written in ``evaluations.npz``
        if log_path is not None:
            log_path = os.path.join(log_path, "evaluations")
        self.log_path = log_path
        self.evaluations_results = []
        self.evaluations_timesteps = []
        self.evaluations_length = []
        # For computing success rate
        self._is_success_buffer = []
        self.evaluations_successes = []


    def _on_step(self) -> bool:
        continue_training = True

        if self.eval_freq > 0 and self.n_calls % self.eval_freq == 0:
            # Sync training and eval env if there is VecNormalize
            if self.model.get_vec_normalize_env() is not None:
                try:
                    sync_envs_normalization(self.training_env, self.eval_env)
                except AttributeError as e:
                    raise AssertionError(
                        "Training and eval env are not wrapped the same way, "
                        "see https://stable-baselines3.readthedocs.io/en/master/guide/callbacks.html#evalcallback "
                        "and warning above."
                    ) from e

            # Reset success rate buffer
            self._is_success_buffer = []

            episode_rewards, episode_lengths, success_rate, attack_count, attack_success = evaluate_policy_adv(
                self.model,
                self.trained_agent,
                self.eval_env,
                n_eval_episodes=self.n_eval_episodes,
                render=self.render,
                deterministic=self.deterministic,
                return_episode_rewards=True,
                warn=self.warn,
                unlimited_attack=self.unlimited_attack,
                attack_method=self.attack_method,
                attack_eps = self.attack_eps
            )

            if self.log_path is not None:
                assert isinstance(episode_rewards, list)
                assert isinstance(episode_lengths, list)
                self.evaluations_timesteps.append(self.num_timesteps)
                self.evaluations_results.append(episode_rewards)
                self.evaluations_length.append(episode_lengths)

                kwargs = {}
                # Save success log if present
                if len(self._is_success_buffer) > 0:
                    self.evaluations_successes.append(self._is_success_buffer)
                    kwargs = dict(successes=self.evaluations_successes)

                np.savez(
                    self.log_path,
                    timesteps=self.evaluations_timesteps,
                    results=self.evaluations_results,
                    ep_lengths=self.evaluations_length,
                    **kwargs,
                )

            mean_reward, std_reward = np.mean(episode_rewards), np.std(episode_rewards)
            mean_ep_length, std_ep_length = np.mean(episode_lengths), np.std(episode_lengths)
            self.last_mean_reward = float(mean_reward)

            if self.verbose >= 1:
                print(f"Eval num_timesteps={self.num_timesteps}, " f"episode_reward={mean_reward:.2f} +/- {std_reward:.2f}")
                print(f"Episode length: {mean_ep_length:.2f} +/- {std_ep_length:.2f}")
                print(f"success_rate={success_rate:.2f}, attack_count={attack_count:.2f}, attack_success={attack_success:.2f}")

            # Add to current Logger
            self.logger.record("eval_adv/mean_reward", float(mean_reward))
            self.logger.record("eval_adv/mean_ep_length", mean_ep_length)
            self.logger.record("eval_adv/success_rate", success_rate)
            self.logger.record("eval_adv/attack_count", attack_count)
            self.logger.record("eval_adv/attack_success", attack_success)

            # Dump log so the evaluation results are printed with the correct timestep
            self.logger.record("time_adv/total_timesteps", self.num_timesteps, exclude="tensorboard")
            self.logger.dump(self.num_timesteps)

            if mean_reward < self.best_mean_reward:
                if self.verbose >= 1:
                    print("New best mean reward!")
                if self.best_model_save_path is not None:
                    self.model.save(os.path.join(self.best_model_save_path, "best_model"))
                self.best_mean_reward = float(mean_reward)
                # Trigger callback on new best model, if needed
                if self.callback_on_new_best is not None:
                    continue_training = self.callback_on_new_best.on_step()

            # Trigger callback after every evaluation, if needed
            if self.callback is not None:
                continue_training = continue_training and self._on_event()

        return continue_training

class CustomEvalCallback_def(EvalCallback):
    def __init__(
            self,
            eval_env,
            trained_agent,
            trained_adv,
            attack_eps,
            callback_on_new_best=None,
            callback_after_eval=None,
            n_eval_episodes=10,
            eval_freq=10000,
            log_path=None,
            best_model_save_path=None,
            deterministic=True,
            render=False,
            verbose=1,
            unlimited_attack=False,
            attack_method='fgsm',
    ):
        super().__init__(eval_env, n_eval_episodes=n_eval_episodes, eval_freq=eval_freq,
                         best_model_save_path=best_model_save_path, deterministic=deterministic, render=render,
                         verbose=verbose)

        self.callback_on_new_best = callback_on_new_best
        if self.callback_on_new_best is not None:
            # Give access to the parent
            self.callback_on_new_best.parent = self
        self.trained_agent = trained_agent
        self.trained_adv = trained_adv
        self.n_eval_episodes = n_eval_episodes
        self.eval_freq = eval_freq
        self.best_mean_reward = -np.inf
        self.last_mean_reward = -np.inf
        self.deterministic = deterministic
        self.render = render
        self.unlimited_attack = unlimited_attack
        self.attack_method = attack_method
        self.attack_eps = attack_eps
        # Convert to VecEnv for consistency
        if not isinstance(eval_env, VecEnv):
            eval_env = DummyVecEnv([lambda: eval_env])  # type: ignore[list-item, return-value]

        self.eval_env = eval_env
        self.best_model_save_path = best_model_save_path
        # Logs will be written in ``evaluations.npz``
        if log_path is not None:
            log_path = os.path.join(log_path, "evaluations")
        self.log_path = log_path
        self.evaluations_results = []
        self.evaluations_timesteps = []
        self.evaluations_length = []
        # For computing success rate
        self._is_success_buffer = []
        self.evaluations_successes = []

    def _on_step(self) -> bool:
        continue_training = True

        if self.eval_freq > 0 and self.n_calls % self.eval_freq == 0:
            # Sync training and eval env if there is VecNormalize
            if self.model.get_vec_normalize_env() is not None:
                try:
                    sync_envs_normalization(self.training_env, self.eval_env)
                except AttributeError as e:
                    raise AssertionError(
                        "Training and eval env are not wrapped the same way, "
                        "see https://stable-baselines3.readthedocs.io/en/master/guide/callbacks.html#evalcallback "
                        "and warning above."
                    ) from e

            # Reset success rate buffer
            self._is_success_buffer = []

            episode_rewards, episode_lengths, success_rate, attack_count, attack_success = evaluate_policy_def(
                self.model,
                self.trained_agent,
                self.trained_adv,
                self.eval_env,
                n_eval_episodes=self.n_eval_episodes,
                render=self.render,
                deterministic=self.deterministic,
                return_episode_rewards=True,
                warn=self.warn,
                unlimited_attack=self.unlimited_attack,
                attack_method=self.attack_method,
                attack_eps = self.attack_eps
            )

            if self.log_path is not None:
                assert isinstance(episode_rewards, list)
                assert isinstance(episode_lengths, list)
                self.evaluations_timesteps.append(self.num_timesteps)
                self.evaluations_results.append(episode_rewards)
                self.evaluations_length.append(episode_lengths)

                kwargs = {}
                # Save success log if present
                if len(self._is_success_buffer) > 0:
                    self.evaluations_successes.append(self._is_success_buffer)
                    kwargs = dict(successes=self.evaluations_successes)

                np.savez(
                    self.log_path,
                    timesteps=self.evaluations_timesteps,
                    results=self.evaluations_results,
                    ep_lengths=self.evaluations_length,
                    **kwargs,
                )

            mean_reward, std_reward = np.mean(episode_rewards), np.std(episode_rewards)
            mean_ep_length, std_ep_length = np.mean(episode_lengths), np.std(episode_lengths)
            self.last_mean_reward = float(mean_reward)

            if self.verbose >= 1:
                print(
                    f"Eval num_timesteps={self.num_timesteps}, " f"episode_reward={mean_reward:.2f} +/- {std_reward:.2f}")
                print(f"Episode length: {mean_ep_length:.2f} +/- {std_ep_length:.2f}")

                print(f"success_rate={success_rate:.2f}, attack_count={attack_count:.2f}, attack_success={attack_success:.2f}")

            # Add to current Logger
            self.logger.record("eval_def/mean_reward", float(mean_reward))
            self.logger.record("eval_def/mean_ep_length", mean_ep_length)
            self.logger.record("eval_def/success_rate", success_rate)
            self.logger.record("eval_def/attack_count", attack_count)
            self.logger.record("eval_def/attack_success", attack_success)


            # Dump log so the evaluation results are printed with the correct timestep
            self.logger.record("time_def/total_timesteps", self.num_timesteps, exclude="tensorboard")
            self.logger.dump(self.num_timesteps)

            if mean_reward > self.best_mean_reward:
                if self.verbose >= 1:
                    print("New best mean reward!")
                if self.best_model_save_path is not None:
                    self.model.save(os.path.join(self.best_model_save_path, "best_model"))
                self.best_mean_reward = float(mean_reward)
                # Trigger callback on new best model, if needed
                if self.callback_on_new_best is not None:
                    continue_training = self.callback_on_new_best.on_step()

            # Trigger callback after every evaluation, if needed
            if self.callback is not None:
                continue_training = continue_training and self._on_event()

        return continue_training