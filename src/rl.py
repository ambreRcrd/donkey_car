import gym_donkeycar
import gym
import numpy as np
import os
from stable_baselines3 import PPO
from stable_baselines3.common.evaluation import evaluate_policy

from stable_baselines3.common.callbacks import BaseCallback

class RewardCallback(BaseCallback):
    def __init__(self, verbose=0):
        super(RewardCallback, self).__init__(verbose)
        self.episode_rewards = []
        self.episode_lengths = []

    def _on_step(self) -> bool:
        if self.locals['dones'][0]:
            self.episode_rewards.append(sum(self.locals['rewards']))
            self.episode_lengths.append(self.locals['infos'][0].get('episode', {}).get('l', 0))
            print(f"Episode reward: {self.episode_rewards[-1]} length: {self.episode_lengths[-1]}")

            # Print mean reward and length every N steps
            if self.num_timesteps % 1000 == 0:
                mean_reward = sum(self.episode_rewards[-10:]) / 10.0 if len(self.episode_rewards) >= 10 else 0.0
                mean_length = sum(self.episode_lengths[-10:]) / 10.0 if len(self.episode_lengths) >= 10 else 0.0
                print(f"Step: {self.num_timesteps}, Mean reward: {mean_reward}, Mean length: {mean_length}")

        return True

    def _on_training_end(self) -> None:
        print(f"Training completed. Mean reward: {np.mean(self.episode_rewards)} over {len(self.episode_rewards)} episodes")

def create_env():
	env = gym.make("donkey-waveshare-v0")
	return env

def learn(env, saving_path="../outputs/test.keras"):
    if os.path.exists(saving_path):
        print(f"Loading existing model from {saving_path}")
        model = PPO.load(saving_path, env=env, verbose=1)
    else:
        model = PPO("CnnPolicy", env, n_steps=200, verbose=1)

    callback = RewardCallback(verbose=1)
    print('Learning in process...')
    model.learn(total_timesteps=10_000, callback=callback)
    print('Learning done')
    print(f"Saving model at {saving_path}")
    model.save(saving_path)

def evaluate_policy(env, saving_path="../outputs/test_2.keras"):
	model = PPO.load(saving_path, verbose=1)
	print("Evaluating the policy...")
	mean_reward, std_reward = evaluate_policy(model, env)

	print(f"Mean reward over {n_eval_episodes} episodes: {mean_reward} +/- {std_reward}")


def run_model(env, saving_path="../outputs/test.keras"):
	model = PPO.load(saving_path, verbose=1)

	obs = env.reset()
	done = False
	total_reward = 0.0

	while not done:
	    action, _ = model.predict(obs, deterministic=True)  # Prédire l'action à prendre
	    obs, reward, done, info = env.step(action)  # Exécuter l'action et obtenir le résultat
	    total_reward += reward

	    # Visualisation ou impression des informations d'étape
	    env.render()

	print(f"Total reward: {total_reward}")
	env.close()

if __name__ == "__main__":
	env = create_env()
	learn(env)
	#evaluate_policy(env)
	run_model(env)




