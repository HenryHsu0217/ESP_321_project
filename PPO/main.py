import gym
import torch
from torch import optim
from network import network
import numpy as np
from PPO import PPO
if __name__ == '__main__':
    env = gym.make('InvertedPendulum-v4')
    hyperparameters = {
				'timesteps_per_batch': 2048, 
				'max_timesteps_per_episode': 200, 
				'gamma': 0.99, 
				'n_updates_per_iteration': 10,
				'lr_actor': 0.000001, 
                'lr_critic': 0.0001, 
				'clip': 0.2,
				'render': True,
				'render_every_i': 10
			  }
    ppo=PPO(env,**hyperparameters)
    ppo.learn(total_timesteps=200_000_000)

    