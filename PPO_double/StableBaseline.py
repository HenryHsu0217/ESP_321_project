import gymnasium as gym
import gym as Gym
from stable_baselines3 import PPO
from stable_baselines3.common.env_util import make_vec_env

# Parallel environments
vec_env = make_vec_env("InvertedDoublePendulum-v4")

model = PPO("MlpPolicy", vec_env, verbose=1)
model.learn(total_timesteps=1000000)
model.save("ppo_cartpole")
del model 

model = PPO.load("ppo_cartpole")
test_env=Gym.make("InvertedDoublePendulum-v4",render_mode="human")
obs = test_env.reset()
obs=obs[0]
for _ in range(1000000):
    done=False
    while not done:
        action, _states = model.predict(obs)
        obs, rewards, done ,_,_= test_env.step(action)
    test_env.reset()