import gymnasium as gym
import gym as Gym
from stable_baselines3 import PPO
from stable_baselines3.common.env_util import make_vec_env
import threading
import numpy as np
def get_user_input(force_input,stop_signal):
    while not stop_signal[0]:
        user_input = input()
        if user_input:
            force_input.append(user_input)
"""vec_env = make_vec_env("InvertedPendulum-v4")

model = PPO("MlpPolicy", vec_env, verbose=1)
model.learn(total_timesteps=100000)
model.save("ppo_cartpole")
del model 
"""
model = PPO.load("ppo_cartpole")
test_env=Gym.make("InvertedPendulum-v4",render_mode="human")
obs = test_env.reset()
obs=obs[0]
force = 0
force_input = []
stop_signal = [False]
threading.Thread(target=get_user_input, args=(force_input, stop_signal), daemon=True).start()
while True:
    if force_input:
                force = force_input[0]
                force_input.clear()
    action, _states = model.predict(obs)
    action=np.append(action,force)
    obs, rewards, dones ,_,_= test_env.step(action)
