import gym
import numpy as np
from tqdm import tqdm
import random
import threading
import random
env = gym.make("InvertedPendulum-v4", render_mode="human")
action = np.array(0)
Kp =4.3
Ki = 0.15
Kd = 0.1
k1 = 4.3
k2 = 0.1
k3 = 0.1
k4 = 0.2
max = 0
current = 0
observation, info = env.reset(seed=42)
data_to_save=[]
def get_user_input(force_input,stop_signal):
    while not stop_signal[0]:
        user_input = input()
        if user_input:
            force_input.append(user_input)
stop_signal = [False]
force_input = []
force=0
threading.Thread(target=get_user_input, args=(force_input, stop_signal), daemon=True).start()
for _ in  tqdm (range (10000), desc="Loading..."):
    current+=1
    if force_input:
            force = force_input[0]
            force_input.clear()
    #####################################################
    # Comment out this if want to apply force by manual
    if _%random.randint(1,100)==0:
        force= random.randint(-10,10)
    #####################################################
    action=np.append(action,force)
    observation, reward, terminated, truncated, info = env.step(action)
    if terminated or truncated:
        observation, info = env.reset()
        current = 0
        errorsum = 0
        preverror = 0
        force=0
        stop_signal = [True]
    angle = observation[1]
    angularvel = observation[3]
    position = observation[0]
    velocity = observation[2]
    action = k1 * angle + k2 * angularvel + k3 * position + k4 * velocity
    action = np.clip(action, -3, 3)
    if angle<0.1:
        data=np.append(observation,np.array(action))
        data_to_save.append(data)
    if current>max:
        max = current
env.close()
print(len(data_to_save))
np.savez('./Pure_NN_single/Datas/data_with__random_force.npz', *data_to_save)
