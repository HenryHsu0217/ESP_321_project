import gym as gym
import numpy as np
from tqdm import tqdm
import random
import threading
import random
import time as time

env = gym.make("InvertedDoublePendulum-v4", render_mode="human")
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
    # if _%random.randint(1,100)==0:
    #     force= random.randint(-10,10)
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

    angle1 = np.arctan2(observation[1],observation[3])
    angle2 = np.arctan2(observation[2],observation[4])
    angularvel1 = observation[6]
    angularvel2 = observation[7]
    position = observation[0]
    velocity = observation[5]
    
    action = 2* angle1 + 5 * angle2 + 0.5 * angularvel1 + 0.5 * angularvel2 + 0.1* position + 0.1 * velocity
    action = -np.clip(action, -1, 1)
    print(angle1*57.3,angle2*57.3 , action)
    # if angle<0.1:
    #     data=np.append(observation,np.array(action))
    #     data_to_save.append(data)

env.close()
print(len(data_to_save))
# np.savez('./Pure_NN_double/Datas/data_with__random_force.npz', *data_to_save)
