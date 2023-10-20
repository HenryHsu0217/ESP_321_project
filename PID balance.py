import gymnasium as gym
import numpy as np
env = gym.make("InvertedPendulum-v4", render_mode="human")
action = 0
Kp =4.3
Ki = 0.15
Kd = 0.1
errorsum=0
preverror = 0

max = 0
current = 0
observation, info = env.reset(seed=42)
for _ in range(10000):
    current+=1

    # action = env.action_space.sample()

    observation, reward, terminated, truncated, info = env.step([action])
    
    if terminated or truncated:
        observation, info = env.reset()
        current = 0
        errorsum = 0
        preverror = 0
        print("________________________________________________________________________________________________________")
    # angle = np.arctan2(observation[1],observation[3])
    angle = observation[1]
    error = angle
    errorsum += error
    action = Kp * error + Ki * errorsum + Kd * (error-preverror)
    action = np.clip(action, -1, 1)
    preverror = error
    print(action)

    if current>max:
        max = current
    

env.close()
print(max)