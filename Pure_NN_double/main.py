import  torch
from NN import NeuralNetwork
import gym
import numpy as np
import threading
#####################################################
#After running the code, type the force in the commadnline
#and press enter to iteract with the env
#####################################################
def get_user_input(force_input,stop_signal):
    while not stop_signal[0]:
        user_input = input()
        if user_input:
            force_input.append(user_input)
if __name__ == '__main__':
    """Testing with the gym enviroment"""
    model = NeuralNetwork()
    model.load_state_dict(torch.load('./Models/external_force_9400.pth'))
    model.eval()
    env = gym.make("InvertedDoublePendulum-v4", render_mode="human")
    episodes=10
    force_input = []
    for episode in range(episodes):
        termination=False
        observation,_=env.reset(seed=42)
        force = 0
        stop_signal = [False]
        threading.Thread(target=get_user_input, args=(force_input, stop_signal), daemon=True).start()
        while not termination:
            if force_input:
                force = force_input[0]
                force_input.clear()
            input_=torch.tensor(observation[:8],dtype=torch.float32)
            input_=input_.unsqueeze(0)
            action=model(input_).detach().numpy()[0]
            action=np.append(action,force)
            observation, reward, termination, truncated, info=env.step(action)
            env.render()
            force=0
        stop_signal[0] = True

