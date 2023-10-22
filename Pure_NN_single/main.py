import  torch
from NN import NeuralNetwork
import gym
if __name__ == '__main__':
    """Testing with the gym enviroment"""
    model = NeuralNetwork()
    model.load_state_dict(torch.load('./Trained_models/9644.pth'))
    model.eval()
    env = gym.make("InvertedPendulum-v4", render_mode="human")
    episodes=10
    for episode in range(episodes):
        termination=False
        observation,_=env.reset(seed=42)
        while not termination:
            input=torch.tensor(observation,dtype=torch.float32)
            input=input.unsqueeze(0)
            action=model(input).detach().numpy()[0]
            observation, reward, termination, truncated, info=env.step(action)
            env.render()
