import gym
from NN import NeuralNetwork
import torch
import torch.distributions as distributions
import time
if __name__ == '__main__':
    env = gym.make('InvertedPendulum-v4',render_mode="human")
    episodes=100000
    gamma = 0.9
    policy=NeuralNetwork()
    optim=torch.optim.Adam(policy.parameters(),lr=0.0001)
    for episode in range(episodes):
        env.steps=0
        obs, _, = env.reset()
        obs = obs[:4]
        done=False
        memory =[]
        while not done:
            mean, std_dev = policy(torch.tensor(obs).float())
            dist = distributions.Normal(mean, std_dev)
            action = dist.sample()
            log_prob = dist.log_prob(action)
            a = action.detach().numpy()
            obs, reward, done,_,_, = env.step(a)
            #print(reward)
            env.steps+=1
            obs = obs[:4]
            memory.append((obs, a, log_prob, reward, obs, done))
            env.render()
        returns = []
        G = 0
        for _,_, _, reward, _, _ in reversed(memory):
            #print(reward)
            G = reward + gamma * G
            returns.append(G)
        returns.reverse()
        returns = torch.tensor(returns)
        normalized_returns = (returns - returns.mean()) / (returns.std() + 1e-8)
        log_probs = [item[2] for item in memory]
        loss = -torch.sum(torch.stack(log_probs) * normalized_returns)
        optim.zero_grad()
        loss.backward()
        optim.step()
        print(f"Episode {episode}, Loss: {loss.item()}")
    env.close()
    torch.save(policy, '/Users/henryhsu/ESP_321_project/ANN/Saved_model')