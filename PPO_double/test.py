import torch
import gym
from torch.distributions import Normal
from network import network
if __name__ == '__main__':
    env=gym.make("InvertedPendulum-v4", render_mode="human")
    test=network(1)
    test.load_state_dict(torch.load('ppo_actor.pth'))
    
    for i in range(100000):
        done=False
        obs,_=env.reset()
        while not done:
            input=torch.tensor(obs,dtype=torch.float32)
            action=test(input)
            dist=Normal(action,torch.tensor(0.5))
            action=dist.sample()
            obs,_,done,_,_=env.step(action)
            env.render()
        
