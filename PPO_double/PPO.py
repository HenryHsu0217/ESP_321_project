from network import network
from torch.optim import Adam,SGD
from torch.distributions import Normal
import torch
from torch import nn
import numpy as np
import time
class PPO:
    def __init__(self,env,**hyperparameters):
        self._init_hyperparameters(hyperparameters)
        self.env = env
        self.lr= 3e-4
        self.actor = network(1)                                  
        self.critic = network(1)
        self.actor_optim = SGD(self.actor.parameters(), lr=self.lr_actor)
        self.critic_optim = SGD(self.critic.parameters(), lr=self.lr_critic)
        self.cov_var = torch.tensor(0.5)
        self.logger = {
			'delta_t': time.time_ns(),
			't_so_far': 0,          
			'i_so_far': 0,           
			'batch_lens': [],       
			'batch_rews': [],        
			'actor_losses': [],     
            'critic_losses': [],
		}
    def learn(self, total_timesteps):
        print(f"Learning... Running {self.max_timesteps_per_episode} timesteps per episode, ", end='')
        print(f"{self.timesteps_per_batch} timesteps per batch for a total of {total_timesteps} timesteps")
        t_so_far = 0 
        i_so_far = 0 
        while t_so_far < total_timesteps:    
            batch_obs, batch_acts, batch_log_probs, batch_rtgs, batch_lens = self.collect_trajectories()
            t_so_far += np.sum(batch_lens)
            i_so_far += 1
            self.logger['t_so_far'] = t_so_far
            self.logger['i_so_far'] = i_so_far
            for i in range(len(batch_obs)):
                V, curr_log_probs = self.evaluate(batch_obs[i], batch_acts[i])
                A_k = V.detach()-batch_rtgs[i] 
                A_k = (A_k - A_k.mean()) / (A_k.std() + 1e-10)    
                ratios = torch.exp(curr_log_probs - batch_log_probs[i])
                surr1 = ratios * A_k
                surr2 = torch.clamp(ratios, 1 - self.clip, 1 + self.clip) * A_k
                #print(torch.min(surr1, surr2).size())
                actor_loss = (-torch.min(surr1, surr2)).mean()
                critic_loss = nn.MSELoss()
                critic_losses = critic_loss(V, batch_rtgs[i])
                self.actor_optim.zero_grad()
                actor_loss.backward(retain_graph=True)
                self.actor_optim.step()
                self.critic_optim.zero_grad()
                critic_losses.backward()
                self.critic_optim.step()
                self.logger['actor_losses'].append(actor_loss.detach())
                self.logger['critic_losses'].append(critic_losses.detach())
            self._log_summary()
            if i_so_far % self.save_freq == 0:
                torch.save(self.actor.state_dict(), './ppo_actor.pth')
                torch.save(self.critic.state_dict(), './ppo_critic.pth')
    def evaluate(self, batch_obs, batch_acts):
        V=self.critic(batch_obs).squeeze()
        mean=self.actor(batch_obs)
        dist = Normal(mean, self.cov_var)
        log_probs = dist.log_prob(batch_acts)
        return V,log_probs
    def collect_trajectories(self):
        batch_obs = []
        batch_acts = []
        batch_log_probs = []
        batch_rews = []
        batch_rtgs = []
        batch_lens = []
        ep_rews = []
        t = 0
        while t < self.timesteps_per_batch:
            obs,_ = self.env.reset()
            ep_rews = []
            ep_obs=[]
            ep_acts=[]
            ep_log_probs=[]
            done = False
            for ep_t in range(self.max_timesteps_per_episode):
                t += 1
                ep_obs.append(obs)
                action, log_prob = self.get_action(obs)
                obs, rew, done, _ ,_= self.env.step(action)
                ep_rews.append(rew)
                ep_acts.append(action)
                ep_log_probs.append(log_prob)
                if done:
                    break
            batch_lens.append(ep_t + 1)
            batch_rews.append(torch.tensor(ep_rews, dtype=torch.float))
            batch_acts.append(torch.tensor(ep_acts, dtype=torch.float))
            batch_obs.append(torch.tensor(ep_obs, dtype=torch.float))
            batch_log_probs.append(torch.tensor(ep_log_probs, dtype=torch.float))
        batch_rtgs = self.compute_rtgs(batch_rews)
        self.logger['batch_rews'] = batch_rews
        self.logger['batch_lens'] = batch_lens
        return batch_obs, batch_acts, batch_log_probs, batch_rtgs, batch_lens
    def compute_rtgs(self, batch_rews):
        batch_rtgs = []
        for rws in reversed(batch_rews):
            ep_rew=[]
            discounted_reward = 0 
            for rew in reversed(rws):
                discounted_reward = rew.item()+ discounted_reward * self.gamma
                ep_rew.insert(0, discounted_reward)
            batch_rtgs.insert(0, torch.tensor(ep_rew, dtype=torch.float,requires_grad=True))
        return batch_rtgs
    def get_action(self, obs):
        mean = self.actor(obs)
        dist = Normal(mean, self.cov_var)
        action = dist.sample()
        log_prob = dist.log_prob(action)
        return action.detach().numpy(), log_prob.detach()
    def _init_hyperparameters(self, hyperparameters):
        self.timesteps_per_batch = 4800                  
        self.max_timesteps_per_episode = 1600            
        self.n_updates_per_iteration = 5                
        self.lr_actor = 0.005   
        self.lr_critic= 0.005                              
        self.gamma = 0.95                               
        self.clip = 0.2                                 

        self.render = True                              
        self.render_every_i = 10                        
        self.save_freq = 10                              
        self.seed = None                                 
        for param, val in hyperparameters.items():
            exec('self.' + param + ' = ' + str(val))

        
        if self.seed != None:
             
            assert(type(self.seed) == int)

            # Set the seed 
            torch.manual_seed(self.seed)
            print(f"Successfully set seed to {self.seed}")
    def _log_summary(self):
        delta_t = self.logger['delta_t']
        self.logger['delta_t'] = time.time_ns()
        delta_t = (self.logger['delta_t'] - delta_t) / 1e9
        delta_t = str(round(delta_t, 2))

        t_so_far = self.logger['t_so_far']
        i_so_far = self.logger['i_so_far']
        avg_ep_lens = np.mean(self.logger['batch_lens'])
        avg_ep_rews = np.mean([torch.sum(ep_rews) for ep_rews in self.logger['batch_rews']])
        avg_actor_loss = np.mean([losses.float().mean() for losses in self.logger['actor_losses']])
        avg_critic_loss=np.mean([losses.float().mean() for losses in self.logger['critic_losses']])
         
        avg_ep_lens = str(round(avg_ep_lens, 2))
        avg_ep_rews = str(round(avg_ep_rews, 2))
        avg_actor_loss = str(round(avg_actor_loss, 10))
        avg_critic_loss=str(round(avg_critic_loss, 10))
         
        print(flush=True)
        print(f"-------------------- Iteration #{i_so_far} --------------------", flush=True)
        print(f"Average Episodic Length: {avg_ep_lens}", flush=True)
        print(f"Average Episodic Return: {avg_ep_rews}", flush=True)
        print(f"Average actor Loss: {avg_actor_loss}", flush=True)
        print(f"Average critic Loss: {avg_critic_loss}", flush=True)
        print(f"Timesteps So Far: {t_so_far}", flush=True)
        print(f"Iteration took: {delta_t} secs", flush=True)
        print(f"------------------------------------------------------", flush=True)
        print(flush=True)

        # Reset batch-specific logging data
        self.logger['batch_lens'] = []
        self.logger['batch_rews'] = []
        self.logger['actor_losses'] = []
