import gym
import time
if __name__ == '__main__':
    env = gym.make('InvertedDoublePendulum-v4',render_mode="human")
    num_steps = 1500

    obs = env.reset()
    for step in range(num_steps):
        # take random action
        action = env.action_space.sample()
        # apply the action
        obs,a,b,c,d= env.step(action)
        print(obs)
        # Render the env
        env.render()
        time.sleep(0.001)
    env.close()