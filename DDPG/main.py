from reward import ProgressReward
from trackline import TrackLine
import gym
import numpy as np
import os

map_path = os.path.join(os.getcwd(), 'maps', 'aut_wide')
track = TrackLine('aut_wide',racing_line=False)
reward_fn = ProgressReward(track)

env = gym.make('f110_gym:f110-v0', map=map_path,map_ext='.png', num_agents=1,render_mode='human')
poses = np.array([[0,0,np.pi]])
obs,step_reward,done,info = env.reset(poses)
prev_obs = None

while not done:
    action = np.array([[0,0.5]])
    new_obs,step_reward,done,info = env.step(action)
    reward = reward_fn(new_obs,obs, action)
    env.render()

    print(f"Reward: {reward:.3f}")

    prev_obs = obs

