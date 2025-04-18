from trackline import TrackLine
from Networks import PolicyNetwork, QNetwork
import gym
import numpy as np
import torch
import torch.nn as nn
from torch.nn import functional as F
import os
from Buffers import ReplayMemory
from argparse import Namespace
import matplotlib.pyplot as plt
from observationProcessor import ObservationProcessor  
import random

torch.manual_seed(42)
np.random.seed(42)
random.seed(42)

os.makedirs("ddpg_logs", exist_ok=True)

# ---------- Configuration ----------
conf = Namespace(max_v=5.0, map='Austin')
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
map_path = os.path.join(os.getcwd(), 'maps', conf.map)
track = TrackLine(conf.map, racing_line=False)

# ---------- Initialize Networks ----------
# Update input sizes:
#   Actor input size: 275  (Downsampled LiDAR: 270 + [x, y, theta, v_x, v_y]: 5)
#   Critic input size: 277  (State 275 + action 2)
policy_net = PolicyNetwork(input_size=275, output_size=2).to(device)
policy_net_target = PolicyNetwork(input_size=275, output_size=2).to(device)
_ = policy_net_target.requires_grad_(False)

Q_origin_net = QNetwork(input_size=277, output_size=1).to(device)
Q_target_net = QNetwork(input_size=277, output_size=1).to(device)
Q_target_net.load_state_dict(Q_origin_net.state_dict())

# ---------- Initialize Replay Buffer ----------
buffer = ReplayMemory()

# ---------- OU Noise for Exploration ----------
class OrnsteinUhlenbeckNoise:
    def __init__(self, mu):
        self.theta, self.dt, self.sigma = 0.1, 0.01, 0.1
        self.mu = mu
        self.x_prev = np.zeros_like(self.mu)

    def __call__(self):
        x = self.x_prev + self.theta * (self.mu - self.x_prev) * self.dt + \
            self.sigma * np.sqrt(self.dt) * np.random.normal(size=self.mu.shape)
        self.x_prev = x
        return x

ou_action_noise = OrnsteinUhlenbeckNoise(mu=np.zeros(2))

# ---------- Optimization Function ----------
gamma = 0.99
opt_q = torch.optim.AdamW(Q_origin_net.parameters(), lr=0.001)
opt_mu = torch.optim.AdamW(policy_net.parameters(), lr=0.0005)

def optimize(states, actions, rewards, next_states, dones):
    # Critic loss: Mean squared error between Q(s,a) and y = r + γ Q(s', μ(s'))
    s_a = torch.cat((states, actions), dim=1)
    q_sa = Q_origin_net(s_a)
    
    a_next = policy_net_target(next_states)
    s_a_next = torch.cat((next_states, a_next), dim=1)
    q_sa_next = Q_target_net(s_a_next)
    y = rewards + (1 - dones) * gamma * q_sa_next

    critic_loss = F.mse_loss(q_sa, y.detach())
    opt_q.zero_grad()
    critic_loss.backward()
    torch.nn.utils.clip_grad_norm_(Q_origin_net.parameters(), max_norm=1.0)
    opt_q.step()

    # Actor loss: maximize Q(s, μ(s)), i.e. minimize -Q(s, μ(s))
    a_current = policy_net(states)
    s_a_current = torch.cat((states, a_current), dim=1)
    actor_loss = -Q_origin_net(s_a_current).mean()
    opt_mu.zero_grad()
    actor_loss.backward()
    torch.nn.utils.clip_grad_norm_(policy_net.parameters(), max_norm=1.0)
    opt_mu.step()

    for p in Q_origin_net.parameters():
        p.requires_grad = True  # Re-enable gradients
    return critic_loss.item(), actor_loss.item()

# ---------- Target Network Update ----------
tau = 0.005
def update_target():
    for var, var_target in zip(Q_origin_net.parameters(), Q_target_net.parameters()):
        var_target.data = tau * var.data + (1.0 - tau) * var_target.data
    for var, var_target in zip(policy_net.parameters(), policy_net_target.parameters()):
        var_target.data = tau * var.data + (1.0 - tau) * var_target.data

# ---------- Action Selection with Noise ----------
def pick_action_with_noise(states):
    """Returns a noisy action from the policy network, clipped to valid range."""
    with torch.no_grad():
        actions = policy_net(states.unsqueeze(0))  # raw action from the network (tensor shape [1,2])
        actions = actions.detach().cpu().numpy()
        if np.isnan(actions).any():
            raise ValueError("NaN obtained from policy network")
        noise = ou_action_noise()
        noisy_actions = actions + noise
        if np.isnan(noisy_actions).any():
            raise ValueError("NaN obtained from action noise")
        # Clip actions: Steering in [-0.4, 0.4], Velocity in [0.1, 5]
        noisy_actions[0][0] = np.clip(noisy_actions[0][0], -0.4, 0.4)
        noisy_actions[0][1] = np.clip(noisy_actions[0][1], 0.1, 5)
        return noisy_actions.reshape(1, 2)
def construct_full_planning_state(obs_dict, processor):

    pose = [obs_dict['poses_x'][0], obs_dict['poses_y'][0], obs_dict['poses_theta'][0]]
    velocity = [obs_dict['linear_vels_x'][0], obs_dict['linear_vels_y'][0]]
    
    # Handle NaNs for robustness
    pose = [0.0 if not np.isfinite(v) else v for v in pose]
    velocity = [0.0 if not np.isfinite(v) else v for v in velocity]

    state = np.concatenate([processor.downsample_and_normalize_lidar(obs_dict['scans'][0]), pose, velocity])
    return torch.tensor(state, dtype=torch.float32).to(device)
# -------- Reward Function ----------
# -------- Custom Reward Function ----------
def compute_reward(obs, prev_obs, action, step_reward, done, processor):
    """Calculate reward based on driving performance and lap completion"""
    # Extract key values
    x = obs['poses_x'][0]
    y = obs['poses_y'][0]
    theta = obs['poses_theta'][0]
    v_x = obs['linear_vels_x'][0]
    v_y = obs['linear_vels_y'][0]
    ang_vel_z = obs['ang_vels_z'][0]
    steering = action[0, 0]
    throttle = action[0, 1]
    min_scan = np.min(obs['scans'][0])
    collision = obs['collisions'][0]

    # === Speed Reward ===
    speed = np.hypot(v_x, v_y)
    speed_reward = 1.0 * speed
    if speed > 4.0:
        speed_reward += 1.0 * (speed - 4.0)
    if speed > 6.0:
        speed_reward += 1.0 * (speed - 6.0)

    # === Alignment Reward ===
    alignment_reward = 0.0
    if processor.centerline_waypoints:
        alignment = processor.calculate_progress_reward(pos=(x, y), theta=theta)
        alignment_reward = 1.5 * alignment

    # === Waypoint Visit Reward ===
    waypoint_reward_sparse = 5.0 if processor.waypoint_reached else 0.0

    # === Continuous Distance Reward ===
    dist = max(0.1, processor.closest_waypoint_dist)
    waypoint_reward_dense = 1.0 / dist
    waypoint_reward_dense = np.clip(waypoint_reward_dense, 0, 10.0)

    # === Lap Progress Position Reward ===
    progress_position_reward = 10.0 * processor.lap_progress

    # === Wall and Collision Penalties ===
    SAFE_DISTANCE = 0.2
    wall_penalty = 100.0 * (SAFE_DISTANCE - min_scan) / SAFE_DISTANCE if min_scan < SAFE_DISTANCE else 0.0
    collision_penalty = 200.0 if collision == 1 else 0.0

    # === Stability Penalties ===
    steering_penalty = 0.5 * abs(steering)
    spin_penalty = 0.5 * abs(ang_vel_z)
    step_penalty = 0.1

    # === Lap Completion Rewards ===
    lap_reward = 0.0
    if done and collision == 0:
        lap_reward += 100.0
    if processor.lap_completed:
        lap_reward += 100.0
        if 50.0 < processor.current_lap_time < processor.best_lap_time:
            lap_reward += 20.0

    # === Total Reward ===
    reward = (
        speed_reward +
        alignment_reward +
        waypoint_reward_sparse +
        waypoint_reward_dense +
        progress_position_reward +
        lap_reward -
        wall_penalty -
        collision_penalty -
        steering_penalty -
        spin_penalty -
        step_penalty
    )

    reward_components = {
        'speed': speed_reward,
        'alignment': alignment_reward,
        'waypoint': waypoint_reward_sparse + waypoint_reward_dense,
        'position': progress_position_reward,
        'lap': lap_reward,
        'wall': -wall_penalty,
        'collision': -collision_penalty,
        'steering': -steering_penalty,
        'spin': -spin_penalty,
        'step': -step_penalty,
        'total': reward,
        'lap_time': processor.current_lap_time
    }

    return reward, reward_components

# ---------- Main Training Loop ----------
if __name__ == "__main__":
    reward_list = []
    actor_loss_list = []
    critic_loss_list = []
    lap_list = []
    batch_size = 100
    episodes = 3500
    t = 0
    
    processor = ObservationProcessor(270,'maps/Austin_centerline.csv')

    for episode in range(episodes):
        render_flag = (episode+1)%50 == 0
        render_mode = 'human' if render_flag else None
        env = gym.make('f110_gym:f110-v0', map=map_path, map_ext='.png', num_agents=1,render_mode=render_mode)
        # Fixed starting pose
        x = 0
        y = np.random.uniform(-0.5, 0.5)  # Randomize y position within a small range
        theta = -0.61
        poses = np.array([[x, y, theta]])
        obs, _, done, _ = env.reset(poses)
        
        # Use processor to create a state vector; this returns a tensor of shape [1, 275].
        states = processor.process_obs(obs)
        ep_reward = 0
        laps_before = processor.lap_count

        while not done:
            actions = pick_action_with_noise(states)
            new_obs, step_reward, done, _ = env.step(actions)
            new_states = processor.process_obs(new_obs)
            
            # Compute reward with new observation, action, and processor
            reward,_ = compute_reward(new_obs, obs, actions, step_reward, done, processor)
            
            # Insert the transition into the replay buffer
            buffer.insert([states, actions, reward, new_states, done])
            states = new_states

            if render_flag:
                env.render(mode=render_mode)
            
            if buffer.length() > 10 * batch_size:
                # Sample a batch from the replay buffer and perform optimization
                obs_b, act_b, rew_b, new_obs_b, done_b = buffer.sample(batch_size)
                critic_loss, actor_loss = optimize(obs_b, act_b, rew_b, new_obs_b, done_b)
                update_target()
                critic_loss_list.append(critic_loss)
                actor_loss_list.append(actor_loss)

            #print(f"Reward:{reward:.5f}", end="\r")
            
            ep_reward += reward
            t += 1
            obs = new_obs
            # If desired, enable rendering:
            # env.render()
        laps_after = processor.lap_count
        laps_this_episode = laps_after - laps_before
        lap_list.append(laps_this_episode)
        
        print(f"Episode {episode+1}/{episodes} | Steps: {t} | Reward: {ep_reward:.2f} | Laps: {laps_this_episode}", end="\r")
        reward_list.append(ep_reward)
        env.close()

    # ---------- Save the Model and Logs ----------
    save_path = "ddpg_logs/actor_final.pth"
    torch.save(policy_net.state_dict(), save_path)
    import pandas as pd
    pd.DataFrame({'reward': reward_list}).to_csv("ddpg_logs/reward_list.csv", index=False)
    pd.DataFrame({'actor_loss': actor_loss_list}).to_csv("ddpg_logs/actor_loss_list.csv", index=False)
    pd.DataFrame({'critic_loss': critic_loss_list}).to_csv("ddpg_logs/critic_loss_list.csv", index=False)

    print("\nAll training logs saved to 'ddpg_logs/'")

    # ---------- Plot the Results ----------
    fig, axs = plt.subplots(3, 1, figsize=(10, 12), sharex=True)

    plt.figure(figsize=(8, 4))
    plt.plot(range(len(reward_list)), reward_list)
    plt.title("Training Progress - DDPG")
    plt.xlabel("Episode")
    plt.ylabel("Total Reward")
    plt.grid(True)
    plt.tight_layout()
    plt.show()

    plt.figure(figsize=(8, 4))
    plt.plot(range(len(critic_loss_list)), critic_loss_list)
    plt.title("Critic Loss over Time")
    plt.xlabel("Training Step")
    plt.ylabel("Critic Loss")
    plt.grid(True)
    plt.tight_layout()
    plt.show()

    plt.figure(figsize=(8, 4))
    plt.plot(range(len(actor_loss_list)), actor_loss_list)
    plt.title("Actor Loss over Time")
    plt.xlabel("Training Step")
    plt.ylabel("Actor Loss")
    plt.grid(True)
    plt.tight_layout()
    plt.show()
