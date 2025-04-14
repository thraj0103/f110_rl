
import gym
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Normal
import matplotlib.pyplot as plt
from tqdm import tqdm
# import time
import yaml
from argparse import Namespace
# from collections import deque
import random

# Set device (prefer GPU if available)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")


class ActorNet(nn.Module):
    def __init__(self, obs_dim, act_dim, hidden_dim=512):
        super().__init__()
        self.fc1 = nn.Linear(obs_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.fc3 = nn.Linear(hidden_dim, hidden_dim)
        print(f"ActorNet: obs_dim={obs_dim}, act_dim={act_dim}, hidden_dim={hidden_dim}")

        self.mu_head = nn.Linear(hidden_dim, act_dim)
        self.log_std = nn.Parameter(torch.tensor([-2.5, 0.22], dtype=torch.float32))

        # Action bounds (steering, velocity)
        self.action_bounds = {
            'low': torch.tensor([-0.4, 0.0], device=device),
            'high': torch.tensor([0.4, 7.0], device=device)
        }

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = F.relu(self.fc3(x))

        # Mean output
        mu = self.mu_head(x)
        mu = torch.tanh(mu)  # constrain to [-1, 1]
        mu = (mu + 1) / 2 * (self.action_bounds['high'] - self.action_bounds['low']) + self.action_bounds['low']

        # Use broadcasted, fixed log_std per dim
        std = torch.exp(self.log_std).expand_as(mu)

        return mu, std

class CriticNet(nn.Module):
    def __init__(self, obs_dim, hidden_dim=256):
        super().__init__()
        self.fc1 = nn.Linear(obs_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.fc3 = nn.Linear(hidden_dim, hidden_dim)
        self.out = nn.Linear(hidden_dim, 1)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = F.relu(self.fc3(x))
        return self.out(x)

class RunningMeanStd:
    def __init__(self, shape=(), epsilon=1e-4):
        self.mean = np.zeros(shape, dtype=np.float32)
        self.var = np.ones(shape, dtype=np.float32)
        self.count = epsilon

    def update(self, x):
        batch_mean = np.mean(x, axis=0)
        batch_var = np.var(x, axis=0)
        batch_count = x.shape[0]
        
        delta = batch_mean - self.mean
        total_count = self.count + batch_count
        
        self.mean = self.mean + delta * batch_count / total_count
        m_a = self.var * self.count
        m_b = batch_var * batch_count
        M2 = m_a + m_b + np.square(delta) * self.count * batch_count / total_count
        self.var = M2 / total_count
        self.count = total_count


# Observation Processor using only centerline waypoints
class ObservationProcessor:
    def __init__(self, num_beams=270, centerline_file=None):
        self.num_beams = num_beams
        self.obs_rms = RunningMeanStd(shape=(num_beams + 5,))
        self.is_training = True

        self.centerline_waypoints = []
        self.waypoints_finalized = False
        self.load_centerline(centerline_file)

        self.last_waypoint_idx = 0
        self.lap_progress = 0.0
        self.lap_completed = False
        self.previous_progress = 0.0

        self.current_lap_steps = 0
        self.best_lap_steps = float('inf')
        self.lap_count = 0
        self.current_lap_time = 0.0
        self.best_lap_time = float('inf')

        self.start_line_pos = None
        self.last_reached_waypoint = -1
        self.waypoint_reached = False
        self.closest_waypoint_dist = float('inf')
        self.waypoint_direction = None
        self.total_waypoints = 0
        self.waypoints_visited = set()

    def load_centerline(self, centerline_file):
        if centerline_file is None:
            return
        try:
            import csv
            with open(centerline_file, 'r') as f:
                reader = csv.reader(f)
                next(reader)
                for row in reader:
                    x = float(row[0])
                    y = float(row[1])
                    self.centerline_waypoints.append((x, y))
                self.centerline_waypoints = self.centerline_waypoints[::5] 
                self.total_waypoints = len(self.centerline_waypoints)
                self._calculate_waypoint_directions()
                if len(self.centerline_waypoints) > 0:
                    self.waypoints_finalized = True
                print(f"Loaded {self.total_waypoints} waypoints from {centerline_file}")
        except Exception as e:
            print(f"Error loading centerline file: {e}")
            self.centerline_waypoints = []

    def _calculate_waypoint_directions(self):
        if len(self.centerline_waypoints) < 2:
            return
        self.waypoint_direction = []
        for i in range(len(self.centerline_waypoints)):
            next_idx = (i + 1) % len(self.centerline_waypoints)
            curr = self.centerline_waypoints[i]
            next_wp = self.centerline_waypoints[next_idx]
            dir_x = next_wp[0] - curr[0]
            dir_y = next_wp[1] - curr[1]
            length = (dir_x**2 + dir_y**2)**0.5
            if length > 0:
                dir_x /= length
                dir_y /= length
            self.waypoint_direction.append((dir_x, dir_y))

    def process_obs(self, obs, update_stats=True):
        scan = obs['scans'][0][::4]
        scan = np.clip(scan, 0.0, 30.0)
        scan[~np.isfinite(scan)] = 30.0
        x = np.array([obs['poses_x'][0]])
        y = np.array([obs['poses_y'][0]])
        theta = np.array([obs['poses_theta'][0]])
        v_x = np.array([obs['linear_vels_x'][0]])
        v_y = np.array([obs['linear_vels_y'][0]])
        pos = (x[0], y[0])

        if self.start_line_pos is None:
            self.start_line_pos = pos

        self.waypoint_reached = False
        if len(self.centerline_waypoints) > 0:
            self._update_waypoint_progress(pos, theta[0])

        flat_obs = np.concatenate([scan, x, y, theta, v_x, v_y])
        if update_stats and self.is_training:
            self.obs_rms.update(flat_obs.reshape(1, -1))

        obs_mean = self.obs_rms.mean
        obs_var = self.obs_rms.var
        normalized_obs = (flat_obs - obs_mean) / np.sqrt(obs_var + 1e-8)
        return torch.tensor(normalized_obs, dtype=torch.float32).to(device)

    def _find_closest_waypoint(self, pos):
        closest_idx = -1
        closest_dist = float('inf')
        for i, waypoint in enumerate(self.centerline_waypoints):
            dist = np.sqrt((pos[0] - waypoint[0])**2 + (pos[1] - waypoint[1])**2)
            if dist < closest_dist:
                closest_dist = dist
                closest_idx = i
        return closest_idx, closest_dist

    def _update_waypoint_progress(self, pos, theta):
        closest_idx, closest_dist = self._find_closest_waypoint(pos)
        self.closest_waypoint_dist = closest_dist

        if closest_dist < 0.5 and closest_idx != self.last_reached_waypoint:
            self.waypoint_reached = True
            self.waypoints_visited.add(closest_idx)
            self.last_reached_waypoint = closest_idx

        self.lap_progress = closest_idx / max(1, len(self.centerline_waypoints))
        if len(self.waypoints_visited) > len(self.centerline_waypoints) * 0.9:
            start_waypoint = 0
            dist_to_start = np.sqrt((pos[0] - self.centerline_waypoints[start_waypoint][0])**2 + 
                                    (pos[1] - self.centerline_waypoints[start_waypoint][1])**2)
            if dist_to_start < 1.0 and self.previous_progress > 0.9:
                self.lap_completed = True
                self.lap_count += 1
                if self.current_lap_time < self.best_lap_time and self.current_lap_time > 10.0:
                    self.best_lap_time = self.current_lap_time
                self.waypoints_visited = set()
                self.current_lap_time = 0.0
            else:
                self.lap_completed = False
        else:
            self.lap_completed = False

        self.previous_progress = self.lap_progress
        self.current_lap_steps += 1

    def calculate_progress_reward(self, pos, theta):
        if len(self.centerline_waypoints) < 2:
            return 0.0
        closest_idx, _ = self._find_closest_waypoint(pos)
        next_idx = (closest_idx + 1) % len(self.centerline_waypoints)
        curr_wp = self.centerline_waypoints[closest_idx]
        next_wp = self.centerline_waypoints[next_idx]
        vec_to_next = (next_wp[0] - pos[0], next_wp[1] - pos[1])
        car_dir = (np.cos(theta), np.sin(theta))
        dot_product = car_dir[0] * vec_to_next[0] + car_dir[1] * vec_to_next[1]
        dist_to_next = np.sqrt(vec_to_next[0]**2 + vec_to_next[1]**2)
        alignment = dot_product / dist_to_next if dist_to_next > 0 else 0
        return max(0, alignment)

def sample_action_and_logprob(actor, obs_flat, deterministic=False, training_progress=0.0):
    mu, std = actor(obs_flat)
    
    if deterministic:
        action = mu
        # We'll still compute log_prob for API consistency
        dist = Normal(mu, std)
        log_prob = dist.log_prob(action).sum(dim=-1)
        entropy = dist.entropy().sum(dim=-1)
    else:
        dist = Normal(mu, std)
        action = dist.sample()
        log_prob = dist.log_prob(action).sum(dim=-1)
        entropy = dist.entropy().sum(dim=-1)
    
    # Ensure action is within valid bounds
    action = torch.clamp(action, 
                        min=actor.action_bounds['low'],
                        max=actor.action_bounds['high'])
    
    return action, log_prob, entropy, dist

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
    dist = max(0.1, processor.closest_waypoint_dist)  # prevent divide by zero
    waypoint_reward_dense = 1.0 / dist  # gets higher as car gets closer
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

def collect_rollout(env, actor, critic, processor, max_steps=100000, render=False, deterministic=False, episode_num=0, total_episodes=5000):
    """Collect a training episode with the current policy"""
    obs_list, act_list, logprob_list, value_list, reward_list, reward_components_list = [], [], [], [], [], []
    
    # Reset environment with fixed starting position
    start_x = 0.0
    start_y = 0.0
    start_theta = -0.6524  # Fixed heading to ensure car starts in the right direction
    
    obs, step_reward, done, info = env.reset(poses=np.array([[start_x, start_y, start_theta]]))
    
    # Reset processor lap tracking for this episode
    processor.lap_completed = False
    processor.current_lap_steps = 0
    processor.current_lap_time = 0.0
    processor.waypoints_visited = set()  # Reset visited waypoints properly
    
    # Episode-specific variables
    done = False
    steps = 0
    # prev_obs = obs  # Store previous observation for reward calculation
    
    # Calculate training progress for exploration rate
    training_progress = episode_num / total_episodes
    
    while not done and steps < max_steps:
        obs_flat = processor.process_obs(obs)
        with torch.no_grad():
            value = critic(obs_flat)
            action, log_prob, _, _ = sample_action_and_logprob(actor, obs_flat, deterministic, training_progress)

        # if episode_num % 10 == 0 and steps % 25 == 0:
        #     mu, std = actor(obs_flat)
        #     print(f"\n🎯 Episode {episode_num} | Step {steps}")
        #     print(f"    ➤ mu (steer, velocity):     {mu.detach().cpu().numpy()}")
        #     print(f"    ➤ std (steer, velocity):    {std.detach().cpu().numpy()}")
        #     print(f"    ➤ action (sampled):         {action.detach().cpu().numpy()}")    
        
        obs_list.append(obs_flat.detach())
        act_list.append(action.detach())
        logprob_list.append(log_prob.detach())
        value_list.append(value.detach())
        
        action_np = action.cpu().numpy().reshape(1, -1)
        next_obs, step_reward, done, info = env.step(action_np)
        
        # Update lap time using step_reward from environment
        processor.current_lap_time += step_reward
        
        # Calculate reward
        reward, reward_components = compute_reward(next_obs, obs, action_np, step_reward, done, processor)
        
        reward_list.append(reward)
        reward_components_list.append(reward_components)
        
        # Update for next iteration - fixed to use correct observation
        # prev_obs = obs
        obs = next_obs
        
        steps += 1
        
        if render:
            env.render(mode='human_fast')
    
    return obs_list, act_list, logprob_list, value_list, reward_list, reward_components_list, steps

def compute_returns_and_advantages(rewards, values, gamma=0.99, lam=0.95):
    """Calculate GAE advantages and returns"""
    rewards = torch.tensor(rewards, dtype=torch.float32).to(device)
    values = torch.cat(values).squeeze().to(device)
    
    returns = torch.zeros_like(rewards)
    advantages = torch.zeros_like(rewards)
    
    next_value = 0
    next_advantage = 0
    
    for t in reversed(range(len(rewards))):
        delta = rewards[t] + gamma * next_value - values[t]
        advantages[t] = delta + gamma * lam * next_advantage
        returns[t] = advantages[t] + values[t]
        next_value = values[t]
        next_advantage = advantages[t]
    
    # Normalize advantages for more stable learning
    advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)
    
    return returns.detach(), advantages.detach()

def ppo_update(actor, critic, optimizer, obs_batch, act_batch, old_logprobs, returns, advantages,
              clip_eps=0.1, vf_coeff=0.5, ent_coeff=0.005, max_grad_norm=0.5):
    """Update actor and critic networks using PPO algorithm"""
    obs_batch = torch.stack(obs_batch).to(device)
    act_batch = torch.stack(act_batch).to(device)
    old_logprobs = torch.stack(old_logprobs).to(device)
    returns = returns.to(device)
    advantages = advantages.to(device)
    
    # Calculate batch size based on rollout length
    batch_size = len(obs_batch)
    minibatch_size = min(64, batch_size)
    num_updates = max(10, batch_size // minibatch_size)
    
    total_loss = 0
    total_policy_loss = 0
    total_value_loss = 0
    total_entropy = 0
    
    for _ in range(num_updates):
        # Randomly sample minibatch
        idx = torch.randperm(batch_size)[:minibatch_size]
        
        mb_obs = obs_batch[idx]
        mb_acts = act_batch[idx]
        mb_old_logprobs = old_logprobs[idx]
        mb_returns = returns[idx]
        mb_advantages = advantages[idx]
        
        # Get current policy distribution
        mu, std = actor(mb_obs)
        dist = Normal(mu, std)
        new_logprobs = dist.log_prob(mb_acts).sum(dim=-1)
        entropy = dist.entropy().sum(dim=-1)
        
        # PPO policy loss
        ratio = torch.exp(new_logprobs - mb_old_logprobs)
        surr1 = ratio * mb_advantages
        surr2 = torch.clamp(ratio, 1 - clip_eps, 1 + clip_eps) * mb_advantages
        policy_loss = -torch.min(surr1, surr2).mean()
        
        # Value function loss
        value_pred = critic(mb_obs).squeeze()
        value_loss = F.mse_loss(value_pred, mb_returns)
        
        # Total loss
        loss = policy_loss + vf_coeff * value_loss - ent_coeff * entropy.mean()
        
        # Perform update
        optimizer.zero_grad()
        loss.backward()
        
        # Gradient clipping for stability
        nn.utils.clip_grad_norm_(list(actor.parameters()) + list(critic.parameters()), max_grad_norm)
        
        optimizer.step()
        
        # Accumulate loss values
        total_loss += loss.item()
        total_policy_loss += policy_loss.item()
        total_value_loss += value_loss.item()
        total_entropy += entropy.mean().item()
    
    # Return average losses
    n = num_updates
    return total_loss/n, total_policy_loss/n, total_value_loss/n, total_entropy/n

def train_ppo(env_config='Austin_map.yaml', num_episodes=10000, save_interval=25, render_interval=100):
    """Train the PPO agent on the F1TENTH environment"""
    # Load environment configuration
    with open(env_config) as file:
        conf_dict = yaml.load(file, Loader=yaml.FullLoader)
    
    if conf_dict is None:
        raise ValueError("⚠️ YAML file is empty or malformed!")
    
    conf = Namespace(**conf_dict)
    
    # Create environment
    env = gym.make('f110_gym:f110-v0', map=conf.map_path, map_ext=conf.map_ext, num_agents=1)
    
    # Initialize environment
    init_poses = np.array([[0.0, 0.0, 0.0]])
    obs, _, _, _ = env.reset(poses=init_poses)
    
    # Create observation processor
    centerline_file = 'Austin_centerline.csv'
    processor = ObservationProcessor(centerline_file=centerline_file)
    
    # Process observation to get dimensions
    flat_obs = processor.process_obs(obs)
    obs_dim = flat_obs.shape[0]
    act_dim = 2  # Steering and acceleration
    
    # Create actor and critic networks
    actor = ActorNet(obs_dim, act_dim, hidden_dim=512).to(device)
    critic = CriticNet(obs_dim, hidden_dim=512).to(device)
    
    # Create optimizer and LR scheduler
    optimizer = torch.optim.Adam(list(actor.parameters()) + list(critic.parameters()), lr=1e-4)
    lr_scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lambda epoch: max(1.0 - epoch / num_episodes, 0.1))
    
    # Metrics
    reward_records, episode_length_records, lap_records = [], [], []
    best_reward, best_lap_time = -np.inf, float('inf')

    print("✅ Starting PPO training...")
    for episode in tqdm(range(num_episodes), desc="Training PPO"):
        render_flag = (episode % render_interval == 0)

        rollout = collect_rollout(env, actor, critic, processor,
                                  render=render_flag, episode_num=episode, total_episodes=num_episodes)
        obs_list, act_list, logprob_list, value_list, reward_list, reward_components_list, steps = rollout

        returns, advantages = compute_returns_and_advantages(reward_list, value_list)

        loss, p_loss, v_loss, entropy = ppo_update(
            actor, critic, optimizer, obs_list, act_list, logprob_list, returns, advantages
        )
        lr_scheduler.step()

        total_reward = sum(reward_list)
        reward_records.append(total_reward)
        episode_length_records.append(steps)
        lap_records.append(processor.lap_count)

        if processor.best_lap_time < best_lap_time and processor.best_lap_time < float('inf'):
            best_lap_time = processor.best_lap_time
            torch.save({
                'actor': actor.state_dict(),
                'critic': critic.state_dict(),
                'obs_rms_mean': processor.obs_rms.mean,
                'obs_rms_var': processor.obs_rms.var,
                'best_lap_time': best_lap_time,
            }, "best_laptime_model.pt")
            print(f"\n🏆 New best lap time: {best_lap_time:.2f} seconds! Model saved.")

        if total_reward > best_reward:
            best_reward = total_reward
            torch.save({
                'actor': actor.state_dict(),
                'critic': critic.state_dict(),
                'obs_rms_mean': processor.obs_rms.mean,
                'obs_rms_var': processor.obs_rms.var,
                'best_reward': best_reward,
            }, "best_reward_model.pt")

        if episode % 250 == 0:
            avg_reward = np.mean(reward_records[-10:]) if len(reward_records) >= 10 else np.mean(reward_records)
            avg_steps = np.mean(episode_length_records[-10:]) if len(episode_length_records) >= 10 else np.mean(episode_length_records)

            print(f"\nEpisode {episode}: Reward={total_reward:.2f}, Steps={steps}, Laps={processor.lap_count}")
            print(f"Avg Reward (10): {avg_reward:.2f}, Avg Steps: {avg_steps:.1f}")
            if processor.best_lap_time < float('inf'):
                print(f"Best lap time: {processor.best_lap_time:.2f} seconds")

            if reward_components_list:
                components = {k: 0 for k in reward_components_list[0].keys()}
                for comp in reward_components_list:
                    for k, v in comp.items():
                        components[k] += v
                n = len(reward_components_list)
                for k in components:
                    components[k] /= n

                print("Avg Reward Components:")
                for k in ['speed', 'waypoint', 'position', 'lap', 'steering', 'collision', 'wall']:
                    print(f"  {k.capitalize()}: {components[k]:.2f}")

    # Plot metrics
    plt.figure(figsize=(15, 10))
    plt.subplot(3, 2, 1)
    plt.plot(reward_records)
    plt.xlabel("Episode"); plt.ylabel("Total Reward"); plt.title("Reward Curve"); plt.grid()

    plt.subplot(3, 2, 2)
    plt.plot(episode_length_records)
    plt.xlabel("Episode"); plt.ylabel("Steps"); plt.title("Episode Length"); plt.grid()

    plt.subplot(3, 2, 3)
    plt.plot([np.mean(reward_records[max(0, i-10):i+1]) for i in range(len(reward_records))])
    plt.xlabel("Episode"); plt.ylabel("Smoothed Reward"); plt.title("Smoothed Reward (10)"); plt.grid()

    plt.subplot(3, 2, 4)
    plt.plot(lap_records)
    plt.xlabel("Episode"); plt.ylabel("Laps Completed"); plt.title("Lap Count"); plt.grid()

    plt.subplot(3, 2, 5)
    plt.plot(range(len(reward_records)-min(50, len(reward_records)), len(reward_records)), reward_records[-min(50, len(reward_records)):])
    plt.xlabel("Episode"); plt.ylabel("Reward"); plt.title("Last Episodes Reward"); plt.grid()

    if max(lap_records) > 0:
        lap_diff = [lap_records[i] - lap_records[i-1] if i > 0 else lap_records[i] for i in range(len(lap_records))]
        plt.subplot(3, 2, 6)
        plt.plot([np.mean(lap_diff[max(0, i-10):i+1]) for i in range(len(lap_diff))])
        plt.xlabel("Episode"); plt.ylabel("Lap Completion Rate"); plt.title("Smoothed Lap Completion Rate"); plt.grid()

    plt.tight_layout()
    plt.savefig("training_curves.png")
    plt.show()

    print("\n===== Training Summary =====")
    print(f"Total Episodes: {num_episodes}")
    print(f"Best Reward: {best_reward:.2f}")
    if best_lap_time < float('inf'):
        print(f"Best Lap Time: {best_lap_time:.2f} seconds")
    print(f"Total Laps Completed: {processor.lap_count}")

    return actor, critic, processor

def render_policy(env, actor, processor, max_steps=50000):
    """Render the trained policy and evaluate performance"""
    # Switch processor to evaluation mode
    processor.is_training = False
    
    # Reset environment
    obs, step_reward, done, info = env.reset(poses=np.array([[0.0, 0.0, 0.0]]))
    
    # Reset lap tracking
    processor.lap_completed = False
    processor.current_lap_time = 0.0
    
    total_reward = 0
    step = 0
    lap_times = []
    
    print("\n🏁 Starting evaluation run...")
    
    while not done and step < max_steps:
        # Process observation
        obs_flat = processor.process_obs(obs, update_stats=False)
        
        # Get action from policy (deterministic)
        with torch.no_grad():
            action, _, _, _ = sample_action_and_logprob(actor, obs_flat, deterministic=True)
        
        # Take step in environment
        action_np = action.cpu().numpy().reshape(1, -1)
        next_obs, step_reward, done, info = env.step(action_np)
        
        # Update lap time
        processor.current_lap_time += step_reward
        
        # Calculate reward (for display only)
        reward, components = compute_reward(next_obs, obs, action_np, step_reward, done, processor)
        total_reward += reward
        
        # Check if lap was completed this step
        if processor.lap_completed:
            lap_times.append(processor.current_lap_time)
            print(f"🏎️ Lap {len(lap_times)} completed in {processor.current_lap_time:.2f} seconds")
            processor.current_lap_time = 0.0
        
        # Update for next iteration
        obs = next_obs
        step += 1
        
        # Print occasional status
        if step % 100 == 0:
            speed = np.hypot(obs['linear_vels_x'][0], obs['linear_vels_y'][0])
            print(f"Step {step}: Speed = {speed:.2f} m/s, Progress = {processor.lap_progress:.2f}")
        
        # Render environment
        env.render()
        # time.sleep(0.01)  # Slow down rendering for better visualization
    
    # Summary
    print(f"\n✅ Evaluation completed after {step} steps with total reward {total_reward:.2f}")
    print(f"Completed {processor.lap_count} laps")
    
    if lap_times:
        print("\nLap times:")
        for i, time in enumerate(lap_times):
            print(f"  Lap {i+1}: {time:.2f} seconds")
        print(f"  Best lap: {min(lap_times):.2f} seconds")
    
    # Switch processor back to training mode
    processor.is_training = True
    
    return lap_times

def load_model(checkpoint_path, obs_dim, act_dim):
    """Load a trained model from checkpoint file"""
    checkpoint = torch.load(checkpoint_path, weights_only=False)

    actor = ActorNet(obs_dim, act_dim).to(device)
    critic = CriticNet(obs_dim).to(device)

    actor.load_state_dict(checkpoint['actor'])
    critic.load_state_dict(checkpoint['critic'])

    # Create processor with saved normalization parameters
    processor = ObservationProcessor()
    processor.obs_rms.mean = checkpoint['obs_rms_mean']
    processor.obs_rms.var = checkpoint['obs_rms_var']

    print(f"Loaded model from {checkpoint_path}")
    print(f"Centerline waypoints loaded: {len(processor.centerline_waypoints)}")

    if 'best_lap_time' in checkpoint:
        print(f"Best lap time in checkpoint: {checkpoint['best_lap_time']:.2f} seconds")
    if 'best_reward' in checkpoint:
        print(f"Best reward in checkpoint: {checkpoint['best_reward']:.2f}")

    return actor, critic, processor

if __name__ == "__main__":
    # Set random seeds for reproducibility
    torch.manual_seed(42)
    np.random.seed(42)
    random.seed(42)
    
    # Centerline file path
    centerline_file = 'Austin_centerline.csv'  # Path to the centerline file
    
    # Train or load model
    TRAIN_NEW_MODEL = True  # Set to False to load a saved model
    
    if TRAIN_NEW_MODEL:
        print("Training new model...")
        # Create environment to get observation dimensions
        with open('Austin_map.yaml') as file:
            conf_dict = yaml.load(file, Loader=yaml.FullLoader)
        conf = Namespace(**conf_dict)
        env = gym.make('f110_gym:f110-v0', map=conf.map_path, map_ext=conf.map_ext, num_agents=1)
        obs, _, _, _ = env.reset(poses=np.array([[0.0, 0.0, 0.0]]))
        
        # Create processor with centerline file
        processor = ObservationProcessor(centerline_file=centerline_file)
        flat_obs = processor.process_obs(obs)
        obs_dim = flat_obs.shape[0]
        act_dim = 2
        
        # Create networks with proper dimensions
        actor = ActorNet(obs_dim, act_dim, hidden_dim=512).to(device)
        critic = CriticNet(obs_dim, hidden_dim=512).to(device)
        
        # Train
        actor, critic, processor = train_ppo(num_episodes=3500)
    else:
        # Load environment to get observation dimensions
        with open('Austin_map.yaml') as file:
            conf_dict = yaml.load(file, Loader=yaml.FullLoader)
        conf = Namespace(**conf_dict)
        env = gym.make('f110_gym:f110-v0', map=conf.map_path, map_ext=conf.map_ext, num_agents=1)
        obs, _, _, _ = env.reset(poses=np.array([[0.0, 0.0, -0.6524]]))
        
        # Create processor to get observation dimensions
        temp_processor = ObservationProcessor(centerline_file=centerline_file)
        flat_obs = temp_processor.process_obs(obs)
        obs_dim = flat_obs.shape[0]
        act_dim = 2
        
        # Load model
        model_path = "best_laptime_model.pt"  # Change to desired model
        actor, critic, processor = load_model(model_path, obs_dim, act_dim)
    
    # Create environment for rendering
    with open('Austin_map.yaml') as file:
        conf_dict = yaml.load(file, Loader=yaml.FullLoader)
    conf = Namespace(**conf_dict)
    env = gym.make('f110_gym:f110-v0', map=conf.map_path, map_ext=conf.map_ext, num_agents=1)
    
    # Render and evaluate trained policy
    lap_times = render_policy(env, actor, processor)
    
    print("✅ Evaluation complete.")


