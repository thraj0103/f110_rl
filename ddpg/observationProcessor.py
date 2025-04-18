import gym
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Normal
import matplotlib.pyplot as plt

# import time
import yaml
from argparse import Namespace
# from collections import deque
import random

# Set device (prefer GPU if available)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

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