import numpy as np
import torch
import os
import gym
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
import os, shutil, yaml, csv
from argparse import Namespace
from numba import njit
import numpy as np
import cProfile
import pstats
import io
from pstats import SortKey

class Utils:
    @staticmethod
    def flatten_obs(obs,for_buffer=False):
        scans = obs['scans'][0]

        obs['poses_x'][0] = 0.0 if not np.isfinite(obs['poses_x'][0]) else np.clip(obs['poses_x'][0], -100.0, 100.0)
        obs['poses_y'][0] = 0.0 if not np.isfinite(obs['poses_y'][0]) else np.clip(obs['poses_y'][0], -100.0, 100.0)
        obs['poses_theta'][0] = 0.0 if not np.isfinite(obs['poses_theta'][0]) else np.clip(obs['poses_theta'][0], -2*np.pi, 2*np.pi)
        obs['linear_vels_x'][0] = 0.0 if not np.isfinite(obs['linear_vels_x'][0]) else np.clip(obs['linear_vels_x'][0], -20.0, 20.0)
        obs['linear_vels_y'][0] = 0.0 if not np.isfinite(obs['linear_vels_y'][0]) else np.clip(obs['linear_vels_y'][0], -5.0, 5.0)
        obs['ang_vels_z'][0] = 0.0 if not np.isfinite(obs['ang_vels_z'][0]) else np.clip(obs['ang_vels_z'][0], -10.0, 10.0)

        # For binary/integer-like values
        obs['collisions'][0] = 0 if not np.isfinite(obs['collisions'][0]) else int(obs['collisions'][0])
        obs['lap_counts'][0] = 0 if not np.isfinite(obs['lap_counts'][0]) else int(obs['lap_counts'][0])

        features = np.array([
        obs['poses_x'][0],
        obs['poses_y'][0],
        obs['poses_theta'][0],
        obs['linear_vels_x'][0],
        obs['linear_vels_y'][0],
        obs['ang_vels_z'][0],
        obs['collisions'][0],
        obs['lap_counts'][0]
        ])
        
        features = np.nan_to_num(features, nan=0.0)

        state_flatten = np.concatenate([scans, features])
        
        if np.any(np.isnan(state_flatten)):
            np.set_printoptions(threshold=np.inf)
            raise ValueError("NaN found in final flattened obs")
            

        if for_buffer:
            return state_flatten
        else:
            return torch.tensor(state_flatten, dtype=torch.float32).unsqueeze(0).to(device)
    @staticmethod
    def safe_extract(value, default=0.0, clip=None):
        val = value if np.isfinite(value) else default
        return np.clip(val, *clip) if clip else val
    @staticmethod
    def downsample_lidar(scan, num_beams=20):
        """Uniformly downsample LiDAR scan to `num_beams`."""
        indices = np.linspace(0, len(scan) - 1, num=num_beams, dtype=int)
        return scan[indices]
    @staticmethod
    def normalize_lidar(scan):
        """Normalize LiDAR readings to [-1, 1] based on 10m max range."""
        scan = np.clip(scan, 0.0, 10.0)
        return 2.0 * scan / 10.0 - 1.0
    
    @staticmethod
    def construct_state(obs,new_obs):
        """
        Build the state vector as [prev_lidar, curr_lidar, velocity].
        Each scan is assumed to be already downsampled and normalized.
        """
        prev_scan = obs['scans'][0]
        curr_scan = new_obs['scans'][0]
        if np.isnan(prev_scan).any() or np.isnan(curr_scan).any():
            raise ValueError("NaN found in LiDAR scan")
        if len(prev_scan) < 20 or len(curr_scan) < 20:
            raise ValueError("Insufficient LiDAR beams for downsampling")

        velocity = new_obs['linear_vels_x'][0]
        if not np.isfinite(velocity):
            velocity = 0.0

        down_prev_scan = Utils.downsample_lidar(prev_scan)
        down_curr_scan = Utils.downsample_lidar(curr_scan)

        norm_prev_scan = Utils.normalize_lidar(down_prev_scan)
        norm_curr_scan = Utils.normalize_lidar(down_curr_scan)

        velocity = np.array([velocity])  # shape (1,)
        state = np.concatenate([norm_prev_scan, norm_curr_scan, velocity])  # shape (41,)
        
        return state.astype(np.float32)
    

    @staticmethod
    def init_file_struct(path):
        if os.path.exists(path):
            try:
                os.rmdir(path)
            except:
                shutil.rmtree(path)
        os.mkdir(path)

    @staticmethod
    def soft_update(net, net_target, tau):
        for param_target, param in zip(net_target.parameters(), net.parameters()):
            param_target.data.copy_(param_target.data * (1.0 - tau) + param.data * tau)
        
    @staticmethod    
    def load_conf(fname):
        full_path =  "experiments/" + fname + '.yaml'
        with open(full_path) as file:
            conf_dict = yaml.load(file, Loader=yaml.FullLoader)

        conf = Namespace(**conf_dict)

        return conf 
    @staticmethod
    def load_run_dict(full_path):
        with open(full_path) as file:
            run_dict = yaml.load(file, Loader=yaml.FullLoader)

        run_dict = Namespace(**run_dict)

        return run_dict 

    @staticmethod
    def generate_test_name(file_name):
        n = 1
        while os.path.exists(f"Data/{file_name}_{n}"):
            n += 1
        os.mkdir(f"Data/{file_name}_{n}")
        return file_name + f"_{n}"
    @staticmethod
    def latest_test_name(file_name):
        n = 1
        while os.path.exists(f"Data/{file_name}_{n}"):
            n += 1
        return file_name + f"_{n-1}"
    @staticmethod
    def setup_run_list(experiment_file, new_run=True):
        full_path =  "experiments/" + experiment_file + '.yaml'
        with open(full_path) as file:
            experiment_dict = yaml.load(file, Loader=yaml.FullLoader)
            
        set_n = experiment_dict['set_n']
        test_name = experiment_file + f"_{set_n}"
        if not os.path.exists(f"Data/{test_name}"):
            os.mkdir(f"Data/{test_name}")

        run_list = []
        for rep in range(experiment_dict['start_n'], experiment_dict['n_repeats']):
            for run in experiment_dict['runs']:
                # base is to copy everything from the original
                for key in experiment_dict.keys():
                    if key not in run.keys() and key != "runs":
                        run[key] = experiment_dict[key]

                # only have to add what isn't already there
                set_n = run['set_n']
                max_speed = run['max_speed']
                run["n"] = rep
                run['run_name'] = f"{run['planner_type']}_{run['algorithm']}_{run['state_vector']}_{run['map_name']}_{run['id_name']}_{max_speed}_{set_n}_{rep}"
                run['path'] = f"{test_name}/"

                run_list.append(Namespace(**run))

        return run_list

    @staticmethod
    @njit(cache=True)
    def calculate_speed(delta, f_s=0.8, max_v=7):
        b = 0.523
        g = 9.81
        l_d = 0.329

        if abs(delta) < 0.03:
            return max_v
        if abs(delta) > 0.4:
            return 0

        V = f_s * np.sqrt(b*g*l_d/np.tan(abs(delta)))

        V = min(V, max_v)

        return V
    @staticmethod
    def save_csv_array(data, filename):
        with open(filename, 'w') as file:
            writer = csv.writer(file)
            writer.writerows(data)
    @staticmethod
    def moving_average(data, period):
        return np.convolve(data, np.ones(period), 'same') / period
    @staticmethod
    def true_moving_average(data, period):
        if len(data) < period:
            return np.zeros_like(data)
        ret = np.convolve(data, np.ones(period), 'same') / period
        # t_end = np.convolve(data, np.ones(period), 'valid') / (period)
        # t_end = t_end[-1] # last valid value
        for i in range(period): # start
            t = np.convolve(data, np.ones(i+2), 'valid') / (i+2)
            ret[i] = t[0]
        for i in range(period):
            length = int(round((i + period)/2))
            t = np.convolve(data, np.ones(length), 'valid') / length
            ret[-i-1] = t[-1]
        return ret
    @staticmethod
    def save_run_config(run_dict, path):
        path = path +  f"/TrainingRunDict_record.yaml"
        with open(path, 'w') as file:
            yaml.dump(run_dict, file)


    @staticmethod            
    def profile_and_save(function):
        with cProfile.Profile(builtins=False) as pr:
            function()
            
            with open("Data/Profiling/main.prof", "w") as f:
                ps = pstats.Stats(pr, stream=f)
                ps.strip_dirs()
                ps.sort_stats('cumtime')
                ps.print_stats()
                
            with open("Data/Profiling/main_total.prof", "w") as f:
                ps = pstats.Stats(pr, stream=f)
                ps.strip_dirs()
                ps.sort_stats('tottime')
                ps.print_stats()