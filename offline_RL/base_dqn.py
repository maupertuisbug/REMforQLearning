import random
import torch 
import numpy as np 
import gymnasium as gym 
import argparse 
from omegaconf import OmegaConf
import wandb
from replay_buffer.fixed_replay_buffer import FixedReplayBuffer
from stable_baselines3.common.buffers import ReplayBuffer
from stable_baselines3.common.save_util import load_from_pkl
from utils.util import make_env, epsilon_schedule
import ale_py 
import torch.optim as optim 
from utils.util import make_env
import numpy as np


gym.register_envs(ale_py)

class QNetwork(torch.nn.Module):

    def __init__(self, env):

        super(QNetwork, self).__init__() 

        self.conv1 = torch.nn.Conv2d(1, 32, 8, stride=4)
        self.conv2 = torch.nn.Conv2d(32, 64, 4, stride=2)
        self.conv3 = torch.nn.Conv2d(64, 64, 3, stride=1)
        self.f1    = torch.nn.Linear(3136, 512)
        self.f2    = torch.nn.Linear(512, env.single_action_space.n)
        self.activation = torch.nn.ReLU()
        self.flatten = torch.nn.Flatten(start_dim=1)

    def forward(self, x):

        input = x/255.0 

        input = self.batch_preprocessing(input)

        output = self.conv1(input)
        output = self.activation(output)
        output = self.conv2(output)
        output = self.activation(output)
        output = self.conv3(output)
        output = self.activation(output)
        output = self.flatten(output)
        output = self.activation(output).unsqueeze(0)
        output = self.f1(output)
        output = self.activation(output)
        output = self.f2(output)
        return output
    
    def batch_preprocessing(self, input):
        if (input.ndim == 3):
            input = input.unsqueeze(1)
        else :
            input = input.unsqueeze(0).unsqueeze(1)
        
        return input
    

class BaseAgent : 

    def __init__(self, config, wandb_run):

        self.gamma = config.gamma
        self.epsilon_a = config.epsilon_a 
        self.epsilon_b = config.epsilon_b 
        self.batch_size = config.batch_size 
        self.learning_rate = config.learning_rate 
        self.seed = config.seed 
        self.buffer_size = config.buffer_size 
        self.exploration_fraction = config.exploration_fraction 
        self.replay_buffer = None 
        self.env_id = "BreakoutNoFrameskip-v4"
        self.capture_video = config.capture_video 
        self.total_timesteps = config.total_timesteps 
        self.step_start_learning = config.step_start_learning 
        self.training_frequency = config.training_frequency 
        self.num_envs = config.num_envs 
        self.tau = config.tau 
        self.epochs = config.epochs
        self.evaluation_epochs = config.evaluation_epochs
        self.sync_freq = config.target_network_frequency 
        self.k = config.k
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.wandb_run = wandb_run

    def reset(self):

        # --- Set seeds --- #
        random.seed(self.seed)
        np.random.seed(self.seed)
        torch.manual_seed(self.seed)

        # --- Setup Envs and Networks --- #
        self.envs = gym.vector.SyncVectorEnv([make_env(self.env_id, self.seed, i, self.capture_video) for i in range(self.num_envs)])
        self.q_networks = [] 
        self.target_networks = []
        self.optimizers = []
        for k in range(0, self.k):
            self.q_networks.append(QNetwork(self.envs).to(self.device))
            self.target_networks.append(QNetwork(self.envs).to(self.device))
        
        for k in range(0, self.k):
            self.optimizers.append(optim.Adam(self.q_networks[k].parameters(), lr=self.learning_rate))

        ac = self.envs.single_action_space.n 

        # --- Setup Replay Buffer and Load Offline Data --- #
        self.replay_buffer = ReplayBuffer(
            self.buffer_size, 
            self.envs.single_observation_space, 
            self.envs.single_action_space,
            self.device,
            optimize_memory_usage = True,
            handle_timeout_termination = False
        )

        file_path = "offline_data/offline_data_smaller.pkl"
        self.replay_buffer = load_from_pkl(file_path)

    
    def train_agent(self):
        raise NotImplementedError("Subclass must override Train Agent")
    

    def evaluate_agent(self):
        raise NotImplementedError("Subclass must override Evaluate Agent")
            

