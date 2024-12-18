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
    

class REMAgent : 

    def __init__(self, config):

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
        self.target_network_frequency = config.target_network_frequency 
        self.k = config.k
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

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

        for k in range(self.k):
            self.q_networks.append(QNetwork(self.envs).to(self.device))
            self.target_networks.append(QNetwork(self.envs).to(self.device))
            self.optimizers.append(optim.Adam(self.q_networks[-1].parameters(), lr=self.learning_rate))

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

        file_path = "offline_data/offline_data"
        self.replay_buffer = load_from_pkl(file_path)
        print(self.replay_buffer.pos)

    
    def train_rem(self):

        for epoch in range(self.epochs):
            total_loss = 0

            data = self.replay_buffer.sample(self.batch_size)
            probabilities = np.random.dirichlet(alpha=np.ones(self.k), size=1)[0]
            td_target = 0
            current_value = 0
            for k in range(self.k):
                    
                q_network = self.q_networks[k]
                target_network = self.target_networks[k]
                    
                target_output = q_network(data.next_observations)
                target_max, _ = target_network(data.next_observations).max(dim=2)
                td_target = td_target + data.rewards.flatten() + probabilities[k] * (self.gamma * target_max * (1 - data.dones.flatten()))

                current_value_ = q_network(data.observations).squeeze(0)
                current_value = current_value + probabilities[k] * (current_value_.gather(1, data.actions).squeeze().unsqueeze(0))
                    
            loss = torch.nn.functional.mse_loss(td_target, current_value)

            for k in range(self.k):
                self.optimizers[k].zero_grad()
                
            loss.backward()

            for k in range(self.k):
                self.optimizers[k].step()


            if epoch % self.target_network_frequency == 0:

                for k in range(self.k) : 
                    for target_network_param, q_network_param in zip(self.target_networks[k].parameters(), self.q_networks[k].parameters()):
                        target_network_param.data.copy_(
                            self.tau * q_network_param.data + (1.0 - self.tau)*target_network_param.data
                        )

    def evaluate(self, wandb_run):

        step = 0
        episode = 0
        episode_reward = []
        episode_length = []
        for epoch in range(self.evaluation_epochs):
            total_reward = 0
            obs, _ = self.envs.reset(seed=self.seed)
            ep_length = 0 
            
            while True :
                step = step + 1
                ep_length = ep_length + 1 
                epsilon = epsilon_schedule(self.epsilon_a, self.epsilon_b, self.exploration_fraction * self.total_timesteps, step)
                probabilities = np.random.dirichlet(alpha=np.ones(self.k), size=1)[0]
                if random.random() < epsilon :
                    actions = self.envs.action_space.sample()
                else :
                    q_values = 0
                    for k in range(self.k):
                        q_network = self.q_networks[k]
                        q_values_ = q_network(torch.tensor(obs, device=self.device)).squeeze(0)
                        q_values  = q_values + probabilities[k]*q_values_
                    
                    actions = torch.argmax(q_values, dim=1).cpu().numpy()
                
                next_obs, rewards, terminated, truncated, info = self.envs.step(actions)
                # --- To do -- Does new experience help? -- #
                _next_obs = next_obs.copy()
                obs = _next_obs
                total_reward = total_reward + rewards

                if terminated:
                    episode += 1
                    episode_length.append(ep_length)
                    episode_reward.append(total_reward)
                    break
            
            if epoch % 100 == 0:
                wandb_run.log({"Average Episode Reward" : np.mean(episode_reward)}, step = int(epoch/100))
                wandb_run.log({"Average Episode Length" : np.mean(episode_length)}, step = int(epoch/100))
                episode_reward = []
                episode_length = []


def run_exp():

    wandb_run = wandb.init(
        project="REM_DQN"
    )
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    config = wandb.config
    agent = REMAgent(config)
    agent.reset()
    agent.train_rem()
    agent.evaluate(wandb_run)


if __name__ == "__main__":
   
    # --- Parse the config files --- #

    parser = argparse.ArgumentParser()
    parser.add_argument("-config")
    args = parser.parse_args()
    config = OmegaConf.load(args.config)
    config_dict = OmegaConf.to_container(config, resolve=True )

    # --- Create Wandb Sweep --- #

    project_name = "REM_DQN"
    sweep_id = wandb.sweep(sweep=config_dict, project=project_name)
    agent = wandb.agent(sweep_id, function=run_exp, count=1)
        
            

