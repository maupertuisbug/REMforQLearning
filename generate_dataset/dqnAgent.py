import random 
import torch
import numpy as np
import gymnasium as gym
import argparse
from omegaconf import OmegaConf
import wandb
from replay_buffer.fixed_replay_buffer import FixedReplayBuffer
from stable_baselines3.common.buffers import ReplayBuffer
from utils.util import make_env, epsilon_schedule
import ale_py
import torch.optim as optim

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
        self.flatten = torch.nn.Flatten(start_dim=1)  # This does not include the batch dim

    
    def forward(self, x):
        input = x/255.0
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
    

class DQNAgent:

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
        self.target_network_frequency = config.target_network_frequency
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    def reset(self):

        random.seed(self.seed)
        np.random.seed(self.seed)
        torch.manual_seed(self.seed)
        self.envs = gym.vector.SyncVectorEnv([make_env(self.env_id, self.seed, i, self.capture_video) for i in range(self.num_envs)])
        
        self.q_network = QNetwork(self.envs).to(self.device)
        self.optimizer = optim.Adam(self.q_network.parameters(), lr=self.learning_rate)
        self.target_network = QNetwork(self.envs).to(self.device)

        ac = self.envs.single_action_space.n
        

        self.replay_buffer = ReplayBuffer(
            self.buffer_size, 
            self.envs.single_observation_space, 
            self.envs.single_action_space, 
            self.device, 
            optimize_memory_usage = True, 
            handle_timeout_termination = False
        )
    
    def train(self, wandb_run):

        obs, _ = self.envs.reset(seed=self.seed)

        total_reward = 0

        for step in range(self.total_timesteps):

            epsilon = epsilon_schedule(self.epsilon_a, self.epsilon_b, self.exploration_fraction*self.total_timesteps, step)

            if random.random() < epsilon:
                actions = self.envs.action_space.sample()
            else :
                q_values = self.q_network(torch.tensor(obs.unsqueeze(0), device=self.device))
                actions = torch.argmax(q_values, dim=1).cpu().numpy()
            
            next_obs, rewards, terminated, truncated, info = self.envs.step(actions)

            _next_obs = next_obs.copy()

            for idx, d in enumerate(truncated):
                if d :
                    _next_obs[idx] = info["final_observation"][idx]
            
            self.replay_buffer.add(obs, _next_obs, actions, rewards, terminated, info)

            obs = _next_obs
            total_reward = total_reward + rewards

            if step > self.step_start_learning :
                if step% self.training_frequency == 0:
                    data = self.replay_buffer.sample(self.batch_size)
                    with torch.no_grad():
                        target_output = self.target_network(data.next_observations.unsqueeze(1))
                        target_max, _ = self.target_network(data.next_observations).max(dim=2)
                        td_target = data.rewards.flatten() + self.gamma * target_max * (1 - data.dones.flatten())

                    current_value = self.q_network(data.observations).gather(1, data.actions).squeeze()
                    loss = torch.nn.F.mse_loss(td_target, current_value)


                    if step%100 == 0:
                        """
                        wandbd log
                        """
                        wandb_run.log({"total_reward" : total_reward}, step = step)
                    
                    self.optimizer.zero_grad()
                    loss.backward()
                    self.optimizer.step()

                if step % self.target_network_frequency == 0:
                    for target_network_param, q_network_param in zip(self.target_network.parameters(), self.q_network.parameters()):
                        target_network_param.data.copy_(
                            self.tau * q_network_param.data + (1.0 - self.tau)*target_network_param.data
                        )
def run_exp():

    wandb_run = wandb.init(
        project="test_dqn"
    )

    config = wandb.config
    agent = DQNAgent(config)
    agent.reset()
    agent.train(wandb_run)


if __name__ == "__main__":
    """
    Parse the config file
    """

    parser = argparse.ArgumentParser()
    parser.add_argument("-config")
    args = parser.parse_args()
    config = OmegaConf.load(args.config)
    config_dict = OmegaConf.to_container(config, resolve=True )

    """
    Create the wandb run
    """

    project_name = "test_dqn"
    sweep_id = wandb.sweep(sweep=config_dict, project=project_name)
    agent = wandb.agent(sweep_id, function=run_exp, count=1)







        

        




    



