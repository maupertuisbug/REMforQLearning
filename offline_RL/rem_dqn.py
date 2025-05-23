from offline_RL.base_dqn import BaseAgent






class REMAgent(BaseAgent):
    def __init__(self, config, wandb_run):

        super().__init__(config, wandb_run)
    

    def train_agent(self):

        for epoch in range(self.epochs):

            total_loss = 0 
            data = self.replay_buffer.sample(self.batch_size)
            probabilities = np.random.dirichlet(alpha=np.ones(self.k), size=1)[0]
            td_target = 0 
            current_value = 0 
            for k in range(self.k):

                q_network = self.q_network[k]
                target_network  = self.target_network[k]

                target_output = q_network(data.next_observations)
                target_max, _ = target_network(data.next_observations).max(dim=2)
                td_target = td_target + data.rewards.flatten() + probabilities[k] * (self.gamma * target_max * (1- data.dones.flatten()))

                current_value_ = q_network(data.observations).squeeze(0)
                current_value  = current_value + probabilities[k] * (current_value_.gather(1, data.actions).squeeze().unsqueeze(0))

            loss = torch.nn.functional.mse_loss(td_target, current_value)

            for k in range(self.k):
                self.optimizers[k].zero_grad()

            loss.backward()

            for k in range(self.k):
                self.optimizers[k].step()

            if epoch % self.target_network_frequency == 0:

                for k in range(self.k):
                    for target_network_param, q_network_param in zip(self.target_networks[k].parameters(), self.q_networks[k].parameters()):
                        target_network_param.data.copy_(
                            self.tau * q_network_param.data + (1.0 - self.tau) * target_network_param.data
                        )

    def evaluate_agent(self):

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
                        q_network = self.q_network[k]
                        q_values_ = q_network(torch.tensor(obs, device = self.device)).squeeze(0)
                        e_values =  q_values + probabilities[k] * q_values_
                    
                    actions = torch.argmax(q_values, dim=1).cpu().numpy()
                
                next_obs, rewards, terminated, truncated, info = self.envs.step(actions)
                _next_obs = next_obs.copy()
                obs = _next_obs
                total_reward = total_reward + rewards 

                if terminated :
                    episode += 1 
                    episode_length.append(ep_length)
                    episode_reward.append(total_reward)
                    break 
            
            if epoch % 100 == 0 :
                self.wandb_run.log({"Average Episode Reward " : np.mean(episode_reward)}, step = int(epoch/100))
                self.wandb_run.log({"Average Episode Length " : np.mean(episode_length)}, step = int(epoch/100))
                episode_reward = []
                episode_length = []



