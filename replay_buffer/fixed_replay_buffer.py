import numpy as np 

class FixedReplayBuffer:

    def __init__(self, capacity):
        
        self.capacity = capacity+10 
        self.buffer = [] 
        self.index = 0 

    
    def add(self, obs, action, next_obs, reward, done):

        if len(self.buffer) < self.capacity:
            self.buffer.append([obs, action, next_obs, reward, done])
            self.index = self.index + 1
        
        else : 
            self.buffer = self.buffer[20000:]
            self.index = 180000
            self.buffer.append([obs, action, next_obs, reward, done])
            self.index = self.index + 1

    def sample(self, batch_size):

        indices = np.random.choice(len(self.buffer), batch_size, replace=False)
        batch = [self.buffer[idx] for idx in indices]

        obs, action, next_obs, reward, done = map(np.stack, zip(*batch))

        obs = obs.astype(np.float64)
        action = action.astype(np.float64)
        next_obs = next_obs.astype(np.float64)
        reward = reward.astype(np.float64)
        done = done.astype(np.float64)

        return obs, action, next_obs, reward, done
    
    def sample_all(self):

        indices = np.random.choice(len(self.buffer), self.index, replace=False)
        batch = [self.buffer[idx] for idx in indices]
        obs, action, next_obs, reward, done = map(np.stack, zip(*batch))

        obs = obs.astype(np.float64)
        action = action.astype(np.float64)
        next_obs = next_obs.astype(np.float64)
        reward = reward.astype(np.float64)
        done = done.astype(np.float64)

        return obs, action, next_obs, reward, done 
