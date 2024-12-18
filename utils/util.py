import gymnasium as gym
import ale_py
gym.register_envs(ale_py)
from stable_baselines3.common.atari_wrappers import(
    NoopResetEnv,
    ClipRewardEnv,
    EpisodicLifeEnv,
    FireResetEnv,
    MaxAndSkipEnv
)


def make_env(env_id, seed, idx, capture_video):

    def callable_env():

        if capture_video and idx == 0:
            env = gym.make(env_id, render_mode="rgb_array")
            env = gym.wrappers.RecordVideo(env, f"videos/{env_id}")
        else :
            env = gym.make(env_id)
        
        env = gym.wrappers.RecordEpisodeStatistics(env)
        env = NoopResetEnv(env, noop_max=30)
        env = MaxAndSkipEnv(env, skip=4)
        env = EpisodicLifeEnv(env)

        if "FIRE" in env.unwrapped.get_action_meanings():
            env = FireResetEnv(env)

        env = ClipRewardEnv(env)
        env = gym.wrappers.ResizeObservation(env, (84, 84))
        env = gym.wrappers.GrayscaleObservation(env)
        env.action_space.seed(seed)

        return env 
    
    return callable_env


def epsilon_schedule(epsilon_a, epsilon_b, duration, t):
    if t < 1000000:
        slope = (epsilon_b - epsilon_a)/duration
        return max(slope*t + epsilon_a, epsilon_b)
    else :
        return 0.1
