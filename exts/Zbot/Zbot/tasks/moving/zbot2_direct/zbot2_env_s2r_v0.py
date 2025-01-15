import gym
from gym import spaces
import numpy as np
import torch

class Zbot2EnvS2RV0(gym.Env):
    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        # Define action and observation space
        self.action_space = spaces.Discrete(4)
        self.observation_space = spaces.Box(low=0, high=255, shape=(64, 64, 3), dtype=np.uint8)
        
        # Initialize state
        self.state = None

    def reset(self):
        self.state = np.zeros((64, 64, 3), dtype=np.uint8)
        return self.state

    def step(self, action):
        self.state = np.random.randint(0, 256, (64, 64, 3), dtype=np.uint8)
        reward = 0
        done = False
        info = {"observations": {"critic": self.state}}
        return self.state, reward, done, info

    def render(self, mode='human'):
        pass

    def close(self):
        pass

    def get_observations(self):
        obs = torch.tensor(self.state, dtype=torch.float32, device=self.device).unsqueeze(0)
        extras = {"observations": {"critic": obs}}
        return obs, extras