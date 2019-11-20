import numpy as np
import gym
from gym.spaces import Box, Discrete, MultiDiscrete

import ray
from ray import tune


class A2BEnv(gym.Env):
    def __init__(self, env_config):
        # goal position for agent
        self.desired_state = np.array([0.0,0.0])
        # agent can move by up to 0.1 units in any direction
        self.action_space = Box(low=-1.0, high=1.0, shape=(2,), dtype=np.float32)
        # agent's state is its position in R^2
        self.observation_space = Box(low=-np.inf, high=np.inf, shape=(2,), dtype=np.float32)


    def step(self, action):
        # agent moves
        self.state += action
        reward = - np.sqrt(np.linalg.norm(self.state - self.desired_state))
        done == reward > -0.3

        return (self.state, reward, done, {})

    def reset(self):
        # agent's initial position
        self.state = np.array([3.0,-3.0])  
        return self.state


# if doesn't work, test if custom_env works
if __name__ == "__main__":
    ray.init()
    tune.run(
        "PPO",
        config={
            "env": A2BEnv,
            "horizon": 200,
            "num_workers": 3,
            "lr": 1e-4,
            "gamma": 0.99,
        },
    )