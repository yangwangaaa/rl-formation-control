from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
import gym
from gym.spaces import Discrete, Box

import ray
from ray import tune
from ray.tune import grid_search

import random as rd
from math import pi, sin, cos


class SwarmRoboticsEnv(gym.Env):
    def __init__(self, env_config):
        self.w = 4
        self.h = 4
        # self.action_space = Box(low=0, high=1, shape=(5,), dtype=np.float32)
        self.observation_space = Discrete(2)
        self.action_space = Box(low=-0.1,
                                high=0.1,
                                shape=self.w * 3,
                                dtype=np.float32)
        # self.observation_space = Box(low=-np.inf, 
        #                              high=np.inf,
        #                              shape=(self.h, self.w, 3),
        #                              dtype=np.float32)

    def step(self, action):
        reward = 0
        done = False
        return (self.state, reward, done, {})

    def reset(self):
        # self.state = np.zeros((self.h, self.w, 3))
        self.state = 0
        return self.state


if __name__ == "__main__":
    ray.init()
    tune.run(
        "PPO",
        config={
            "env": SwarmRoboticsEnv,
        },
    )