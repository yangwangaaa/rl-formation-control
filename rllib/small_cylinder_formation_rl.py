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
from math import *


# doc: https://ray.readthedocs.io/en/latest/rllib.html
# example: https://github.com/ray-project/ray/blob/master/rllib/examples/custom_env.py


HORIZON = 500



class SwarmRoboticsEnv(gym.Env):
    def __init__(self, env_config):
        self.w = env_config['width']
        self.h = env_config['height']
        self.r = 1
        # params
        self.h_theta = 2.0 * pi / (self.w - 1)
        self.h_s = 1.0 / (self.h - 1)
        self.betas = np.array([0, 0, 0])
        self.lambdas = np.array([1, 1, 1])
        # pre-computations
        self.h_theta2 = self.h_theta * self.h_theta
        self.h_s2 = self.h_s * self.h_s
        self.betas_hs = self.betas / (2 * self.h_s)
        # simulation params
        self.dt = 1e-4

        # state and action spaces
        self.action_space = Box(low=-0.1,
                                high=0.1,
                                shape=(self.w * 3,),
                                dtype=np.float32)
        self.observation_space = Box(low=-np.inf, 
                                     high=np.inf,
                                     shape=(self.h, self.w, 3),
                                     dtype=np.float32)

        # compute objective state
        self.desired_state = np.zeros((self.h, self.w, 3))

        for i, theta in enumerate(np.linspace(0, 2 * pi, num=self.w, endpoint=False)):
            self.desired_state[:, i, 0] = self.r * cos(theta)
            self.desired_state[:, i, 1] = self.r * sin(theta)
            self.desired_state[1:-2, i, 0] = self.r * cos(theta)
            self.desired_state[1:-2, i, 1] = self.r * sin(theta)
        for z in range(self.h):
            self.desired_state[z, :, 2] = float(z) / (self.h - 1)


    def step(self, action):
        """Run one timestep of the environment's dynamics. When end of
        episode is reached, you are responsible for calling `reset()`
        to reset this environment's state.

        input: action
        returns: observation (obj), reward (float), done (bool), info (dict)
        """
        # apply actions
        self.state[self.h-1] += action.reshape((self.w, 3))

        for _ in range(10):
            # one env step
            new_state = np.copy(self.state)
            for z in range(1, self.h - 1):
                for i in range(self.w):
                    # update rule for (x, y, z)
                    new_state[z, i] = self.state[z, i] + self.dt * (
                        (self.state[z, (i + 1) % self.w] - 2 * self.state[z, i] + self.state[z, i - 1]) / self.h_theta2 +
                        (self.state[z + 1, i] - 2 * self.state[z, i] + self.state[z - 1, i]) / self.h_s2 +
                        self.betas_hs * (self.state[z, (i + 1) % self.w] - self.state[z, i - 1]) +
                        self.lambdas * self.state[z, i]
                    )
            self.state = np.copy(new_state)

        # compute reward
        reward =  np.linalg.norm(self.desired_state - self.state) / np.linalg.norm(self.desired_state)

        done = False
        return (self.state, reward, done, {})

    def reset(self):
        """Resets the state of the environment and returns an initial observation.
        
        returns: observation (obj): the initial observation.
        """
        agents = np.zeros((self.h, self.w, 3))

        for i, theta in enumerate(np.linspace(0, 2 * pi, num=self.w, endpoint=False)):
            agents[:, i, 0] = self.r * cos(theta)
            agents[:, i, 1] = self.r * sin(theta)
            agents[1:-2, i, 0] = 3*self.r * cos(theta)
            agents[1:-2, i, 1] = 3*self.r * sin(theta)
        for z in range(self.h):
            agents[z, :, 2] = float(z) / (self.h - 1)

        print("RESET ENV")

        self.state = agents
        return self.state

    def render(self, mode='human'):
        if mode != 'human':
            raise NotImplementedError
        pass
        print("||| RENDERING |||")




# https://ray.readthedocs.io/en/latest/rllib-env.html


# config: https://github.com/ray-project/ray/blob/master/rllib/agents/trainer.py
# algo: available options are SAC, PPO, PG, A2C, A3C, IMPALA, ES, DDPG, DQN, MARWIL, APEX, and APEX_DDPG

# checkpoint_freq, restore?

# see results: tensorboard --logdir=~/ray_results


if __name__ == "__main__":
    ray.init()

    # params https://github.com/ray-project/ray/blob/master/python/ray/tune/tune.py
    tune.run(
        "PPO",
        # name="small_cylinder_formation_rl",
        # stop={
        #     "timesteps_total": 1000000,
        # },
        # checkpoint_freq=50,
        # checkpoint_at_end=True,
        # resources_per_trial={"cpu":4, "gpu":0},
        # local_dir="~/ray_results",
        # restore=None,
        # resume=False,
        # max_failures=3,
        # reuse_actors=False,
        config={
            "env": SwarmRoboticsEnv,
            "env_config": { "width": 4, "height": 2 },  # config to pass to env class
            # "vf_share_layers": True,
            "gamma": 0.99,
            "horizon": HORIZON,
            "lr": 1e-4,
            # "num_envs_per_worker": 4,
            "num_workers": 3,
            "num_gpus": 0,
            # "monitor": True # save gym episode videos to the result dir?
        },
    )


"""
    training_iteration
    episode_reward_mean
    episodes_this_iter
    timesteps_this_iter
    time_this_iter
    episode_len_mean
    episodes_total
    episode_reward_min
    episode_reward_max
    timesteps_per_iteration
    (num_workers)
"""


# pip install setproctitle or ray[debug]
# pip install lz4
# use python3