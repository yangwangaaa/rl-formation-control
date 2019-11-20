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


# doc: https://ray.readthedocs.io/en/latest/rllib.html
# example: https://github.com/ray-project/ray/blob/master/rllib/examples/custom_env.py


class MatchesEnv(gym.Env):
    def __init__(self, env_config):
        self.action_space = Discrete(3)
        self.observation_space = Discrete(22)
        self.initial = env_config["initial"]

    def step(self, action):
        """Run one timestep of the environment's dynamics. When end of
        episode is reached, you are responsible for calling `reset()`
        to reset this environment's state.

        input: action
        returns: observation (obj), reward (float), done (bool), info (dict)
        """
        action = action + 1
        self.state -= action

        if self.state <= 0:
            reward = -1
            done = True
        else:
            self.state -= rd.randint(1, 3)
            if self.state <= 0:
                reward = 1
                done = True
            else:
                reward = 0
                done = False

        # if done:
        #     self.reset()
        if done:
            self.state = 0

        return (self.state, reward, done, {})

    def reset(self):
        """Resets the state of the environment and returns an initial observation.
        
        returns: observation (obj): the initial observation.
        """
        self.state = self.initial
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
        name="test_matches",
        # stop={
        #     "timesteps_total": 1000000,
        # },
        checkpoint_freq=50,
        checkpoint_at_end=True,
        ressources_per_trial={"cpu":4, "gpu":0},
        local_dir="~/ray_results",
        restore=None,
        resume=False,
        max_failures=3,
        reuse_actors=False,
        config={
            "env": MatchesEnv,
            "env_config": { "initial": 21 },  # config to pass to env class
            # "vf_share_layers": True,
            "gamma": 0.99,
            "horizon": 1000,
            "lr": 1e-4,
            "num_envs_per_worker": 4,
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