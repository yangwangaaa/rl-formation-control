import gym, ray
from ray.rllib.agents import ppo
from gym import spaces
import random as rd


"""

State space: 
    (x,y,z) for each agent (PO: for each leader)

Action space:
    (dx,dy,dz) for each leader

Reward function:
    opposite of 2-norm between agents formation and desired formation


Step function:
    state += action
    reward = state - desired_state

"""


class SwarmRoboticsEnv(gym.Env):
    def __init__(self, env_config):
        self.action_space = spaces.Discrete(3)
        self.observation_space = spaces.Discrete(22)

        self.reset()

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

        if done:
            self.reset()

        return (self.state, reward, done, {})

    def reset(self):
        """Resets the state of the environment and returns an initial observation.
        
        returns: observation (obj): the initial observation.
        """
        self.state = 21
        return self.state

    def render(self, mode='human'):
        if mode != 'human':
            raise NotImplementedError
        pass
        print("||| RENDERING |||")




# https://ray.readthedocs.io/en/latest/rllib-env.html

ray.init()
trainer = ppo.PPOTrainer(env=MatchesEnv, config={
    "env_config": {},  # config to pass to env class
})

while True:
    print(trainer.train())