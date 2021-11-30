'''
This file define an indiviual for Evolution startigies algorithms
'''
import gym 
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as opt
from torch.multiprocessing import Pool
from torch.distributions import MultivariateNormal

import tqdm as tqdm

from evostrat import compute_centered_ranks, NormalPopulation


from typing import Dict

from evostrat import Individual

from src.models import MLP_policy
from src.congif import conf
from src.env_wrappers import SkillWrapper, RewardWrapper


class HalfCheetah(Individual):
    """
    Swimmer controlled by a feedforward policy network
    """

    def __init__(self):
        self.net = MLP_policy(17 + conf.n_Z, [conf.layer_size_policy, conf.layer_size_policy], 6)
        self.d = None

    @staticmethod
    def from_params(params: Dict[str, t.Tensor]) -> 'HalfCheetah':
        agent = HalfCheetah()
        agent.net.load_state_dict(params)
        return agent
    
    def set_discriminator(self, d):
        self.d = d

    def fitness(self, render=False) -> float:
        print(f"Discriminator is {self.d}")
        assert not (self.d is None)
        env = RewardWrapper(SkillWrapper(gym.make("HalfCheetah-v2"), conf.n_Z, max_steps=self.conf.max_steps), self.d, conf.n_z)
        obs = env.reset()
        done = False
        r_tot = 0
        while not done:
            action = self.action(obs)
            obs, r, done, _ = env.step(action)
            r_tot += r
            if render:
                env.render()

        env.close()
        return r_tot

    def get_params(self) -> Dict[str, t.Tensor]:
        return self.net.state_dict()

    def action(self, obs):
        # print(f"obs {obs}")
        with t.no_grad():
            return self.net.sample_action(torch.tensor(obs).unsqueeze(dim=0)).numpy()
