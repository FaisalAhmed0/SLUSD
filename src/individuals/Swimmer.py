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

from src.models.models import MLP_policy
from src.config import conf
from src.environment_wrappers.env_wrappers import SkillWrapper, RewardWrapper, SkillWrapperFinetune


class Swimmer(Individual):
    """
    Swimmer controlled by a feedforward policy network
    """
    # Static variables
    d = None # the discriminator
    n_skills = None # the number of skills
    skill = None # skill for the case of finetuning
    paramerization = None
    def __init__(self):
        self.net = MLP_policy(8 + conf.n_z, [conf.layer_size_policy, conf.layer_size_policy], 2)
        self.conf = conf
        self.t = 0

    @staticmethod
    def from_params(params: Dict[str, torch.Tensor]) -> 'Swimmer':
        agent = Swimmer()
        agent.net.load_state_dict(params)
        return agent

    def fitness(self, render=False) -> float:
        assert not (Swimmer.d == None)
        assert not (Swimmer.n_skills == None)
        # save the data for the discriminator replay buffer
        data = []
        if Swimmer.skill:
            env = SkillWrapperFinetune(gym.make("Swimmer-v2"), Swimmer.n_skills, max_steps=self.conf.max_steps, skill=Swimmer.skill)
        else:
            env = RewardWrapper(SkillWrapper(gym.make("Swimmer-v2"), Swimmer.n_skills, max_steps=self.conf.max_steps), Swimmer.d, Swimmer.n_skills, Swimmer.paramerization)
        obs = env.reset()
        done = False
        r_tot = 0
        self.t += 1
        while not done:
            self.t += 1
            env_obs, skill = self.split_obs(obs)
            env_obs = torch.clone(env_obs)
            skill = torch.tensor(skill.numpy())
            data.append((env_obs.numpy(), skill.item()))
            action = self.action(obs)
            obs, r, done, _ = env.step(action)
            r_tot += r
        env.close()
        return r_tot, data, self.t

    def get_params(self) -> Dict[str, torch.Tensor]:
        return self.net.state_dict()
    
    def load_model_params(self, path):
        self.net.load_state_dict(torch.load(path))

    def action(self, obs):
        # print(f"obs {obs}")
        with torch.no_grad():
            return self.net.sample_action(torch.tensor(obs).unsqueeze(dim=0)).numpy()
        
    def split_obs(self, obs):
        env_obs = torch.clone(torch.tensor(obs[: -Swimmer.n_skills]))
        skills = torch.argmax(torch.tensor(obs[-Swimmer.n_skills:]), dim=-1)
        return env_obs, skills
