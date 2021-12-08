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
from typing import Dict

from evostrat import Individual

from src.models.models import MLP_policy
from src.config import conf
from src.environment_wrappers.env_wrappers import SkillWrapper, RewardWrapper, SkillWrapperFinetune


class Hopper(Individual):
    """
    Hopper controlled by a feedforward policy network
    """
    # Static variables
    d = None # the discriminator
    n_skills = None # the number of skills
    skill = None # skill for the case of finetuning
    def __init__(self):
        self.net = MLP_policy(11 + conf.n_z, [conf.layer_size_policy, conf.layer_size_policy], 3)
        self.conf = conf
        self.t = 0

    @staticmethod
    def from_params(params: Dict[str, t.Tensor]) -> 'Hopper':
        agent = Hopper()
        agent.net.load_state_dict(params)
        return agent

    def fitness(self, render=False) -> float:
        assert not (Hopper.d == None)
        assert not (Hopper.n_skills == None)
        # save the data for the discriminator replay buffer
        data = []
        if Hopper.skill:
            env = SkillWrapperFinetune(gym.make("Hopper-v2"), Hopper.n_skills, max_steps=self.conf.max_steps, skill=Hopper.skill)
        else:
            env = RewardWrapper(SkillWrapper(gym.make("Hopper-v2"), Hopper.n_skills, max_steps=self.conf.max_steps), Hopper.d, Hopper.n_skills)
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

    def get_params(self) -> Dict[str, t.Tensor]:
        return self.net.state_dict()
    
    def load_model_params(self, path):
        self.net.load_state_dict(torch.load(path))

    def action(self, obs):
        # print(f"obs {obs}")
        with t.no_grad():
            return self.net.sample_action(torch.tensor(obs).unsqueeze(dim=0)).numpy()
        
    def split_obs(self, obs):
        env_obs = torch.clone(torch.tensor(obs[: -Hopper.n_skills]))
        skills = torch.argmax(torch.tensor(obs[-Hopper.n_skills:]), dim=-1)
        return env_obs, skills

