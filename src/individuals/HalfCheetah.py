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
from src.environment_wrappers.tasks_wrappers import HalfCheetahTaskWrapper


class HalfCheetah(Individual):
    """
    HalfCheetah controlled by a feedforward policy network
    """
    # Static variables
    # the discriminator
    d = None 
    n_skills = None # the number of skills
    skill = None # skill for the case of finetuning
    paramerization = None
    task = None
    def __init__(self):
        self.net = MLP_policy(17 + HalfCheetah.n_skills, [conf.layer_size_policy, conf.layer_size_policy], 6)
        self.conf = conf
        self.t = 0

    @staticmethod
    def from_params(params: Dict[str, torch.Tensor]) -> 'HalfCheetah':
        agent = HalfCheetah()
        agent.net.load_state_dict(params)
        return agent

    def fitness(self, render=False) -> float:
        assert not (HalfCheetah.d == None)
        assert not (HalfCheetah.n_skills == None)
        # save the data for the discriminator replay buffer
        data = []
        env = self.environment()
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
            # print(f"action: {action}")
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
        env_obs = torch.clone(torch.tensor(obs[: -HalfCheetah.n_skills]))
        skills = torch.argmax(torch.tensor(obs[-HalfCheetah.n_skills:]), dim=-1)
        return env_obs, skills
    
    def environment(self):
        '''
        Return a gym environment according to the class variables
        '''
        if HalfCheetah.task:
            env = HalfCheetahTaskWrapper(gym.make("HalfCheetah-v2"), task=task)
        else:
            env = gym.make("HalfCheetah-v2")
        if HalfCheetah.skill:
            env = SkillWrapperFinetune(env, HalfCheetah.n_skills, max_steps=self.conf.max_steps, skill=HalfCheetah.skill)
        else:
            env = RewardWrapper(SkillWrapper(env, HalfCheetah.n_skills, max_steps=self.conf.max_steps), HalfCheetah.d, HalfCheetah.n_skills, HalfCheetah.paramerization)
        return env
        

