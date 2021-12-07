'''
This file record a video for each skill of the pretrained agent, and its performance on the final task
'''
import gym
import argparse

from stable_baselines3.common import callbacks
from stable_baselines3.common.vec_env import VecVideoRecorder, DummyVecEnv

from stable_baselines3 import PPO, SAC
from stable_baselines3.common.callbacks import  EvalCallback
from stable_baselines3.common.utils import get_schedule_fn
from stable_baselines3.common.monitor import Monitor

from src.environment_wrappers.env_wrappers import SkillWrapperFinetune, RewardWrapper, SkillWrapper
from src.config import conf


import torch
import torch.nn as nn
from torch.utils.tensorboard import SummaryWriter

import numpy as np
import pandas as pd

import time
import os
import random

from gym.wrappers import Monitor
import pyglet
from pyglet import *


# make sure the same seed
# set the seed
seed = 10
random.seed(seed)
np.random.seed(seed)
torch.random.manual_seed(seed)



# ppo hyperparams
ppo_hyperparams = dict(
    learning_rate = 3e-4,
    gamma = 0.99,
    batch_size = 64,
    n_epochs = 10,
    gae_lambda = 0.95,
    n_steps = 2048,
    clip_range = 0.1,
    ent_coef=1,
    algorithm = "ppo"
)

# SAC Hyperparameters 
sac_hyperparams = dict(
    learning_rate = 3e-4,
    gamma = 0.99,
    batch_size = 128,
    buffer_size = int(1e7),
    tau = 0.01,
    gradient_steps = 1,
    ent_coef=1,
    learning_starts = 10000,
    algorithm = "sac"
)

# Extract arguments from terminal
def cmd_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--env', type=str, default="MountainCarContinuous-v0")
    parser.add_argument("--alg", type=str, default="ppo")
    parser.add_argument("--skills", type=int, default=4)
    parser.add_argument("--stamp", type=str, required=True)
    parser.add_argument("--bestskill", type=int, default=-1)
    args = parser.parse_args()
    return args

# A method to augment observations with the skills reperesentation
def augment_obs(obs, skill, n_skills):
    onehot = np.zeros(n_skills)
    onehot[skill] = 1
    aug_obs = np.array(list(obs) + list(onehot)).astype(np.float32)
    # print(aug_obs)
    return torch.FloatTensor(aug_obs).unsqueeze(dim=0)


# A method to return the best performing skill
@torch.no_grad()
def record_skill(model, env_name, args):
    n_skills = args.skills
    display = pyglet.canvas.get_display()
    screen = display.get_screens()
    config = screen[0].get_best_config()
    pyglet.window.Window(width=1024, height=1024, display=display, config=config)
    # seeds = [0, 10, 1234, 5, 42]
    seeds = [0]
    total = []
    for seed in seeds:
        total_rewards = []
        for skill in range(n_skills):
            print(f"Running Skill: {skill}")
            env = gym.make(env_name)
            env.seed(seed)
            video_folder = f"recorded_agents/env:{args.env},alg: {args.alg}_skills_videos"
            env = Monitor(env, video_folder, resume=True,force=False, uid=f"env: {env_name}, skill: {skill}")
            obs = env.reset()
            aug_obs = augment_obs(obs, skill, n_skills)
            total_reward = 0
            done = False
            while not done:
                action, _ = model.predict(aug_obs, deterministic=True)
                obs, reward, done, info = env.step(action)
                aug_obs = augment_obs(obs, skill, n_skills)
                total_reward += reward
            total_rewards.append(total_reward)
            env.close()
        total.append(total_rewards)
    total_mean  = np.mean(total, axis=0)
    total_std = np.std(total, axis=0)
    print(f"Total: {total}")
    print(f"Total rewards mean: {total_mean}")
    print(f"Best skill: {np.argmax(total_mean)} and best reward is: {np.max(total_mean)} with std: {total_std[np.argmax(total_mean)]}")
    return np.argmax(total_mean)



def record_skills(args):
    env = DummyVecEnv([lambda: SkillWrapper(gym.make(args.env),args.skills, max_steps=1000)])
    # model_dir = conf.log_dir_finetune + f"{args.alg}_{args.env}_{args.stamp}" + "/best_model"
    model_dir = conf.log_dir_finetune + f"{args.alg}_{args.env}_skills:{args.skills}_{args.stamp}" + "/best_model"
    if args.alg == "sac":
        model = SAC.load(model_dir, env=env, seed=seed)
    elif args.alg == "ppo":
        model = PPO.load(model_dir, env=env,  clip_range= get_schedule_fn(ppo_hyperparams['clip_range']), seed=seed)
    
    best_skill_index = record_skill(model, args.env, args)
    print(f"Best skill is {best_skill_index}")


def record_learned_agent(best_skill, args):
    # TODO: test this
    env = SkillWrapperFinetune(gym.make(args.env), args.skills, max_steps=gym.make(args.env)._max_episode_steps, skill=best_skill)
    video_folder = f"recorded_agents/env:{args.env}_alg: {args.alg}_finetune_videos"

    env = Monitor(env, video_folder, resume=True,force=False, uid=f"env: {args.env}, skill: {best_skill}")
    
    # env = DummyVecEnv([lambda: SkillWrapperFinetune(gym.make(args.env), args.skills, max_steps=gym.make(args.env)._max_episode_steps, skill=best_skill)])
    # env = VecVideoRecorder(env, video_folder=f"recorded_agents/env:{args.env}_alg: {args.alg}_finetune_videos",
    #                           record_video_trigger=lambda step: step == 0, video_length=1000,
    #                           name_prefix = f"env: {args.env}, alg: {args.alg}, best_skill: {best_skill}")
    # input()
    directory =  conf.log_dir_finetune + f"{args.alg}_{args.env}_skills:{args.skills}_{args.stamp}/" + f"best_finetuned_model_skillIndex:{best_skill}/" + "/best_model"
    # model_dir = conf.log_dir_finetune + f"{args.alg}_{args.env}_{args.stamp}" + "/best_model"
    if args.alg == "sac":
        model = SAC.load(directory, seed=seed)
    elif args.alg == "ppo":
        model = PPO.load(directory, clip_range= get_schedule_fn(ppo_hyperparams['clip_range']), seed=seed)
    print(model)
    obs = env.reset()
    # print("#######")
    # print(obs)
    n_skills = args.skills
    # print(f"Number of skills: {n_skills}")
    # aug_obs = augment_obs(obs, best_skill, n_skills)
    # print("#########here#########")
    # print(aug_obs)
    # print(len(aug_obs))
    total_reward = 0
    done = False
    while not done:
        action, _ = model.predict(obs, deterministic=True)
        obs, reward, done, _ = env.step(action)
        obs = obs
        # aug_obs = augment_obs(obs, best_skill, n_skills)
        total_reward += reward
    # print(f"Final reward: {total_reward}")
    env.close()

if __name__ == "__main__":
    args = cmd_args()
    record_skills(args)
    if args.bestskill != -1:    
        best_skill = args.bestskill
        record_learned_agent(best_skill, args)

