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

from src.environment_wrappers.env_wrappers import SkillWrapperFinetune, RewardWrapper, SkillWrapper, TimestepsWrapper
from src.config import conf
from src.utils import seed_everything


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
seed = 0
seed_everything(seed)



# ppo hyperparams
ppo_hyperparams = dict(
    n_steps = 2048,
    learning_rate = 3e-4,
    n_epochs = 10,
    batch_size = 64,
    gamma = 0.99,
    gae_lambda = 0.95,
    clip_range = 0.1,
    ent_coef=0.5,
    n_actors = 8,
    algorithm = "ppo",
    
)

# SAC Hyperparameters 
sac_hyperparams = dict(
    learning_rate = 3e-4,
    gamma = 0.99,
    buffer_size = int(1e7),
    batch_size = 128,
    tau = 0.005,
    gradient_steps = 1,
    ent_coef=0.5,
    learning_starts = 1000,
    algorithm = "sac"
)

# Extract arguments from terminal
def cmd_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--envs', nargs="*", type=str, default=["MountainCarContinuous-v0"])
    parser.add_argument("--algs", nargs="*", type=str, default=["ppo"]) # pass the algorithms that correspond to each timestamp
    parser.add_argument("--skills", nargs="*", type=int, default=[6])
    parser.add_argument("--bestskills", nargs="*", type=int, default=[-1])
    parser.add_argument("--stamps", nargs="*", type=str, required=True) # pass the timestamps as a list
    parser.add_argument("--cls", nargs="*", type=str, required=True) # pass the timestamps as a list
    parser.add_argument("--lbs", nargs="*", type=str, required=True) # pass the timestamps as a list
    args = parser.parse_args()
    return args

# A method to augment observations with the skills reperesentation
def augment_obs(obs, skill, n_skills):
    onehot = np.zeros(n_skills)
    onehot[skill] = 1
    aug_obs = np.array(list(obs) + list(onehot)).astype(np.float32)
    return torch.FloatTensor(aug_obs).unsqueeze(dim=0)


# A method to return the best performing skill
@torch.no_grad()
def record_skill(model, env_name, alg, n_skills):
    display = pyglet.canvas.get_display()
    screen = display.get_screens()
    config = screen[0].get_best_config()
    pyglet.window.Window(width=1024, height=1024, display=display, config=config)
    total = []
    total_rewards = []
    for skill in range(n_skills):
        print(f"Running Skill: {skill}")
        env = TimestepsWrapper(gym.make(env_name))
        env.seed(seed)
        video_folder = f"recorded_agents/env:{env_name},alg: {alg}_skills_videos"
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



def record_skills(env_name, alg, skills, stamp, pm, lb):
    main_exper_dir = conf.log_dir_finetune + f"cls:{pm}, lb:{lb}/"
    env_dir = main_exper_dir + f"env: {env_name}, alg:{alg}, stamp:{stamp}/"
    seed_dir = env_dir + f"seed:{seed}/"
    model_dir = seed_dir + "/best_model"
    env = DummyVecEnv([lambda: SkillWrapper(gym.make(env_name),skills, max_steps=1000)])
    if alg == "sac":
        model = SAC.load(model_dir, env=env, seed=seed)
    elif alg == "ppo":
        model = PPO.load(model_dir, env=env,  clip_range= get_schedule_fn(ppo_hyperparams['clip_range']), seed=seed)
    best_skill_index = record_skill(model, env_name, alg, skills)
    print(f"Best skill is {best_skill_index}")


def record_learned_agent(best_skill, env_name, alg, skills, stamp, pm, lb):
    # TODO: test this
    video_folder = f"recorded_agents/env:{env_name}_alg: {alg}_finetune_videos"
    env = SkillWrapperFinetune(gym.make(env_name), skills, max_steps=gym.make(env_name)._max_episode_steps, skill=best_skill)
    env = Monitor(env, video_folder, resume=True,force=False, uid=f"env: {env}, skill: {best_skill}")
    main_exper_dir = conf.log_dir_finetune + f"cls:{pm}, lb:{lb}/"
    env_dir = main_exper_dir + f"env: {env_name}, alg:{alg}, stamp:{stamp}/"
    seed_dir = env_dir + f"seed:{seed}/"
    # I stopped here
    directory =  seed_dir + f"best_finetuned_model_skillIndex:{best_skill}/" + "/best_model"
    if alg == "sac":
        model = SAC.load(directory, seed=seed)
    elif alg == "ppo":
        model = PPO.load(directory, clip_range= get_schedule_fn(ppo_hyperparams['clip_range']), seed=seed)
    obs = env.reset()
    n_skills = skills
    total_reward = 0
    done = False
    while not done:
        action, _ = model.predict(obs, deterministic=True)
        obs, reward, done, _ = env.step(action)
        obs = obs
        total_reward += reward
    env.close()

if __name__ == "__main__":
    args = cmd_args()
    n = len(args.envs)
    for i in range(n):
        env_name = args.envs[i]
        alg = args.algs[i]
        skills = args.skills[i]
        stamp = args.stamps[i]
        pm = args.cls[i]
        lb = args.lbs[i]
        best_skill = args.bestskills[i]
        record_skills(env_name, alg, skills, stamp, pm, lb)
        if best_skill != -1:    
            record_learned_agent(best_skill, env_name, alg, skills, stamp, pm, lb)

