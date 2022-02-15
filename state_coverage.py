'''
This file plot the state distrbution of an policy trained on MountainCar given the list of algorithms, timestamps, labels (optional), and the number of skills.
'''
import torch
import gym
import argparse


from src.utils import augment_obs
from src.environment_wrappers.env_wrappers import SkillWrapper
from src.config import conf
from src.utils import seed_everything

from stable_baselines3.common.vec_env import VecVideoRecorder, DummyVecEnv
from stable_baselines3.common.utils import get_schedule_fn
from stable_baselines3 import PPO, SAC

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
sns.set()
sns.set_style('whitegrid')
sns.set_context("paper", font_scale = conf.font_scale)

import os
# Extract arguments from terminal
seed = 10



def run_pretrained_policy(env_name, n_skills, pm, lb, alg, stamp):
    # load the model
    main_exper_dir = conf.log_dir_finetune + f"cls:{pm}, lb:{lb}/"
    env_dir = main_exper_dir + f"env: {env_name}, alg:{alg}, stamp:{stamp}/"
    seed_dir = env_dir + f"seed:{seed}/"
    seed_everything(seed)
    env = DummyVecEnv([lambda: SkillWrapper(gym.make(env_name), n_skills, max_steps=1000)])
    if alg == "sac":
        model_dir = seed_dir + "/best_model"
        model = SAC.load(model_dir, env=env, seed=seed)
    elif alg == "ppo":
        model_dir = seed_dir + "/best_model"
        hyper_params = pd.read_csv(env_dir + "ppo_hyperparams.csv").to_dict('records')[0]
        model = PPO.load(model_dir, env=env, seed=seed, clip_range= get_schedule_fn(hyper_params['clip_range']))
    entropy_list = []
    data = []
    with torch.no_grad():
        for i in range(10):
            env = DummyVecEnv([lambda: gym.make(env_name) ])
            env.seed(seed)
            for skill in range(n_skills):
                obs = env.reset()
                obs = obs[0]
                data.append(obs.copy())
                aug_obs = augment_obs(obs, skill, n_skills)
                total_reward = 0
                done = False
                while not done:
                    action, _ = model.predict(aug_obs, deterministic=False)
                    obs, _, done, _ = env.step(action)
                    obs = obs[0]
                    # print(obs)
                    data.append(obs.copy())
                    aug_obs = augment_obs(obs, skill, n_skills)
    data = np.array(data)
    np.random.shuffle(data)
    return data
    
def scatter_plotter(data, alg, color):
    plt.scatter(data[:, 0], data[:, 1], c=color, label=alg, alpha=0.2)
    
    
    
if __name__ == "__main__":
    # run_pretrained_policy(args)
    env_name = "MountainCarContinuous-v0"
    skills_l = [10, 10]
    algs = ["sac", "ppo"]
    stamps = [1643572784.9260342, 1644513661.0212696]
    cls = ["MLP", "MLP"]
    lbs = ["ba", "ba"]
    colors = ["b", "r"]
    legends = [a.upper() for a in algs]
    plt.figure()
    plt.title("MountainCar \nState Coverage")
    for i in range(len(algs)):
        color = colors[i]
        alg = algs[i]
        stamp = stamps[i]
        pm = cls[i]
        lb = lbs[i]
        n_skills = skills_l[i]
        data = run_pretrained_policy(env_name, n_skills, pm, lb, alg, stamp)
        scatter_plotter(data, algs[i], color)
    plt.legend()
    plt.savefig("Vis/MountainCar state covarge.png")