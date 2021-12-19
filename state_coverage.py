'''
This file plot the state distrbution of an policy trained on MountainCar given the list of algorithms, timestamps, labels (optional), and the number of skills.
'''
import torch
import gym
import argparse

import numpy as np
import matplotlib.pyplot as plt

from src.utils import augment_obs
from src.environment_wrappers.env_wrappers import SkillWrapper
from src.config import conf

from stable_baselines3.common.vec_env import VecVideoRecorder, DummyVecEnv
from stable_baselines3.common.utils import get_schedule_fn
from stable_baselines3 import PPO, SAC

import os
# Extract arguments from terminal
def cmd_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--env', type=str, default="MountainCarContinuous-v0")
    parser.add_argument("--algs", nargs="*", type=str, default=["ppo"]) # pass the algorithms that correspond to each timestamp
    parser.add_argument("--skills", type=int, default=6)
    parser.add_argument("--stamps", nargs="*", type=str, required=True) # pass the timestamps as a list
    parser.add_argument("--colors", nargs="*", type=str, required=False, default=['b', 'r']) # pass the timestamps as a list
    parser.add_argument("--labels", nargs="*", type=str, required=False, default=None) # add custom labels 
    args = parser.parse_args()
    return args


def run_pretrained_policy(args):
    # load the model
    env = DummyVecEnv([lambda: SkillWrapper(gym.make(args.env), args.skills, max_steps=1000)])
    # directory =  conf.log_dir_finetune + f"{args.alg}_{args.env}_{args.stamp}" + "/best_model"
    directory = conf.log_dir_finetune + f"{args.alg}_{args.env}_skills:{args.skills}_{args.stamp}" + "/best_model"
    if args.alg == "sac":
        model = SAC.load(directory)
    elif args.alg == "ppo":
        model = PPO.load(directory, clip_range= get_schedule_fn(0.1))
    # run the model to collect the data
    # print(model)
    seeds = [0, 10, 1234, 5, 42]
    entropy_list = []
    data = []
    for seed in seeds:
        with torch.no_grad():
            for i in range(5):
                env = DummyVecEnv([lambda: gym.make(args.env) ])
                env.seed(seed)
                for skill in range(args.skills):
                    obs = env.reset()
                    obs = obs[0]
                    data.append(obs.copy())
                    aug_obs = augment_obs(obs, skill, args.skills)
                    total_reward = 0
                    done = False
                    while not done:
                        action, _ = model.predict(aug_obs, deterministic=False)
                        obs, _, done, _ = env.step(action)
                        obs = obs[0]
                        # print(obs)
                        data.append(obs.copy())
                        aug_obs = augment_obs(obs, skill, args.skills)
    data = np.array(data)
    np.random.shuffle(data)
    return data
    
def scatter_plotter(data, alg, color):
    plt.scatter(data[:, 0], data[:, 1], c=color, label=alg, alpha=0.2)
    
    
    
if __name__ == "__main__":
    args = cmd_args()
    # run_pretrained_policy(args)
    stamps = args.stamps
    colors = args.colors
    algs = args.algs
    plt.figure()
    plt.title("MountainCar \nState Coverage")
    for i in range(len(algs)):
        color = colors[i]
        alg = algs[i]
        args.stamp = stamps[i]
        args.alg = alg
        data = run_pretrained_policy(args)
        scatter_plotter(data, algs[i], color)
    plt.legend()
    plt.savefig("Vis/MountainCarContinuous-v0/MountainCar state covarge.png")