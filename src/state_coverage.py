'''
This file plot the a multivarite normal fit and calculate its entropy, this file is used to check the state coverage of a pretrained policy on the MountainCar gym environment
'''
import torch
import gym
import argparse

import numpy as np

from src.utils import plot_multinorm, augment_obs
from src.environment_wrappers.env_wrappers import SkillWrapper
from src.config import conf

from stable_baselines3.common.vec_env import VecVideoRecorder, DummyVecEnv
from stable_baselines3.common.utils import get_schedule_fn
from stable_baselines3 import PPO, SAC

# Extract arguments from terminal
def cmd_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--env', type=str, default="MountainCarContinuous-v0")
    parser.add_argument("--alg", type=str, default="ppo")
    parser.add_argument("--skills", type=int, default=4)
    parser.add_argument("--stamp", type=str, required=True)
    args = parser.parse_args()
    return args


def run_pretrained_policy(args):
    # load the model
    env = DummyVecEnv([lambda: SkillWrapper(gym.make(args.env), args.skills, max_steps=1000)])
    directory =  conf.log_dir_finetune + f"{args.alg}_{args.env}_{args.stamp}" + "/best_model"
    if args.alg == "sac":
        model = SAC.load(directory, env = env)
    elif args.alg == "ppo":
        model = PPO.load(directory, env = env, clip_range= get_schedule_fn(0.1))
    data = []
    # run the model to collec the data
    with torch.no_grad():
        for i in range(10):
            for skill in range(args.skills):
                env = DummyVecEnv([lambda: gym.make(args.env) ])
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
    print(f"Size of the data is: {len(data)}")
    plot_multinorm(data, args)
    
    
if __name__ == "__main__":
    args = cmd_args()
    run_pretrained_policy(args)
    