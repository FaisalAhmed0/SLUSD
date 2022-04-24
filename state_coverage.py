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
seeds = [0, 10, 42]



def run_pretrained_policy(env_name, n_skills, pm, lb, alg, stamp):
    # load the model
    main_exper_dir = conf.log_dir_finetune + f"cls:{pm}, lb:{lb}/"
    env_dir = main_exper_dir + f"env: {env_name}, alg:{alg}, stamp:{stamp}/"
    data = []
    skills_list = []
    for seed in seeds:
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
        data_per_seed = []
        with torch.no_grad():
            for i in range(1):
                env = DummyVecEnv([lambda: gym.make(env_name) ])
                env.seed(seed)
                for skill in range(n_skills):
                    obs = env.reset()
                    obs = obs[0]
                    data_per_seed.append(obs.copy())
                    if seed == 0:
                        skills_list.append(skill)
                    aug_obs = augment_obs(obs, skill, n_skills)
                    total_reward = 0
                    done = False
                    i = 0
                    for _ in range(1000):
                        i += 1
                        action, _ = model.predict(aug_obs, deterministic=False)
                        obs, _, done, _ = env.step(action)
                        obs = obs[0]
                        # print(obs)
                        data_per_seed.append(obs.copy())
                        if seed == 0:
                            skills_list.append(skill)
                        aug_obs = augment_obs(obs, skill, n_skills)
                        # if done:
                        #     print(f"in done")
                        #     obs = env.reset()
                        #     obs = obs[0]
                        #     data_per_seed.append(obs.copy())
                        #     aug_obs = augment_obs(obs, skill, n_skills)
                        #     if seed == 0:
                        #         skills_list.append(skill)
                        #     total_reward = 0
                        #     done = False
                    print(f"i: {i}")
                print(obs.shape)
                print(len(data_per_seed))
            
        data.append(np.array(data_per_seed).astype(np.float32))
        print(np.array(data_per_seed))
        print(f"data per seed shape: {np.array(data_per_seed).shape}")
    data = np.array(data).mean(axis=0)
    data_d = {
        "Algorithm": alg.upper(),
        "Car Position": data[:,0],
        "Car Velocity": data[:,1],
        "Skill": skills_list
    }
    print(alg)
    print(data.shape)
    print(len(skills_list))
    # input()
    df = pd.DataFrame(data_d)
    print(data.shape)
    np.random.shuffle(data)
    return data, df
    
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
    dfs = []
    for i in range(len(algs)):
        color = colors[i]
        alg = algs[i]
        stamp = stamps[i]
        pm = cls[i]
        lb = lbs[i]
        n_skills = skills_l[i]
        data, df = run_pretrained_policy(env_name, n_skills, pm, lb, alg, stamp)
        dfs.append(df)
        scatter_plotter(data, algs[i], color)
    plt.legend()
    os.makedirs("Vis", exist_ok=True)
    plt.savefig(f"Vis/MountainCar_state_covarge.png")
    for i in range(len(dfs)):
        plt.figure()
        d = dfs[i]
        sns.scatterplot(x="Car Position", y="Car Velocity", hue="Skill", data=d)
        plt.legend()
        plt.savefig(f"Vis/MountainCar state covarge, {algs[i]}.png")
        