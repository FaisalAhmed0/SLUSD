import gym
import argparse
import os

from src.config import conf
from src.diayn import DIAYN

import torch

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import time
import os
import random

# set the seed
seed = 10
random.seed(seed)
np.random.seed(seed)
torch.random.manual_seed(seed)





# shared parameters
params = dict( n_skills = 6,
           pretrain_steps = int(1e5),
           finetune_steps = int(1e5),
           buffer_size = int(1e7),
           min_train_size = int(1e4)
             )


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
    ent_coef=0.1,
    learning_starts = 10000,
    algorithm = "sac"
)

discriminator_hyperparams = dict(
    learning_rate = 3e-4,
    batch_size = 64,
    n_epochs = 1,
    weight_decay = 0,
    dropout = None,
    label_smoothing = None,
    gp = None,
    mixup = False
)

# save a timestamp
timestamp = time.time()


# Extract arguments from terminal
def cmd_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--env', type=str, default="MountainCarContinuous-v0")
    parser.add_argument('--skills', type=int, default=6)
    args = parser.parse_args()
    return args

# save the hyperparameters
def save_params(args, alg, directory):
    # save the hyperparams in a csv files
    alg_hyperparams = ppo_hyperparams if alg == "ppo" else sac_hyperparams
    
    alg_hyperparams_df = pd.DataFrame(alg_hyperparams, index=[0])
    alg_hyperparams_df.to_csv(f"{directory}/{alg}_hyperparams.csv")
    
    discriminator_hyperparams_df = pd.DataFrame(discriminator_hyperparams, index=[0])
    discriminator_hyperparams_df.to_csv(f"{directory}/discriminator_hyperparams.csv")
    
    # convert the config namespace to a dictionary
    exp_params_df = pd.DataFrame(params, index=[0])
    exp_params_df.to_csv(f"{directory}/params.csv")
    
    # general configrations
    config_d = vars(conf)
    config_df = pd.DataFrame(config_d, index=[0])
    config_df.to_csv(f"{directory}/configs.csv")

def final_results(dir):
    logs = np.load(dir)
    # print(f"tiemsteps: {logs['timesteps']}")
    # print(f"results: {logs['results']}")
    final_reward = np.mean(logs['results'][-1])
    rewards_std = np.std(logs['results'][-1])
    # print(f"reward: {final_reward}, std: {rewards_std}")
    return final_reward, rewards_std

def plot_results(results_dict, args, stamp, reward_threshold):
    plt.figure()
    for k in results_dict:
        results = results_dict[k]
        results = np.array(results)
        # example data
        print(results)
        x = results[:, 0]
        y = results[:, 1] / reward_threshold * 100
        yerr = results[:, 2] / reward_threshold * 100
        # First illustrate basic pyplot interface, using defaults where possible.
        plt.errorbar(x, y, yerr=yerr, label=k)
    plt.legend()
    plt.xlabel("Pretrain steps")
    plt.ylabel("Normalized Return")
    if args.env == "MountainCarContinuous-v0":
        plt.title(f"MountainCar")
    else:
        plt.title(f"{args.env[:args.env.index('-') ]}")
    files_dir = f"Vis/{args.env}"
    os.makedirs(files_dir, exist_ok=True)
    plt.savefig(f'./{files_dir}/pretrain_steps_exper_{args.skills}_skills_{stamp}.png', dpi=150)
    print("plotted and saved")
    # except:
    #     print("plotted and saved with exception")
    #     plt.savefig(f"./pretrain_exp_{args.env}_ppo.png", dpi=150)

    # plt.show()
    

if __name__ == "__main__":
    # timestamp = 1638611737.0454986
    print(f"Experiment timestamp: {timestamp}")
    args = cmd_args()
    pretrain_steps = [int(5e6), int(5e6)]
    n_samples = 100
    algs = ['sac', 'ppo']
    # algs = ['sac', 'ppo']
    experiment directory
    for alg, pretrain_step in zip(algs, pretrain_steps):
        # print("############HERE#############")
        exp_directory = conf.pretrain_steps_exper_dir + f"{args.env}/" + f"{alg}_{args.env}_skills:{params['n_skills']}_{timestamp}"
        # create a folder for the expirment
        os.makedirs(exp_directory)
        # change the pretraining timesteps param
        params['pretrain_steps'] = pretrain_step
        # save the exp parameters
        save_params(args, alg, exp_directory)
        # create a diyan object
        alg_params = ppo_hyperparams if alg == "ppo" else sac_hyperparams
        diayn = DIAYN(params, alg_params, discriminator_hyperparams, args.env, alg, exp_directory, seed=seed, conf=conf, timestamp=timestamp, checkpoints=True, args=args)
        # pretraining step
        diayn.pretrain()
            
    # print(f"steps after pretraining: {steps}")
    # input()
    results_dict = {}
    reward_threshold = gym.make(args.env).spec.reward_threshold
    print(f"reeward threshold is: {reward_threshold}")
    steps = range(pretrain_steps[0]//n_samples, pretrain_steps[0]+n_samples, pretrain_steps[0]//n_samples)
    print(f"steps: {list(steps)}")
    for alg in algs:
        results_dict[alg] = []
        for step in steps:
            exp_directory = conf.pretrain_steps_exper_dir + f"{args.env}/" + f"{alg}_{args.env}_skills:{params['n_skills']}_pretrain_steps:{step}_{timestamp}"
            reward, std = final_results(exp_directory + "/finetune_eval_results/evaluations.npz")
            results_dict[alg].append([step, reward, std])
        
    plot_results(results_dict, args, timestamp, reward_threshold)


