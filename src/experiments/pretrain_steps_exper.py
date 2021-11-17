import argparse

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



# pretraing steps
pretrain_steps = [ int(50e3), int(100e3), int(200e3), int(500e3), int(1e6), int(2e6), int(5e6) ]

# shared parameters
params = dict( n_skills = 4,
           pretrain_steps = int(1e5),
           finetune_steps = int(2e3),
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

# Discriminator Hyperparameters
discriminator_hyperparams = dict(
    learning_rate = 3e-4,
    batch_size = 64,
    n_epochs = 1
)

# save a timestamp
timestamp = time.time()


# Extract arguments from terminal
def cmd_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--env', type=str, default="MountainCarContinuous-v0")
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
    print(f"reeard: {final_reward}, std: {rewards_std}")
    return final_reward, rewards_std

def plot_results(results_dict, args, stamp):
    plt.figure()
    for k in results_dict:
        results = results_dict[k]
        results = np.array(results)
        # example data
        print(results)
        x = results[:, 0]
        y = results[:, 1]
        yerr = results[:, 2]

        # First illustrate basic pyplot interface, using defaults where possible.
        plt.errorbar(x, y, yerr=yerr, label=k)
    plt.legend()
    # files_dir = f"Vis/{args.env}"
    # plt.savefig(f'{files_dir}/pretrain_steps_exper_{args.alg}_{args.skills}_skills_{stamp}', dpi=150)
    # plt.savefig("")

    plt.show()
    

if __name__ == "__main__":
    print(f"Experiment timestamp: {timestamp}")
    args = cmd_args()
    results_dict = {}
    algs = ['ppo', 'sac']
    # experiment directory
    for alg in algs:
        results_dict[alg] = []
        for steps in pretrain_steps:
            # print("############HERE#############")
            exp_directory = conf.pretrain_steps_exper_dir + f"{alg}_{args.env}_skills:{params['n_skills']}_pretrain_steps:{steps}_{timestamp}"
            # create a folder for the expirment
            os.makedirs(exp_directory)
            # change the pretraining timesteps param
            params['pretrain_steps'] = steps
            # save the exp parameters
            save_params(args, alg, exp_directory)
            # create a diyan object
            alg_params = ppo_hyperparams if alg == "ppo" else sac_hyperparams
            diayn = DIAYN(params, alg_params, discriminator_hyperparams, args.env, alg, exp_directory, seed=seed, conf=conf, timestamp=timestamp)
            # pretraining step
            diayn.pretrain()
            # fine-tuning step 
            diayn.finetune()
            # save the results
            reward, std = final_results(exp_directory + "/finetune_eval_results/evaluations.npz")
            results_dict[alg].append([steps, reward, std])
    plot_results(results_dict, args, timestamp)


