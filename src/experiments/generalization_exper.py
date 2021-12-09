import gym
import argparse
import os

from src.config import conf
from src.diayn import DIAYN
from src.diayn_es import DIAYN_ES
from src.environment_wrappers import tasks_wrappers

import torch

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import time
import os
import random

#Note: this is still in testing I will only this with halfcheeta


# set the seed
seed = 10
random.seed(seed)
np.random.seed(seed)
torch.random.manual_seed(seed)

# Algorithms to compare 
algs = ['ppo', 'sac']
# algs = ['ppo', 'sac', 'es']

# Tasks
tasks = ['run_forward', 'run_backward' ,'jump']

# Environments
envs = ['HalfCheetah-v2']
# envs = ['HalfCheetah-v2', 'Walker2d-v2', 'Ant-v2']


# diayn params for each environment
# original values
'''
diayn_params = {
    'ppo':{
        'HalfCheetah-v2': dict( 
           pretrain_steps = int(50e6),
           finetune_steps = int(1e5),
             ),
    },
    'sac':{
        'HalfCheetah-v2': dict( 
           pretrain_steps = int(2e6),
           finetune_steps = int(1e5),
             ),
    }
    
}
'''
# dummy testing values
diayn_params = {
    'ppo':{
        'HalfCheetah-v2': dict( 
           pretrain_steps = int(1e6),
             ),
    },
    'sac':{
        'HalfCheetah-v2': dict( 
           pretrain_steps = int(200e3),
             ),
    }
    
}

# shared parameters
params = dict( n_skills = 6,
           buffer_size = int(1e7),
           min_train_size = int(1e4),
           finetune_steps = int(1e5),
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
    ent_coef=0.4,
    learning_starts = 10000,
    algorithm = "sac"
)

# Evolution Stratigies Hyperparameters 
es_hyperparams = dict(
    lr = 1e-3, # learning rate 
    iterations = 2500, # iterations 
    pop_size = 80, # population size
    algorithm = "es"
)

hyperparams = {
    'ppo': ppo_hyperparams,
    'sac': sac_hyperparams,
    'es' : es_hyperparams
}

# Discriminator Hyperparameters
# weight_decay=0.0, label_smoothing=None, gp=None, mixup=False
discriminator_hyperparams = dict(
    learning_rate = 3e-4,
    batch_size = 64,
    n_epochs = 1,
    weight_decay = 0, 
    dropout = None, # The dropout probability
    label_smoothing = False,
    gp = None, # the weight of the gradient penalty term
    mixup = False
)


# save a timestamp
timestamp = time.time()

# save the hyperparameters
def save_params(alg, directory):
    # save the hyperparams in a csv files
    alg_hyperparams = hyperparams[alg]
    
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
    print(f"{alg} hyperparameters: {alg_hyperparams}" )
    print(f"discriminator hyperparameters: {discriminator_hyperparams}" )
    print(f"shared experiment parameters: {params}" )
    print(f"configurations: {config_d }" )
    # input()

# run a diayn experiment
def run_diyan(alg, task, env, exp_directory, seed):
    alg_params = hyperparams[alg]
    if alg != 'es':
        diayn = DIAYN(params, alg_params, discriminator_hyperparams, env, alg, exp_directory, seed=seed, conf=conf, timestamp=timestamp, task=task)
        # pretraining step
        diayn.pretrain()
        # fine-tuning step 
        diayn.finetune()
    elif alg == "es":
        diayn_es = DIAYN_ES(params, alg_params, discriminator_hyperparams, env, exp_directory, seed=seed, conf=conf, timestamp=timestamp, args=None)
        # pretraining step
        diayn_es.pretrain()
        # fine-tuning step 
        diayn_es.finetune()
        
# report the result after each diayn experiment
def final_results(file_dir):
    files = np.load(file_dir)
    results = files['results']
    data_mean = results.mean(axis=1)
    data_mean = np.convolve(data_mean, np.ones(10)/10, mode='valid') 
    data_std = results.std(axis=1)
    data_std = np.convolve(data_std, np.ones(10)/10, mode='valid') 
    # print(f"tiemsteps: {logs['timesteps']}")
    # print(f"results: {logs['results']}")
    final_reward = data_mean[-1]
    rewards_std = data_std[-1]
    print(f"reward: {final_reward}, std: {rewards_std}")
    return final_reward, rewards_std


if __name__ == "__main__":
    print(f"Experiment timestamp: {timestamp}")
    # TODO: define how you will save your results
    results = {}
    results['tasks'] = tasks
    # main expirment directory
    main_dir = conf.generalization_exper_dir + str(timestamp)
    os.makedirs(main_dir)
    for alg in algs:
        for env in envs:
            params['pretrain_steps'] = diayn_params[alg][env]['pretrain_steps']
            if not (env in results):
                results[env] = []
            tasks_rewards = []
            for task in tasks:
                # directory per algorithm, env and task 
                exp_directory = f"{main_dir}/{alg}_{env}_skills:{params['n_skills']}_{task}"
                os.makedirs(exp_directory)
                # save the exp parameters
                save_params(alg, exp_directory)
                # run a diayn experiment
                run_diyan(alg, task, env, exp_directory, seed)
                # report the results
                results_dir = exp_directory + "/finetune_eval_results/evaluations.npz"
                final_reward, rewards_std = final_results(results_dir)
                tasks_rewards.append(final_reward)
            results[env].append(tasks_rewards)
    print(f"final results: {results}")
                