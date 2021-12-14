
import argparse

from src.config import conf
from src.diayn import DIAYN # diayn with model-free RL
from src.diayn_es import DIAYN_ES # diayn with evolution star
from src.utils import report_resuts

import torch

import numpy as np
import pandas as pd

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
           pretrain_steps = int(2e6),
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
    mixup = False,
    parametrization = "CPC" # TODO: add this as a CMD argument
)

# save a timestamp
timestamp = time.time()


# Extract arguments from terminal
def cmd_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--env', type=str, default="MountainCarContinuous-v0")
    parser.add_argument("--alg", type=str, default="ppo")
    parser.add_argument("--skills", type=int, default=6)
    parser.add_argument("--presteps", type=int, default=int(1e6))
    args = parser.parse_args()
    return args

# save the hyperparameters
def save_params(args, directory):
    # save the hyperparams in a csv files
    alg_hyperparams = hyperparams[args.alg]
    
    alg_hyperparams_df = pd.DataFrame(alg_hyperparams, index=[0])
    alg_hyperparams_df.to_csv(f"{directory}/{args.alg}_hyperparams.csv")
    
    discriminator_hyperparams_df = pd.DataFrame(discriminator_hyperparams, index=[0])
    discriminator_hyperparams_df.to_csv(f"{directory}/discriminator_hyperparams.csv")
    
    # convert the config namespace to a dictionary
    exp_params_df = pd.DataFrame(params, index=[0])
    exp_params_df.to_csv(f"{directory}/params.csv")
    
    # general configrations
    config_d = vars(conf)
    config_df = pd.DataFrame(config_d, index=[0])
    config_df.to_csv(f"{directory}/configs.csv")
    print(f"{args.alg} hyperparameters: {alg_hyperparams}" )
    print(f"discriminator hyperparameters: {discriminator_hyperparams}" )
    print(f"shared experiment parameters: {params}" )
    print(f"configurations: {config_d }" )
    input()

def run_diyan(args):
    if args.alg != 'es':
        alg_params = ppo_hyperparams if args.alg == "ppo" else sac_hyperparams
        diayn = DIAYN(params, alg_params, discriminator_hyperparams, args.env, args.alg, exp_directory, seed=seed, conf=conf, timestamp=timestamp)
        # pretraining step
        diayn.pretrain()
        # fine-tuning step 
        diayn.finetune()
    elif args.alg == "es":
        diayn_es = DIAYN_ES(params, alg_params, discriminator_hyperparams, args.env, exp_directory, seed=seed, conf=conf, timestamp=timestamp, args=args)
        # pretraining step
        diayn_es.pretrain()
        # fine-tuning step 
        diayn_es.finetune()
        

        

if __name__ == "__main__":
    print(f"Experiment timestamp: {timestamp}")
    args = cmd_args()
    params['pretrain_steps'] = args.presteps
    params['n_skills'] = args.skills
    # experiment directory
    exp_directory = conf.log_dir_finetune + f"{args.alg}_{args.env}_skills:{params['n_skills']}_{timestamp}"
    # create a folder for the expirment
    os.makedirs(exp_directory)
    # save the exp parameters
    save_params(args, exp_directory)
    # create a diyan object
    alg_params = ppo_hyperparams if args.alg == "ppo" else sac_hyperparams
    diayn = DIAYN(params, alg_params, discriminator_hyperparams, args.env, args.alg, exp_directory, seed=seed, conf=conf, timestamp=timestamp)
    # pretraining step
    model = diayn.pretrain()
    # fine-tuning step 
    diayn.finetune()
    # save the final results
    # extract data_r for finetuning results and data_i for the pretrained policy
    file_dir_finetune = conf.log_dir_finetune +  f"{args.alg}_{args.env}_skills:{args.skills}_{timestamp}/" + "finetune_eval_results/" + "evaluations.npz"
    files = np.load(file_dir_finetune)
    data_r = files['results']
    file_dir_skills = conf.log_dir_finetune +  f"{args.alg}_{args.env}_skills:{args.skills}_{timestamp}/" + "eval_results/" + "evaluations.npz"
    files = np.load(file_dir_skills)
    data_i = files['results']
    df = report_resuts(args.env, args.alg, params['n_skills'], model, data_r, data_i, timestamp, exp_directory)
    print(df)



    

