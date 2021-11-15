import argparse

from stable_baselines3.common.vec_env.dummy_vec_env import DummyVecEnv
from stable_baselines3 import PPO, SAC
from stable_baselines3.common.callbacks import  EvalCallback
from stable_baselines3.common.monitor import Monitor

from src.config import conf

import torch
import torch.nn as nn
from torch.utils.tensorboard import SummaryWriter

import numpy as np
import pandas as pd
import gym

import time
import os
import random

# set the seed
seed = 10
random.seed(seed)
np.random.seed(seed)
torch.random.manual_seed(seed)

train_timesteps = int(1e5)

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

# save a timestamp
timestamp = time.time()


# Extract arguments from terminal
def cmd_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--env', type=str, default="MountainCarContinuous-v0")
    parser.add_argument("--alg", type=str, default="ppo")
    args = parser.parse_args()
    return args



# save the hyperparameters
def save_params(args, directory):
    # save the hyperparams in a csv files
    alg_hyperparams = ppo_hyperparams if args.alg == "ppo" else sac_hyperparams
    
    alg_hyperparams_df = pd.DataFrame(alg_hyperparams, index=[0])
    alg_hyperparams_df.to_csv(f"{directory}/{args.alg}_hyperparams.csv")
    
    # general configrations
    config_d = vars(conf)
    config_df = pd.DataFrame(config_d, index=[0])
    config_df.to_csv(f"{directory}/configs.csv")


# Train the agent
def train(args, directory):
    if args.alg == "ppo":
        env = DummyVecEnv([
                    lambda: Monitor(gym.make(args.env),  directory),
                    lambda: Monitor(gym.make(args.env),  directory),
                    lambda: Monitor(gym.make(args.env),  directory),
                    lambda: Monitor(gym.make(args.env),  directory),])
        
        # create the model with the speicifed hyperparameters        
        model = PPO('MlpPolicy', env, verbose=1, 
                learning_rate = ppo_hyperparams['learning_rate'], 
                n_steps = ppo_hyperparams['n_steps'], 
                batch_size = ppo_hyperparams['batch_size'],
                n_epochs = ppo_hyperparams['n_epochs'],
                gamma = ppo_hyperparams['gamma'],
                gae_lambda = ppo_hyperparams['gae_lambda'],
                clip_range = ppo_hyperparams['clip_range'],
                policy_kwargs = dict(activation_fn=nn.Tanh,
                                net_arch=[dict(pi=[conf.layer_size_policy, conf.layer_size_policy], vf=[conf.layer_size_value, conf.layer_size_value])]),
                tensorboard_log = directory,
                seed = seed
                )

        eval_env =  gym.make(args.env)
        eval_env = Monitor(eval_env, f"{directory}/eval_results")
        eval_callback = EvalCallback(eval_env, best_model_save_path=directory,
                                    log_path=f"{directory}/eval_results", eval_freq=1000,
                                    deterministic=True, render=False)
        # train the agent
        model.learn(total_timesteps=train_timesteps, callback=[eval_callback], log_interval=1, tb_log_name="PPO baseline")
        
    elif args.alg == "sac":
        env = DummyVecEnv([lambda: Monitor(gym.make(args.env),  directory)])
        # create the model with the speicifed hyperparameters
        model = SAC('MlpPolicy', env, verbose=1, 
                    learning_rate = sac_hyperparams['learning_rate'], 
                    batch_size = sac_hyperparams['batch_size'],
                    gamma = sac_hyperparams['gamma'],
                    buffer_size = sac_hyperparams['buffer_size'],
                    tau = sac_hyperparams['tau'],
                    ent_coef = sac_hyperparams['ent_coef'],
                    gradient_steps = sac_hyperparams['gradient_steps'],
                    learning_starts = sac_hyperparams['learning_starts'],
                    policy_kwargs = dict(net_arch=dict(pi=[conf.layer_size_policy, conf.layer_size_policy], qf=[conf.layer_size_value, conf.layer_size_value])),
                    tensorboard_log = directory,
                    seed = seed
                    )
        eval_env =  gym.make(args.env)
        eval_env = Monitor(eval_env,  f"{directory}/eval_results")

        eval_callback = EvalCallback(eval_env, best_model_save_path=directory, log_path=f"{directory}/eval_results", eval_freq=1000, deterministic=True, render=False)

        # train the agent
        model.learn(total_timesteps=params['pretrain_steps'], callback=[eval_callback], log_interval=1, tb_log_name="SAC baseline")


if __name__ == "__main__":
    print(f"Experiment timestamp: {timestamp}")
    args = cmd_args()
    # experiment directory
    exp_directory = conf.log_dir_finetune + f"{args.alg}_{args.env}_{timestamp}"
    # create a folder for the expirment
    os.makedirs(exp_directory)
    # save the exp parameters
    save_params(args, exp_directory)
    # pretraining step
    train(args, exp_directory)
