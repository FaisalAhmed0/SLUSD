import argparse

from stable_baselines3.common.vec_env.dummy_vec_env import DummyVecEnv
from stable_baselines3 import PPO, SAC
from stable_baselines3.common.callbacks import  EvalCallback
from stable_baselines3.common.monitor import Monitor


from src.config import conf
from src.utils import extract_results
from src.environment_wrappers.env_wrappers import EvalWrapper
from src.utils import seed_everything


import torch
import torch.nn as nn
from torch.utils.tensorboard import SummaryWriter

import numpy as np
import pandas as pd
import gym

import time
import os
import random

from entropy_estimators import continuous


train_timesteps = int(1e5)

# algorithms
algs = ['sac']
envs = ['MountainCarContinuous-v0', "HalfCheetah-v2", "Walker2d-v2", "Ant-v2"]


# SAC Hyperparameters 
sac_hyperparams = dict(
    learning_rate = 3e-4,
    gamma = 0.99,
    buffer_size = int(1e6),
    batch_size = 256,
    tau = 0.005,
    gradient_steps = 1,
    ent_coef='auto',
    learning_starts = 100,
    algorithm = "sac"
)

# save a timestamp
timestamp = time.time()


# Extract arguments from terminal
def cmd_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--env', type=str, default="MountainCarContinuous-v0")
    parser.add_argument("--alg", type=str, default="ppo") # ppo, sac, es
    parser.add_argument("--btype", type=str, default="rand_init") # rand_init, rand_act, RND, APT
    parser.add_argument("--run_all", type=bool, default=False) # run all on environments
    args = parser.parse_args()
    return args



# save the hyperparameters
def save_params(args, directory):
    # save the hyperparams in a csv files
    alg_hyperparams = hyperparams[args.alg]
    
    alg_hyperparams_df = pd.DataFrame(alg_hyperparams, index=[0])
    alg_hyperparams_df.to_csv(f"{directory}/{args.alg}_hyperparams.csv")
    print(f"Hyperparameters")
    print(alg_hyperparams_df)
    
    # general configrations
    config_d = vars(conf)
    config_df = pd.DataFrame(config_d, index=[0])
    config_df.to_csv(f"{directory}/configs.csv")
    print("Configs")
    print(config_df)
    # input()

@torch.no_grad()
def evaluate_policy(env_name, model):
    print("Evaluating Policy")
    env = gym.make(env_name)
    done = False
    obs = env.reset()
    total_reward = 0
    while not done:
        # if alg in ["ppo", "sac"]:
        action, _ = model.predict(obs, deterministic=True)
        obs, reward, done, info = env.step(action)
        total_reward += reward
    print(f"Total reward: {total_reward}")
    return total_reward


# Train the agent
# (env_params, results_df_list, plots_d_list, seed , time, params_copy=deepcopy(params))
def train_rand_init_baseline(env_name, seed, seed_directory):
    seed_everything(seed)
    env = DummyVecEnv([lambda: Monitor(gym.make(env_name),  seed_directory)])
    # create the model with the speicifed hyperparameters
    model = SAC('MlpPolicy', env, verbose=1,
                learning_rate=sac_hyperparams['learning_rate'],
                batch_size=sac_hyperparams['batch_size'],
                gamma=sac_hyperparams['gamma'],
                buffer_size=sac_hyperparams['buffer_size'],
                tau=sac_hyperparams['tau'],
                ent_coef=sac_hyperparams['ent_coef'],
                gradient_steps=sac_hyperparams['gradient_steps'],
                learning_starts=sac_hyperparams['learning_starts'],
                policy_kwargs=dict(net_arch=dict(pi=[conf.layer_size_policy, conf.layer_size_policy], qf=[
                                   conf.layer_size_q, conf.layer_size_q]), n_critics=2),
                tensorboard_log=seed_directory,
                seed=seed
                )
    eval_env =  EvalWrapper(gym.make(env_name))
    eval_env = Monitor(eval_env,  f"{seed_directory}/eval_results")

    eval_callback = EvalCallback(eval_env, best_model_save_path=seed_directory, log_path=f"{seed_directory}/eval_results", eval_freq=5000, deterministic=True, render=False, n_eval_episodes=conf.eval_runs)

    # train the agent
    model.learn(total_timesteps=train_timesteps, callback=[eval_callback], log_interval=1, tb_log_name="SAC baseline")
    # total reward
    total_reward = np.mean([ evaluate_policy(env_name, model) for _ in range(10)])
    np.savetxt(f"{seed_directory}/total_reward.txt", np.array([total_reward]))
    return total_reward
    
    
        
        

if __name__ == "__main__":
    # TODO Test it
    main_dir = "random_init_baselines/"
    for alg in algs:
        for env in envs:
            print(f"Experiment timestamp: {timestamp}")
            timestamp = time.time()
            env_dir = "random_init_baselines/" + f"{env}"
            # choose the baseline typle
            seed_results = []
            for seed in conf.seeds:
                seed_directory = f"{env_dir}/seed:{seed}"
                os.makedirs(seed_directory, exist_ok=True)
                seed_results.append(train_rand_init_baseline(env, seed, seed_directory))
            mean = np.mean(seed_results)
            std = np.std(seed_results)
            np.savetxt(f"{env_dir}/total_reward_mean.txt", np.array([mean]))
            np.savetxt(f"{env_dir}/total_reward_std.txt", np.array([std]))
#                     file_dir = seed + "eval_results/" + "evaluations.npz"
#                     files = np.load(file_dir)
#                     data_r = files['results']
#                 reward_mean, reward_std = extract_results(data_r)
#                 entropy_mean, entropy_std = state_coverage(env, alg, timestamp)
#                 r['reward_mean'] = reward_mean
#                 r['reward_std'] = reward_std
#                 r['entropy_mean'] = entropy_mean
#                 r['entropy_std'] = entropy_std
#                 results_df = pd.DataFrame(r, index=[0])
#                 results_df.to_csv(f"{exp_directory}/results.csv")
#                 print(results_df)
                
#     else:
#         print(f"Experiment timestamp: {timestamp}")
#         # experiment directory
#         exp_directory = conf.log_dir_finetune + f"{args.alg}_{args.env}_{args.btype}_baseline_{timestamp}"
#         # create a folder for the expirment
#         os.makedirs(exp_directory)
#         # save the exp parameters
#         save_params(args, exp_directory)
#         # choose the baseline typle
#         baseline_type = args.btype
#         baselines = ["rand_init", "rand_act", "RND", "APT"]
#         if baseline_type == baselines[0]:
#             train_rand_init_baseline(args, exp_directory)
#         elif baseline_type == baselines[1]:
#             pass
#             # train_rand_act(args, directory)
#         elif baseline_type == baselines[2]:
#             pass
#             # train_RND(args, directory)
#         elif baseline_type == baselines[3]:
#             pass
#             # train_APT(args, directory)
#         # save results
#         r = {}
#         r['baseline_type'] = baseline_type
#         file_dir = conf.log_dir_finetune +  f"{args.alg}_{args.env}_{args.btype}_baseline_{timestamp}/" + "eval_results/" + "evaluations.npz"
#         files = np.load(file_dir)
#         data_r = files['results']
#         reward_mean, reward_std = extract_results(data_r)
#         entropy_mean, entropy_std = state_coverage(args.env, args.alg, timestamp)
#         r['reward_mean'] = reward_mean
#         r['reward_std'] = reward_std
#         r['entropy_mean'] = entropy_mean
#         r['entropy_std'] = entropy_std
#         results_df = pd.DataFrame(r, index=[0])
#         results_df.to_csv(f"{exp_directory}/results.csv")
#         print(results_df)
