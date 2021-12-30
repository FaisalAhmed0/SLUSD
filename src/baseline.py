import argparse

from stable_baselines3.common.vec_env.dummy_vec_env import DummyVecEnv
from stable_baselines3 import PPO, SAC
from stable_baselines3.common.callbacks import  EvalCallback
from stable_baselines3.common.monitor import Monitor

from src.config import conf
from src.utils import extract_results
from src.environment_wrappers.env_wrappers import EvalWrapper

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

# set the seed
seed = 10
random.seed(seed)
np.random.seed(seed)
torch.random.manual_seed(seed)

train_timesteps = int(1e5)

# algorithms
algs = ['sac']
envs = ['MountainCarContinuous-v0', 'Reacher-v2', 'Swimmer-v2', 'Hopper-v2', 'HalfCheetah-v2', 'Walker2d-v2', 'Ant-v2', 'Humanoid-v2']

# ppo hyperparams
ppo_hyperparams = dict(
    n_steps = 2048,
    learning_rate = 3e-4,
    n_epochs = 10,
    batch_size = 64,
    gamma = 0.99,
    gae_lambda = 0.95,
    clip_range = 0.1,
    ent_coef=0.5,
    n_actors = 8,
    algorithm = "ppo",
    
)

# SAC Hyperparameters 
sac_hyperparams = dict(
    learning_rate = 3e-4,
    gamma = 0.99,
    buffer_size = int(1e7),
    batch_size = 128,
    tau = 0.005,
    gradient_steps = 1,
    ent_coef=0.5,
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


# Train the agent
def train_rand_init_baseline(args, directory):
    alg_params = hyperparams[args.alg]
    if args.alg == "ppo":
        env = DummyVecEnv([lambda: Monitor(gym.make(args.env),  directory)]*alg_params['n_actors'])
        
        # create the model with the speicifed hyperparameters        
        model = PPO('MlpPolicy', env, verbose=1,
                        learning_rate=alg_params['learning_rate'],
                        n_steps=alg_params['n_steps'],
                        batch_size=alg_params['batch_size'],
                        n_epochs=alg_params['n_epochs'],
                        gamma=alg_params['gamma'],
                        gae_lambda=alg_params['gae_lambda'],
                        clip_range=alg_params['clip_range'],
                        policy_kwargs=dict(activation_fn=nn.Tanh,
                                           net_arch=[conf.layer_size_shared, conf.layer_size_shared,dict(pi=[conf.layer_size_policy], vf=[conf.layer_size_value])]),
                        tensorboard_log=directory,
                        seed=seed
                        )

        eval_env =  EvalWrapper(gym.make(args.env))
        eval_env = Monitor(eval_env, f"{directory}/eval_results")
        eval_callback = EvalCallback(eval_env, best_model_save_path=directory,
                                    log_path=f"{directory}/eval_results", eval_freq=5000,
                                    deterministic=True, render=False, n_eval_episodes=conf.eval_runs)
        # train the agent
        model.learn(total_timesteps=train_timesteps, callback=[eval_callback], log_interval=1, tb_log_name="PPO baseline")
        
    elif args.alg == "sac":
        env = DummyVecEnv([lambda: Monitor(gym.make(args.env),  directory)])
        # create the model with the speicifed hyperparameters
        model = SAC('MlpPolicy', env, verbose=1,
                        learning_rate=alg_params['learning_rate'],
                        batch_size=alg_params['batch_size'],
                        gamma=alg_params['gamma'],
                        buffer_size=alg_params['buffer_size'],
                        tau=alg_params['tau'],
                        ent_coef=alg_params['ent_coef'],
                        gradient_steps=alg_params['gradient_steps'],
                        learning_starts=alg_params['learning_starts'],
                        policy_kwargs=dict(net_arch=dict(pi=[2*conf.layer_size_policy, 2*conf.layer_size_policy], qf=[
                                           conf.layer_size_q, conf.layer_size_q])),
                        tensorboard_log=directory,
                        seed=seed
                        )
        eval_env =  EvalWrapper(gym.make(args.env))
        eval_env = Monitor(eval_env,  f"{directory}/eval_results")

        eval_callback = EvalCallback(eval_env, best_model_save_path=directory, log_path=f"{directory}/eval_results", eval_freq=5000, deterministic=True, render=False, n_eval_episodes=conf.eval_runs)

        # train the agent
        model.learn(total_timesteps=train_timesteps, callback=[eval_callback], log_interval=1, tb_log_name="SAC baseline")
        
        
def state_coverage(env_name, alg, timestamp):
    alg_params = hyperparams[alg]
    # load the model
    env = DummyVecEnv([lambda: gym.make(env_name)])
    # directory =  conf.log_dir_finetune + f"{alg}_{env_name}_{timestamp}" + "/best_model"
    directory = conf.log_dir_finetune +  f"{args.alg}_{args.env}_{args.btype}_baseline_{timestamp}" + "/best_model"
    if alg == "sac":
        model = SAC.load(directory)
    elif alg == "ppo":
        model = PPO.load(directory, clip_range= get_schedule_fn(alg_params['clip_range']))
    # run the model to collect the data
    # print(model)
    seeds = [0, 10, 1234, 5, 42]
    entropy_list = []
    for seed in seeds:
        data = []
        with torch.no_grad():
            for i in range(5):
                env = DummyVecEnv([lambda: gym.make(env_name) ])
                env.seed(seed)
                obs = env.reset()
                # obs = obs[0]
                data.append(obs[0].copy())
                total_reward = 0
                done = False
                while not done:
                    action, _ = model.predict(obs, deterministic=False)
                    # print(f"action: {action}")
                    obs, _, done, _ = env.step(action)
                    # obs = obs[0]
                    data.append(obs[0].copy())
                    # aug_obs = augment_obs(obs, skill, n_skills)
        data = np.array(data)
        np.random.shuffle(data)
        print(f"length of the data: {len(data)}")
        kde_entropy = continuous.get_h(data, k=30, min_dist=0.00001)
        entropy_list.append(kde_entropy)
    print(f"Average entropy over all the random seeds: { np.mean(entropy_list) } with std of { np.std(entropy_list) }")
    return np.mean(entropy_list), np.std(entropy_list)
        
        



if __name__ == "__main__":
    # TODO Test it
    args = cmd_args()
    run_all = args.run_all
    if run_all:
        for alg in algs:
            for env in envs:
                print(f"Experiment timestamp: {timestamp}")
                timestamp = time.time()
                exp_directory = conf.log_dir_finetune + f"{alg}_{env}_{args.btype}_baseline_{timestamp}"
                os.makedirs(exp_directory)
                # create a folder for the expirment
                # save the exp parameters
                save_params(args, exp_directory)
                # choose the baseline typle
                baseline_type = args.btype
                baselines = ["rand_init", "rand_act", "RND", "APT"]
                if baseline_type == baselines[0]:
                    train_rand_init_baseline(args, exp_directory)
                elif baseline_type == baselines[1]:
                    pass
                    # train_rand_act(args, directory)
                elif baseline_type == baselines[2]:
                    pass
                    # train_RND(args, directory)
                elif baseline_type == baselines[3]:
                    pass
                    # train_APT(args, directory)
                # save results
                r = {}
                r['baseline_type'] = baseline_type
                file_dir = conf.log_dir_finetune +  f"{alg}_{env}_{args.btype}_baseline_{timestamp}/" + "eval_results/" + "evaluations.npz"
                files = np.load(file_dir)
                data_r = files['results']
                reward_mean, reward_std = extract_results(data_r)
                entropy_mean, entropy_std = state_coverage(env, alg, timestamp)
                r['reward_mean'] = reward_mean
                r['reward_std'] = reward_std
                r['entropy_mean'] = entropy_mean
                r['entropy_std'] = entropy_std
                results_df = pd.DataFrame(r, index=[0])
                results_df.to_csv(f"{exp_directory}/results.csv")
                print(results_df)
                
    else:
        print(f"Experiment timestamp: {timestamp}")
        # experiment directory
        exp_directory = conf.log_dir_finetune + f"{args.alg}_{args.env}_{args.btype}_baseline_{timestamp}"
        # create a folder for the expirment
        os.makedirs(exp_directory)
        # save the exp parameters
        save_params(args, exp_directory)
        # choose the baseline typle
        baseline_type = args.btype
        baselines = ["rand_init", "rand_act", "RND", "APT"]
        if baseline_type == baselines[0]:
            train_rand_init_baseline(args, exp_directory)
        elif baseline_type == baselines[1]:
            pass
            # train_rand_act(args, directory)
        elif baseline_type == baselines[2]:
            pass
            # train_RND(args, directory)
        elif baseline_type == baselines[3]:
            pass
            # train_APT(args, directory)
        # save results
        r = {}
        r['baseline_type'] = baseline_type
        file_dir = conf.log_dir_finetune +  f"{args.alg}_{args.env}_{args.btype}_baseline_{timestamp}/" + "eval_results/" + "evaluations.npz"
        files = np.load(file_dir)
        data_r = files['results']
        reward_mean, reward_std = extract_results(data_r)
        entropy_mean, entropy_std = state_coverage(args.env, args.alg, timestamp)
        r['reward_mean'] = reward_mean
        r['reward_std'] = reward_std
        r['entropy_mean'] = entropy_mean
        r['entropy_std'] = entropy_std
        results_df = pd.DataFrame(r, index=[0])
        results_df.to_csv(f"{exp_directory}/results.csv")
        print(results_df)
