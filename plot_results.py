'''
This script plot the results of a finetune experiment given the environment, list of algorithms, timestamps, labels (optional), and the number of skills.
'''
import argparse
import ast

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

sns.set_theme()

import os
from src.config import conf
from src.utils import seed_everything

import gym


seeds = [0, 10 ]#, 1234, 5]#, 42]
# cmd arguments
def cmd_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--env', type=str, default="MountainCarContinuous-v0")
    parser.add_argument("--algs", nargs="*", type=str, default=["ppo"]) # pass the algorithms that correspond to each timestamp
    parser.add_argument("--skills", nargs="*", type=int, default=[6])
    parser.add_argument("--stamps", nargs="*", type=str, required=True) # pass the timestamps as a list
    parser.add_argument("--colors", nargs="*", type=str, required=False, default=['b', 'r']) # pass the timestamps as a list
    parser.add_argument("--labels", nargs="*", type=str, required=False, default=None) # add custom labels 
    parser.add_argument("--cls", nargs="*", type=str, required=True) # pass the timestamps as a list
    parser.add_argument("--lbs", nargs="*", type=str, required=True) # pass the timestamps as a list
    parser.add_argument("--suffix", type=str, required=False, default=None) # add custom labels 
    args = parser.parse_args()
    return args

if __name__ == "__main__":
    args = cmd_args()
    algs = args.algs
    stamps = args.stamps
    colors = args.colors
    xlabel = "Timesteps"
    ylabel = "Reward"
    legends = args.labels if args.labels else algs
    env_name = args.env
    # file_dir = conf.log_dir_finetune + f"{args.alg}_{args.env}_{args.stamp}/" + "eval_results/" + "evaluations.npz"
    # file_dir = conf.log_dir_finetune + f"{args.alg}_{args.env}_{args.stamp}/" + "evaluations.npz"
    print("######## Skills evaluation ########")
    plt.figure()
    plt.title(f"{env_name[: env_name.index('-')]}")
    for i in range(len(algs)):
        skills = args.skills[i]
        pm = args.cls[i]
        lb = args.lbs[i]
        alg = algs[i]
        print(f"Algorithm: {alg}")
        plot_color = colors[i]
        stamp = stamps[i]
        # legend = alg
        main_exper_dir = conf.log_dir_finetune + f"cls:{pm}, lb:{lb}/"
        env_dir = main_exper_dir + f"env: {env_name}, alg:{alg}, stamp:{stamp}/"
        seeds_list = []
        for seed in seeds:
            seed_everything(seed)
            seed_dir = env_dir + f"seed:{seed}/"
            file_dir_skills = seed_dir + "eval_results/" + "evaluations.npz"
            files = np.load(file_dir_skills)
            steps = files['timesteps']
            results = files['results']
            print(f"results.shape: {results.shape}")
            seeds_list.append(results.mean(axis=1).reshape(-1))
            # data_mean = results.mean(axis=1)
            # data_std = results.std(axis=1)
            # print(f"mean: {data_mean[-1]}")
            # print(f"std: {data_std[-1]}")
        data_mean = np.mean(seeds_list, axis=0)
        print(f"seeds list shape {np.array(seeds_list).shape}")
        # input()
        data_std = np.std(seeds_list, axis=0)
        # data_mean = np.convolve(data_mean, np.ones(10)/10, mode='valid') 
        # data_std = np.convolve(data_std, np.ones(10)/10, mode='valid') 
        # steps = np.convolve(steps, np.ones(10)/10, mode='valid') 
        print(f"data shape: {data_mean.shape}")
        print(f"steps shape: {steps.shape}")
        # input()
        if len(algs) > 1:
            plt.plot(steps, data_mean, label=legends[i], color=plot_color)
            plt.legend()
        else:
            plt.plot(steps, data_mean, color=plot_color)
        plt.fill_between(steps, (data_mean-data_std), (data_mean+data_std), color=plot_color, alpha=0.3)
        plt.xlabel(xlabel)
        plt.ylabel(ylabel)
        # plt.grid(False)
        if len(algs) == 1:
            filename = f"{args.env}_pretraining_{alg}"
            files_dir = f"Vis/{args.env}"
            os.makedirs(files_dir, exist_ok=True)
            plt.savefig(f'{files_dir}/{filename}', dpi=150)    
    if len(algs) > 1:
        filename = f"{args.env}_pretraining_all_algs {args.suffix}"
        files_dir = f"Vis/{args.env}_cls:{pm}_lb:{lb}"
        os.makedirs(files_dir, exist_ok=True)
        plt.savefig(f'{files_dir}/{filename}', dpi=150)    
    
    # loop throgh the finetuning results
    # loop throgh the finetuning results
    plt.figure()
    plt.title(f"{args.env[: args.env.index('-')]}")
    print("######## Finetine evaluation ########")
    seeds_list = []
    for i in range(len(algs)):
        skills = args.skills[i]
        pm = args.cls[i]
        lb = args.lbs[i]
        alg = algs[i]
        print(f"Algorithm: {alg}")
        plot_color = colors[i]
        stamp = stamps[i]
        legend = alg
        main_exper_dir = conf.log_dir_finetune + f"cls:{pm}, lb:{lb}/"
        env_dir = main_exper_dir + f"env: {env_name}, alg:{alg}, stamp:{stamp}/"
        seed_list = []
        for seed in seeds:
            seed_everything(seed)
            seed_dir = env_dir + f"seed:{seed}/"
            file_dir_finetune = seed_dir + "finetune_eval_results/" + "evaluations.npz"
            files = np.load(file_dir_finetune)
            steps = files['timesteps']
            results = files['results']
            seeds_list.append(results.mean(axis=1).reshape(-1))
        data_mean = np.mean(seeds_list, axis=0)
        data_std = np.std(seeds_list, axis=0)
        if len(algs) > 1:
            plt.plot(steps, data_mean, label=legends[i], color=plot_color)
            plt.legend()
        else:
            plt.plot(steps, data_mean, color=plot_color)
        plt.fill_between(steps, (data_mean-data_std), (data_mean+data_std), color=plot_color, alpha=0.3)
        plt.xlabel(xlabel)
        plt.ylabel(ylabel)
        # plt.grid(False)
        if len(algs) == 1:
            filename = f"{args.env}_finetune_{alg} {args.suffix}"
            files_dir = f"Vis/{args.env}_cls:{pm}_lb:{lb}"
            os.makedirs(files_dir, exist_ok=True)
            plt.savefig(f'{files_dir}/{filename}', dpi=150)    
    if len(algs) > 1:
        filename = f"{args.env}_finetune_all_algs"
        files_dir = f"Vis/{args.env}"
        os.makedirs(files_dir, exist_ok=True)
        plt.savefig(f'{files_dir}/{filename}', dpi=150)            
        
        
        
