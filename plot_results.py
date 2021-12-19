'''
This script plot the results of a finetune experiment given the environment, list of algorithms, timestamps, labels (optional), and the number of skills.
'''
import argparse
import ast

import numpy as np
import matplotlib.pyplot as plt

import os
from src.config import conf

import gym



# cmd arguments
def cmd_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--env', type=str, default="MountainCarContinuous-v0")
    parser.add_argument("--algs", nargs="*", type=str, default=["ppo"]) # pass the algorithms that correspond to each timestamp
    parser.add_argument("--skills", type=int, default=4)
    parser.add_argument("--stamps", nargs="*", type=str, required=True) # pass the timestamps as a list
    parser.add_argument("--colors", nargs="*", type=str, required=False, default=['b', 'r']) # pass the timestamps as a list
    parser.add_argument("--labels", nargs="*", type=str, required=False, default=None) # add custom labels 
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
    # file_dir = conf.log_dir_finetune + f"{args.alg}_{args.env}_{args.stamp}/" + "eval_results/" + "evaluations.npz"
    # file_dir = conf.log_dir_finetune + f"{args.alg}_{args.env}_{args.stamp}/" + "evaluations.npz"
    plt.figure()
    plt.title(f"{args.env[: args.env.index('-')]}")
    print("######## Skills evaluation ########")
    for i in range(len(algs)):
        alg = algs[i]
        print(f"Algorithm: {alg}")
        plot_color = colors[i]
        stamp = stamps[i]
        # legend = alg
        file_dir_skills = conf.log_dir_finetune +  f"{alg}_{args.env}_skills:{args.skills}_{stamp}/" + "eval_results/" + "evaluations.npz"
        files = np.load(file_dir_skills)
        steps = files['timesteps']
        results = files['results']
        data_mean = results.mean(axis=1)
        # print(len(data_mean))
        data_mean = np.convolve(data_mean, np.ones(10)/10, mode='valid') 
        data_std = results.std(axis=1)
        data_std = np.convolve(data_std, np.ones(10)/10, mode='valid') 
        print(f"mean: {data_mean[-1]}")
        print(f"std: {data_std[-1]}")
        steps = np.convolve(steps, np.ones(10)/10, mode='valid') 
        if len(algs) > 1:
            plt.plot(steps, data_mean, label=legends[i], color=plot_color)
            plt.legend()
        else:
            plt.plot(steps, data_mean, color=plot_color)
        plt.fill_between(steps, (data_mean-data_std), (data_mean+data_std), color=plot_color, alpha=0.3)
        plt.xlabel(xlabel)
        plt.ylabel(ylabel)
        plt.grid(False)
        if len(algs) == 1:
            filename = f"{args.env}_pretraining_{alg}"
            files_dir = f"Vis/{args.env}"
            os.makedirs(files_dir, exist_ok=True)
            plt.savefig(f'{files_dir}/{filename}', dpi=150)    
    if len(algs) > 1:
        filename = f"{args.env}_pretraining_all_algs"
        files_dir = f"Vis/{args.env}"
        os.makedirs(files_dir, exist_ok=True)
        plt.savefig(f'{files_dir}/{filename}', dpi=150)    
    
    # loop throgh the finetuning results
    plt.figure()
    plt.title(f"{args.env[: args.env.index('-')]}")
    print("######## Finetine evaluation ########")
    for i in range(len(algs)):
        alg = algs[i]
        print(f"Algorithm: {alg}")
        plot_color = colors[i]
        stamp = stamps[i]
        legend = alg
        file_dir_finetune = conf.log_dir_finetune +  f"{alg}_{args.env}_skills:{args.skills}_{stamp}/" + "finetune_eval_results/" + "evaluations.npz"
        files = np.load(file_dir_finetune)
        steps = files['timesteps']
        results = files['results']
        data_mean = results.mean(axis=1)
        # print(len(data_mean))
        data_mean = np.convolve(data_mean, np.ones(10)/10, mode='valid') 
        data_std = results.std(axis=1)
        data_std = np.convolve(data_std, np.ones(10)/10, mode='valid') 
        print(f"mean: {data_mean[-1]}")
        print(f"std: {data_std[-1]}")
        steps = np.convolve(steps, np.ones(10)/10, mode='valid') 
        # normalized (percentage of the reard threshold)
        data_mean = data_mean / gym.make(args.env).spec.reward_threshold
        data_std = data_std / gym.make(args.env).spec.reward_threshold
        # normalize
        # print(len(x))
        if len(algs) > 1:
            plt.plot(steps, data_mean, label=legends[i], color=plot_color)
            plt.legend()
        else:
            plt.plot(steps, data_mean, color=plot_color)
        plt.fill_between(steps, (data_mean-data_std), (data_mean+data_std), color=plot_color, alpha=0.3)
        plt.xlabel(xlabel)
        plt.ylabel(ylabel)
        plt.grid(False)
        if len(algs) == 1:
            filename = f"{args.env}_finetune_{alg}"
            files_dir = f"Vis/{args.env}"
            os.makedirs(files_dir, exist_ok=True)
            plt.savefig(f'{files_dir}/{filename}', dpi=150)    
    if len(algs) > 1:
        filename = f"{args.env}_finetune_all_algs"
        files_dir = f"Vis/{args.env}"
        os.makedirs(files_dir, exist_ok=True)
        plt.savefig(f'{files_dir}/{filename}', dpi=150)    
        
        
        
        
