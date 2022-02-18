'''
This script plot the results of a finetune experiment given the environment, list of algorithms, timestamps, labels (optional), and the number of skills.
'''
import argparse
import ast
import math

from stable_baselines3.common.vec_env.dummy_vec_env import DummyVecEnv
from stable_baselines3 import PPO, SAC
from stable_baselines3.common.utils import get_schedule_fn

import os
from src.config import conf
from src.utils import seed_everything, evaluate_pretrained_policy_ext, evaluate_adapted_policy, evaluate_state_coverage, evaluate_pretrained_policy_intr, evaluate
from src.environment_wrappers.env_wrappers import SkillWrapperFinetune, RewardWrapper, SkillWrapper, TimestepsWrapper
from src.models.models import Discriminator

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
sns.set()
sns.set_style('whitegrid')
sns.set_context("paper", font_scale = conf.font_scale)

import torch
import torch.nn as nn

import gym


# Asymptotic performance of the expert
asymp_perofrmance = {
    'HalfCheetah-v2': 8070.205213363435,
    'Walker2d-v2': 5966.481684037183,
    'Ant-v2': 6898.620003217143,
    "MountainCarContinuous-v0": 95.0

    }

random_performance = {
    'HalfCheetah-v2': -0.37117783,
    'Walker2d-v2': -0.48223244,
    'Ant-v2': 0.39107815,
    "MountainCarContinuous-v0": -0.04685473, 
}
# across all seeds
random_init = [
    {"Environment": 'HalfCheetah', "Normalized Extrinsic Return After Adaptation (%)": 3.977155901244182132e+03/asymp_perofrmance['HalfCheetah-v2']*100, "Algorithm": "Random init"},
    {"Environment": 'HalfCheetah', "Normalized Extrinsic Return After Adaptation (%)": 3.661330217032705150e+03/asymp_perofrmance['HalfCheetah-v2']*100, "Algorithm": "Random init"},
    {"Environment": 'HalfCheetah', "Normalized Extrinsic Return After Adaptation (%)": 3.777719310452177979e+03/asymp_perofrmance['HalfCheetah-v2']*100, "Algorithm": "Random init"},
    
    {"Environment": 'Walker2d', "Normalized Extrinsic Return After Adaptation (%)": 4.950961607268399121e+02/asymp_perofrmance['Walker2d-v2']*100, "Algorithm": "Random init"},
    {"Environment": 'Walker2d', "Normalized Extrinsic Return After Adaptation (%)": 5.050660942998074461e+02/asymp_perofrmance['Walker2d-v2']*100, "Algorithm": "Random init"},
    {"Environment": 'Walker2d', "Normalized Extrinsic Return After Adaptation (%)": 2.915505078088318669e+02/asymp_perofrmance['Walker2d-v2']*100, "Algorithm": "Random init"},
    
    {"Environment": 'Ant', "Normalized Extrinsic Return After Adaptation (%)": 4.935468207680875139e+02/asymp_perofrmance['Ant-v2']*100, "Algorithm": "Random init"},
    {"Environment": 'Ant', "Normalized Extrinsic Return After Adaptation (%)": 2.522865492915514665e+02/asymp_perofrmance['Ant-v2']*100, "Algorithm": "Random init"},
    {"Environment": 'Ant', "Normalized Extrinsic Return After Adaptation (%)": 7.162829405419695377e+02/asymp_perofrmance['Ant-v2']*100, "Algorithm": "Random init"},
    
    {"Environment": 'MountainCarContinuous', "Normalized Extrinsic Return After Adaptation (%)": -1.697574145149261080e-04/asymp_perofrmance['MountainCarContinuous-v0']*100, "Algorithm": "Random init"},
    {"Environment": 'MountainCarContinuous', "Normalized Extrinsic Return After Adaptation (%)": -3.889756080078754464e-04/asymp_perofrmance['MountainCarContinuous-v0']*100, "Algorithm": "Random init"},
    {"Environment": 'MountainCarContinuous', "Normalized Extrinsic Return After Adaptation (%)": -3.093423134238547888e-04/asymp_perofrmance['MountainCarContinuous-v0']*100, "Algorithm": "Random init"},
    
]


seeds = [0, 10, 42 ]#, 1234, 5]#, 42]
columns = ['Algorithm', "Environment", "Normalized Extrinsic Return After Adaptation (%)", "State Entropy", "Intrinsic Return" , "Normalized Extrinsic Return Before Adaptation (%)"]
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
    parser.add_argument("--suffix", type=str, required=False, default="") # add custom labels 
    args = parser.parse_args()
    return args

def extract_results(env_name, n_skills, pm, alg, stamp, lb):
    seeds_list = []
    asym = asymp_perofrmance[env_name]
    main_exper_dir = conf.log_dir_finetune + f"cls:{pm}, lb:{lb}/"
    env_dir = main_exper_dir + f"env: {env_name}, alg:{alg}, stamp:{stamp}/"
    
    params = pd.read_csv(env_dir + "params.csv").to_dict()
    discriminator_hyperparams_df = pd.read_csv(env_dir + "discriminator_hyperparams.csv")
    # print(discriminator_hyperparams_df.replace(math.nan, 12))
    discriminator_hyperparams = discriminator_hyperparams_df.to_dict('records')[0]
    # print(discriminator_hyperparams)
    state_dim = gym.make(env_name).observation_space.shape[0]
    hidden_dims = conf.num_layers_discriminator*[conf.layer_size_discriminator]
    temperature = discriminator_hyperparams['temperature']
    dropout = discriminator_hyperparams['dropout'] if 0 < discriminator_hyperparams['dropout'] < 1 else None
    # print(f"dropout type: {type(dropout)}")
    # print(dropout)
    # input()
    results_list = []
    # (self, env, discriminator, n_skills, parametrization="MLP", batch_size=128)
    for seed in seeds:
        seed_dir = env_dir + f"seed:{seed}/"
        seed_everything(seed)
        discriminator = Discriminator(state_dim, hidden_dims, n_skills, dropout=dropout).to(conf.device)
        # print(discriminator)
        # print(torch.load(seed_dir + "disc.pth"))
        discriminator.load_state_dict(torch.load(seed_dir + "disc.pth"))
        
        env = DummyVecEnv([lambda: RewardWrapper(SkillWrapper(gym.make(env_name), n_skills, max_steps=1000), discriminator, n_skills, pm)])
        if alg == "sac":
            model_dir = seed_dir + "/best_model"
            pretrained_policy = SAC.load(model_dir, env=env, seed=seed)
            best_skill = int([ d[d.index(":")+1:] for d in  os.listdir(seed_dir) if "best_finetuned_model_skillIndex:" in d][0])
            print(best_skill)
            finetuned_model_dir =  seed_dir + f"best_finetuned_model_skillIndex:{best_skill}/" + "/best_model"
            adapted_policy = SAC.load(finetuned_model_dir, env=env, seed=seed)
        elif alg == "ppo":
            model_dir = seed_dir + "/best_model"
            hyper_params = pd.read_csv(env_dir + "ppo_hyperparams.csv").to_dict('records')[0]
            pretrained_policy = PPO.load(model_dir, env=env, seed=seed, clip_range= get_schedule_fn(hyper_params['clip_range']))
            
            best_skill = [ d[d.index(":")+1:] for d in  os.listdir(seed_dir) if "best_finetuned_model_skillIndex:" in d][0]
            print(best_skill)
            finetuned_model_dir =  seed_dir + f"best_finetuned_model_skillIndex:{best_skill}/" + "/best_model"
            adapted_policy = SAC.load(finetuned_model_dir, env=env, seed=seed)
        
        intrinsic_reward_mean, reward_beforeFinetune_mean, reward_mean, entropy_mean = evaluate(env_name, n_skills, pretrained_policy, adapted_policy, discriminator, pm, best_skill, alg)
        # for testing 
        # intrinsic_reward_mean, reward_beforeFinetune_mean, reward_mean, entropy_mean = (1 + np.random.randn()*0.25, 1 + np.random.randn()*0.25, 1 + np.random.randn()*0.25, 1 + np.random.randn()*0.25)
        d = {'Algorithm': alg.upper(), 
                     "Normalized Extrinsic Return After Adaptation (%)": reward_mean/asym*100, 
                     "Intrinsic Return": intrinsic_reward_mean, 
                     "State Entropy": entropy_mean, 
                     "Normalized Extrinsic Return Before Adaptation (Best Skill) (%)": reward_beforeFinetune_mean/asym*100,
                     "Environment": env_name[: env_name.index('-')] if env != "MountainCarContinuous-v0" else "MountainCar"
                    }
        results_list.append(d)
        
    return results_list
    
def barplot(env_names, n_skills_list, lbs, pms, algs, stamps, colors):
    results = []
    for i in range(len(env_names)):
        env_name = env_names[i]
        n_skills = n_skills_list[i]
        pm = pms[i]
        lb = lbs[i]
        alg = algs[i]
        stamp = stamps[i]
        main_exper_dir = conf.log_dir_finetune + f"cls:{pm}, lb:{lb}/"
        env_dir = main_exper_dir + f"env: {env_name}, alg:{alg}, stamp:{stamp}/"
        discriminator_hyperparams = pd.read_csv(env_dir + "discriminator_hyperparams.csv").to_dict('records')[0]
        # (env_name, n_skills, pm, best_skill, alg, stamp, lb)
        r = extract_results(env_name, n_skills, pm, alg, stamp, lb)
        results.extend(r)
    results_df = pd.DataFrame(results)
    results_df.to_csv("result.csv")
    for col in columns[2:]:
        plt.figure(figsize=conf.barplot_figsize)
        y_axis = col
        # plt.title(y_axis)
        # ax = sns.barplot(x="Algorithm", y=y_axis, data=results_df, hue="Environment")
        if col == "Normalized Extrinsic Return After Adaptation (%)":
            print(algs)
            results_copy = results_df.copy()
            envs = set(results_df["Environment"].to_list())
            for e in envs:
                print(e)
                d = [ r for r in random_init if r['Environment'] == e  ]
                print(d)
                results_copy = results_copy.append(d)
            print(results_copy)
            input()
            ax = sns.barplot(x="Environment", y=y_axis, data=results_copy, hue="Algorithm", hue_order=[a.upper() for a in algs] + ["Random init"], palette=colors)
            sns.move_legend(ax, "lower left", title='Algorithm')
        else:
            ax = sns.barplot(x="Environment", y=y_axis, data=results_df, hue="Algorithm", hue_order=[a.upper() for a in algs], palette=colors)
            ax.legend_.remove()
        filename = f"adaptation_experiment_final_{y_axis}.png"
        files_dir = f"Vis/adaptation_experiment_cls:{discriminator_hyperparams['parametrization']}, lb:{discriminator_hyperparams['lower_bound']}/"
        os.makedirs(files_dir, exist_ok=True)
        plt.savefig(files_dir + filename)
        
    
    
    


if __name__ == "__main__":
    # args = cmd_args()
    # file_dir = conf.log_dir_finetune + f"{args.alg}_{args.env}_{args.stamp}/" + "eval_results/" + "evaluations.npz"
    # file_dir = conf.log_dir_finetune + f"{args.alg}_{args.env}_{args.stamp}/" + "evaluations.npz"
    print("######## BarPlot ########")
    env_names = ["MountainCarContinuous-v0", "MountainCarContinuous-v0"]
    skills_l = [10, 10]
    algs = ["sac", "ppo"]
    stamps = [1643572784.9260342, 1644513661.0212696]
    pms = ["MLP", "MLP"]
    lbs = ["ba", "ba"]
    colors = ["b", "r"]
    barplot(env_names, skills_l, lbs, pms, algs, stamps, colors)
    print("######## Skills evaluation ########")
    env_name = "MountainCarContinuous-v0"
    skills_l = [10, 10]
    algs = ["sac", "ppo"]
    stamps = [1643572784.9260342, 1644513661.0212696]
    cls = ["MLP", "MLP"]
    lbs = ["ba", "ba"]
    colors = ["b", "r"]
    legends = [a.upper() for a in algs]
    plt.figure()
    title = "MountainCar" if env_name == "MountainCarContinuous-v0" else env_name[: env_name.index('-')]
    plt.title(title)
    xlabel = "Environment Timesteps"
    ylabel = "Intrinsic Return"
    asym = asymp_perofrmance[env_name]
    for i in range(len(algs)):
        skills = skills_l[i]
        pm = cls[i]
        lb = lbs[i]
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
            sns.lineplot(x=steps, y=data_mean, label=legends[i], color=plot_color)
            plt.legend()
        else:
            sns.lineplot(x=steps, y=data_mean, color=plot_color)
        plt.fill_between(steps, (data_mean-data_std), (data_mean+data_std), color=plot_color, alpha=0.3)
        plt.xlabel(xlabel)
        plt.ylabel(ylabel)
        # plt.grid(False)
        if len(algs) == 1:
            filename = f"{env_name}_pretraining_{alg}"
            files_dir = f"Vis/{env_name}"
            os.makedirs(files_dir, exist_ok=True)
            plt.savefig(f'{files_dir}/{filename}')    
    if len(algs) > 1:
        filename = f"{env_name}_pretraining_all_algs"
        files_dir = f"Vis/{env_name}_cls:{pm}_lb:{lb}"
        os.makedirs(files_dir, exist_ok=True)
        plt.savefig(f'{files_dir}/{filename}')    
    
    # loop throgh the finetuning results
    # loop throgh the finetuning results
    plt.figure()
    title = "MountainCar" if env_name == "MountainCarContinuous-v0" else env_name[: env_name.index('-')]
    plt.title(title)
    print("######## Finetine evaluation ########")
    seeds_list = []
    ylabel = "Normalized Extrinsic Return (%)"
    for i in range(len(algs)):
        skills = skills_l[i]
        pm = cls[i]
        lb = lbs[i]
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
            ax = sns.lineplot(x=steps, y=data_mean, label=legends[i], color=plot_color)
        else:
            ax = sns.lineplot(x=steps, y=data_mean, color=plot_color)
        plt.fill_between(steps, (data_mean-data_std), (data_mean+data_std), color=plot_color, alpha=0.3)
        plt.xlabel(xlabel)
        plt.ylabel(ylabel)
        # plt.grid(False)
        r = random_performance[env_name]
        if len(algs) == 1:
            ax.axhline(r/asym * 100, label="Random Policy",  c='black', linestyle='dashed')
            ax.axhline(100, label="Expert Policy",  c='black', linestyle='dotted')
            plt.legend()
            filename = f"{env_name}_finetune_{alg}"
            files_dir = f"Vis/{env_name}_cls:{pm}_lb:{lb}"
            os.makedirs(files_dir, exist_ok=True)
            plt.savefig(f'{files_dir}/{filename}')    
    if len(algs) > 1:
        ax.axhline(r/asym * 100, label="Random Policy",  c='black', linestyle='dashed')
        ax.axhline(100, label="Expert Policy",  c='black', linestyle='dotted')
        plt.legend()
        filename = f"{env_name}_finetune_all_algs"
        files_dir = f"Vis/{env_name}"
        os.makedirs(files_dir, exist_ok=True)
        plt.savefig(f'{files_dir}/{filename}')  
    
        
        
        
