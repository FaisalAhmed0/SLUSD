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
    "MountainCarContinuous-v0": 100.0

    }

random_performance = {
    'HalfCheetah-v2': -0.37117783,
    'Walker2d-v2': -0.48223244,
    'Ant-v2': 0.39107815,
    "MountainCarContinuous-v0": -0.04685473, 
}
# TODO: These are dummy numbers replace it with the right ones
random_init = [
    {"Environment": 'HalfCheetah', "Normalized Extrinsic Return After Adaptation (%)": 65.557219, "Regularization": "No Regularization"},
    {"Environment": 'HalfCheetah', "Normalized Extrinsic Return After Adaptation (%)": 60.337075, "Regularization": "No Regularization"},
    {"Environment": 'HalfCheetah', "Normalized Extrinsic Return After Adaptation (%)": 79.938285, "Regularization": "No Regularization"},
    
    {"Environment": 'Walker2d', "Normalized Extrinsic Return After Adaptation (%)": 9.739630	, "Regularization": "No Regularization"},
    {"Environment": 'Walker2d', "Normalized Extrinsic Return After Adaptation (%)":4.041106, "Regularization": "No Regularization"},
    {"Environment": 'Walker2d', "Normalized Extrinsic Return After Adaptation (%)": 3.738718, "Regularization": "No Regularization"},
    
    {"Environment": 'Ant', "Normalized Extrinsic Return After Adaptation (%)": 0.810177, "Regularization": "No Regularization"},
    {"Environment": 'Ant', "Normalized Extrinsic Return After Adaptation (%)": 15.792851, "Regularization": "No Regularization"},
    {"Environment": 'Ant', "Normalized Extrinsic Return After Adaptation (%)": 13.807711, "Regularization": "No Regularization"},
    
    {"Environment": 'MountainCar', "Normalized Extrinsic Return After Adaptation (%)": -0.113409, "Regularization": "No Regularization"},
    {"Environment": 'MountainCar', "Normalized Extrinsic Return After Adaptation (%)": -0.809918, "Regularization": "No Regularization"},
    {"Environment": 'MountainCar', "Normalized Extrinsic Return After Adaptation (%)": 100.252294, "Regularization": "No Regularization"},
    
]

# no regularization
# no_reg = [
#     {"Environment": 'HalfCheetah', "Normalized Extrinsic Return After Adaptation (%)": 3.977155901244182132e+03/asymp_perofrmance['HalfCheetah-v2']*100, "Algorithm": "Random init"},
#     {"Environment": 'HalfCheetah', "Normalized Extrinsic Return After Adaptation (%)": 3.661330217032705150e+03/asymp_perofrmance['HalfCheetah-v2']*100, "Algorithm": "Random init"},
#     {"Environment": 'HalfCheetah', "Normalized Extrinsic Return After Adaptation (%)": 3.777719310452177979e+03/asymp_perofrmance['HalfCheetah-v2']*100, "Algorithm": "Random init"},
    
#     {"Environment": 'Walker2d', "Normalized Extrinsic Return After Adaptation (%)": 4.950961607268399121e+02/asymp_perofrmance['Walker2d-v2']*100, "Algorithm": "Random init"},
#     {"Environment": 'Walker2d', "Normalized Extrinsic Return After Adaptation (%)": 5.050660942998074461e+02/asymp_perofrmance['Walker2d-v2']*100, "Algorithm": "Random init"},
#     {"Environment": 'Walker2d', "Normalized Extrinsic Return After Adaptation (%)": 2.915505078088318669e+02/asymp_perofrmance['Walker2d-v2']*100, "Algorithm": "Random init"},
    
#     {"Environment": 'Ant', "Normalized Extrinsic Return After Adaptation (%)": 4.935468207680875139e+02/asymp_perofrmance['Ant-v2']*100, "Algorithm": "Random init"},
#     {"Environment": 'Ant', "Normalized Extrinsic Return After Adaptation (%)": 2.522865492915514665e+02/asymp_perofrmance['Ant-v2']*100, "Algorithm": "Random init"},
#     {"Environment": 'Ant', "Normalized Extrinsic Return After Adaptation (%)": 7.162829405419695377e+02/asymp_perofrmance['Ant-v2']*100, "Algorithm": "Random init"},
    
#     {"Environment": 'MountainCarContinuous', "Normalized Extrinsic Return After Adaptation (%)": -1.697574145149261080e-04/asymp_perofrmance['MountainCarContinuous-v0']*100, "Algorithm": "Random init"},
#     {"Environment": 'MountainCarContinuous', "Normalized Extrinsic Return After Adaptation (%)": -3.889756080078754464e-04/asymp_perofrmance['MountainCarContinuous-v0']*100, "Algorithm": "Random init"},
#     {"Environment": 'MountainCarContinuous', "Normalized Extrinsic Return After Adaptation (%)": -3.093423134238547888e-04/asymp_perofrmance['MountainCarContinuous-v0']*100, "Algorithm": "Random init"},
    
# ]



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


def get_cmap(n, name='hsv'):
    '''Returns a function that maps each index in 0, 1, ..., n-1 to a distinct 
    RGB color; the keyword argument name must be a standard mpl colormap name.'''
    return plt.cm.get_cmap(name, n)

def reg_label(label):
    value = label[-1: -(label[::-1].index('_')+1):-1][::-1]
    reg = (label[: -len(value)-1]).strip()
    # print(label)
    # print(value)
    # print(reg)
    # input()
    if reg == "weight_decay":
        return f"Weight decay_{value}"
    elif reg == "dropout":
        return f"Dropout_{value}"
    elif reg == "label_smoothing":
        return "Label smoothing"
    elif reg == "gp":
        return f"Gradient penalty_{value}"
    elif reg == "mixup":
        return "Mixup"
    
def barplot():
    results_df = pd.read_csv("regularization_exper_final/cls:MLP, lb:ba/results_df.csv")
    results_df = results_df.rename(columns = {"Normalized Reward After Adaptation (%)": "Normalized Extrinsic Return After Adaptation (%)"})
    c = plt.get_cmap("tab20c", 21)
    colors = [c(0), c(1), c(2), c(4), c(5), c(6), c(8), c(9), c(12), c(16), c(20)]
    # print(results_df["Normalized Reward After Adaptation (%)"])
    
    # print(results_df)
    # print()
    for col in columns[2:]:
        # plt.figure(figsize=conf.barplot_figsize)
        plt.figure(figsize=conf.learning_curve_figsize)
        y_axis = col
        # plt.title(y_axis)
        # ax = sns.barplot(x="Algorithm", y=y_axis, data=results_df, hue="Environment")
        if col == "Normalized Extrinsic Return After Adaptation (%)":
            results_copy = results_df.copy()
            envs = set(results_df["Environment"].to_list())
            for e in envs:
                print(e)
                d = [ r for r in random_init if r['Environment'] == e  ]
                print(d)
                results_copy = results_copy.append(d)
            print(results_copy)
            # input()
            ax = sns.barplot(x="Environment", y=y_axis, data=results_copy, hue="Regularization", palette=colors, hue_order=[reg_label(reg) for reg in regs] + ["No Regularization"])
            ax.legend_.remove()
            filename = f"regularization_experiment_{y_axis}.png"
            files_dir = f"Vis/aregularization_experiment_cls:MLP, lb:ba_with_baseline/"
            os.makedirs(files_dir, exist_ok=True)
            plt.savefig(files_dir + filename)
            figLegend = plt.figure(figsize = (10,5.1))
            plt.figlegend(*ax.get_legend_handles_labels(), loc = 'upper left', frameon=False, ncol=2, bbox_to_anchor=(0, 1), fontsize='medium')
            plt.savefig(files_dir + f"regularization_experiment_barplot_legend.png")
            break
    results_copy.to_csv("Vis/aregularization_experiment_cls:MLP, lb:ba_with_baseline/final_results_with_baseline.csv")
        
    
    
regs = ["weight_decay_0.1", "weight_decay_0.001", "weight_decay_1e-05", "dropout_0.1", "dropout_0.5", "dropout_0.9", "gp_5", "gp_10", "label_smoothing_0.3", "mixup_True"]


if __name__ == "__main__":
    print("######## BarPlot ########")
    barplot()
#     print("######## Skills evaluation ########")
    env_names = ["MountainCarContinuous-v0", "HalfCheetah-v2", "Walker2d-v2", "Ant-v2"]
    stamps_and_regs =  [(1645650879.800285, "weight_decay_0.1"), 
                        (1645650879.81344, "weight_decay_0.001"), 
                        (1645650879.8269858, "weight_decay_1e-05") ,
                        (1645650879.8479288, "dropout_0.1")  , 
                        (1645650879.8718913, "dropout_0.5"), 
                        (1645650879.8962636, "dropout_0.9"),
                        (1645650879.923484, "gp_5"), 
                        (1645650879.9514167, "gp_10"),
                        (1645650879.9780629, "label_smoothing_0.3"), 
                        (1645650880.0045395, "mixup_True"),
                        ([1645692889.4627287, 1645692889.5071442, 1645692889.5540564, 1645692889.6045046], "No Regularization")]
    c = plt.get_cmap("tab20c", 17)
    colors = [c(0), c(1), c(2), c(4), c(5), c(6), c(8), c(9), c(12), c(16), c(20)]
    main_exper_dir = "regularization_exper_final/cls:MLP, lb:ba/"
    xlabel = "Environment Timesteps"
    ylabel = "Normalized Extrinsic Return After Adaptation (%)"
    for env_name in env_names:
        plt.figure(figsize=conf.learning_curve_figsize)
        title = "MountainCar" if env_name == "MountainCarContinuous-v0" else env_name[: env_name.index('-')]
        plt.title(title)
        asym = asymp_perofrmance[env_name]
        for i, (stamp, reg) in enumerate(stamps_and_regs):
            if reg == "No Regularization":
                env_dir =  main_exper_dir + f"env: {env_name}, alg:sac, stamp:{stamp[env_names.index(env_name)]}/" 
                print(env_name)
            else:
                env_dir =  main_exper_dir + f"env: {env_name}, alg:sac, stamp:{stamp}_reg:{reg}/" 
            color = colors[i]
            seeds_list = []
            for seed in conf.seeds:
                seed_dir = env_dir + f"seed:{seed}/"
                file_dir_skills = seed_dir + "finetune_eval_results/" + "evaluations.npz"
                files = np.load(file_dir_skills)
                steps = files['timesteps']
                results = files['results']
                seeds_list.append(results.mean(axis=1).reshape(-1))
            data_mean = np.mean(seeds_list, axis=0)/asym * 100
            # print(f"seeds list shape {np.array(seeds_list).shape}")
            data_std = np.std(seeds_list, axis=0)/asym * 100
            if reg == "No Regularization":
                label = "No Regularization"
                print(data_mean)
                print("Here")
            else:
                label = reg_label(reg)
            # print(label)
            ax = sns.lineplot(x=steps, y=data_mean, label=label, color=color)
            plt.fill_between(steps, (data_mean-data_std), (data_mean+data_std), color=color, alpha=0.3)
            # plt.legend()
        plt.xlabel(xlabel)
        plt.ylabel(ylabel)
        ax.legend_.remove()
        filename = f"{env_name}_all_regularizors.png"
        files_dir = f"Vis/aregularization_experiment_cls:MLP, lb:ba_with_baseline/"
        os.makedirs(files_dir, exist_ok=True)
        plt.savefig(f'{files_dir}/{filename}')    
        figLegend = plt.figure(figsize = (9,1.1), )
        plt.figlegend(*ax.get_legend_handles_labels(), loc = 'upper left', frameon=False, ncol=2, bbox_to_anchor=(0, 1), fontsize='large')
        plt.savefig(files_dir + f"regularization_experiment_learning_curves_legend.png", dpi=100)
                
                
                
                
                    
                
                
            
            
            
#     for j in range(len(env_names)):
#         env_name = env_names[j]
#         skills_l = skill_l[j]
#         algs = ["sac", "ppo"]
#         stamps = stamps_l[j]
#         cls = ["MLP", "MLP"]
#         lbs = ["ba", "ba"]
#         colors = ["b", "r"]
#         legends = [a.upper() for a in algs]
        # plt.figure()
        # title = "MountainCar" if env_name == "MountainCarContinuous-v0" else env_name[: env_name.index('-')]
        # plt.title(title)
#         xlabel = "Environment Timesteps"
#         ylabel = "Intrinsic Return"
#         asym = asymp_perofrmance[env_name]
#         for i in range(len(algs)):
#             skills = skills_l[i]
#             pm = cls[i]
#             lb = lbs[i]
#             alg = algs[i]
#             print(f"Algorithm: {alg}")
#             plot_color = colors[i]
#             stamp = stamps[i]
#             # legend = alg
#             main_exper_dir = conf.log_dir_finetune + f"cls:{pm}, lb:{lb}/"
#             env_dir = main_exper_dir + f"env: {env_name}, alg:{alg}, stamp:{stamp}/"
#             seeds_list = []
#             for seed in seeds:
#                 seed_everything(seed)
#                 seed_dir = env_dir + f"seed:{seed}/"
                # file_dir_skills = seed_dir + "eval_results/" + "evaluations.npz"
                # files = np.load(file_dir_skills)
#                 steps = files['timesteps']
#                 results = files['results']
#                 print(f"results.shape: {results.shape}")
#                 seeds_list.append(results.mean(axis=1).reshape(-1))
#                 # data_mean = results.mean(axis=1)
#                 # data_std = results.std(axis=1)
#                 # print(f"mean: {data_mean[-1]}")
#                 # print(f"std: {data_std[-1]}")
            # data_mean = np.mean(seeds_list, axis=0)
            # print(f"seeds list shape {np.array(seeds_list).shape}")
            # # input()
            # data_std = np.std(seeds_list, axis=0)
#             # data_mean = np.convolve(data_mean, np.ones(10)/10, mode='valid') 
#             # data_std = np.convolve(data_std, np.ones(10)/10, mode='valid') 
#             # steps = np.convolve(steps, np.ones(10)/10, mode='valid') 
#             print(f"data shape: {data_mean.shape}")
#             print(f"steps shape: {steps.shape}")
#             # input()
#             if len(algs) > 1:
#                 sns.lineplot(x=steps, y=data_mean, label=legends[i], color=plot_color)
#                 plt.legend()
#             else:
#                 sns.lineplot(x=steps, y=data_mean, color=plot_color)
#             plt.fill_between(steps, (data_mean-data_std), (data_mean+data_std), color=plot_color, alpha=0.3)
            # plt.xlabel(xlabel)
            # plt.ylabel(ylabel)
#             # plt.grid(False)
#             if len(algs) == 1:
#                 filename = f"{env_name}_pretraining_{alg}"
#                 files_dir = f"Vis/{env_name}"
#                 os.makedirs(files_dir, exist_ok=True)
#                 plt.savefig(f'{files_dir}/{filename}')    
#         if len(algs) > 1:
            # filename = f"{env_name}_pretraining_all_algs"
            # files_dir = f"Vis/{env_name}_cls:{pm}_lb:{lb}"
            # os.makedirs(files_dir, exist_ok=True)
            # plt.savefig(f'{files_dir}/{filename}')    

#         # loop throgh the finetuning results
#         # loop throgh the finetuning results
#     print("######## Finetine evaluation ########")
#     seeds_list = []
#     ylabel = "Normalized Extrinsic Return (%)"
#     for j in range(len(env_names)):
#         env_name = env_names[j]
#         skills_l = skill_l[j]
#         stamps = stamps_l[j]
#         plt.figure()
#         title = "MountainCar" if env_name == "MountainCarContinuous-v0" else env_name[: env_name.index('-')]
#         plt.title(title)
#         for i in range(len(algs)):
#             skills = skills_l[i]
#             pm = cls[i]
#             lb = lbs[i]
#             alg = algs[i]
#             print(f"Algorithm: {alg}")
#             plot_color = colors[i]
#             stamp = stamps[i]
#             legend = alg
#             main_exper_dir = conf.log_dir_finetune + f"cls:{pm}, lb:{lb}/"
#             env_dir = main_exper_dir + f"env: {env_name}, alg:{alg}, stamp:{stamp}/"
#             seed_list = []
#             for seed in seeds:
#                 seed_everything(seed)
#                 seed_dir = env_dir + f"seed:{seed}/"
#                 file_dir_finetune = seed_dir + "finetune_eval_results/" + "evaluations.npz"
#                 files = np.load(file_dir_finetune)
#                 steps = files['timesteps']
#                 results = files['results']
#                 seeds_list.append(results.mean(axis=1).reshape(-1))
#             data_mean = np.mean(seeds_list, axis=0)/asym * 100
#             data_std = np.std(seeds_list, axis=0)/asym * 100
#             if len(algs) > 1:
#                 ax = sns.lineplot(x=steps, y=data_mean, label=legends[i], color=plot_color)
#             else:
#                 ax = sns.lineplot(x=steps, y=data_mean, color=plot_color)
#             plt.fill_between(steps, (data_mean-data_std), (data_mean+data_std), color=plot_color, alpha=0.2)
#             plt.xlabel(xlabel)
#             plt.ylabel(ylabel)
#             # plt.grid(False)
#             r = random_performance[env_name]
#             # if len(algs) == 1:
#             #     ax.axhline(r/asym * 100, label="Random Policy",  c='black', linestyle='dashed')
#             #     ax.axhline(100, label="Expert Policy",  c='black', linestyle='dotted')
#             #     plt.legend()
#             #     filename = f"{env_name}_finetune_{alg}"
#             #     files_dir = f"Vis/{env_name}_cls:{pm}_lb:{lb}"
#             #     os.makedirs(files_dir, exist_ok=True)
#             #     plt.savefig(f'{files_dir}/{filename}')    
#         if len(algs) > 1:
#             ax.axhline(r/asym * 100, label="Random Policy",  c='black', linestyle='dashed')
#             ax.axhline(100, label="Expert Policy",  c='black', linestyle='dotted')
#             plt.legend()
#             filename = f"{env_name}_finetune_all_algs"
#             files_dir = f"Vis/{env_name}"
#             os.makedirs(files_dir, exist_ok=True)
#             plt.savefig(f'{files_dir}/{filename}')  

        
        
        
