import argparse
import pickle
import json

from src.config import conf
# diayn with model-free RL
from src.diayn import DIAYN 
# diayn with evolution stratigies
from src.diayn_es import DIAYN_ES 
# diayn with evolution model-based RL
from src.diayn_mb import DIAYN_MB 
from src.utils import seed_everything, evaluate_pretrained_policy_ext, evaluate_adapted_policy, evaluate_state_coverage, evaluate_pretrained_policy_intr, evaluate, save_final_results, plot_learning_curves

import torch
import torch.multiprocessing as mp
torch.set_num_threads(1)
torch.multiprocessing.set_sharing_strategy('file_system')


import numpy as np
import pandas as pd
import gym

import time
import os
import random
from collections import defaultdict

import matplotlib.pyplot as plt
import seaborn as sns
sns.set_theme(style="darkgrid")
sns.set(font_scale = conf.font_scale)

# Asymptotic performance of the expert
asymp_perofrmance = {
    'HalfCheetah-v2': 8070.205213363435,
    'Walker2d-v2': 5966.481684037183,
    'Ant-v2': 6898.620003217143,
    "MountainCarContinuous-v0": 95.0

    }

# shared parameters
params = dict( n_skills = 30,
           pretrain_steps = int(20e3),
           finetune_steps = int(1e5),
           buffer_size = int(1e7),
           min_train_size = int(1e4),
             )


# list for multip-processing
# Todo: complete the rest of the environment
envs_mp = [
    {   # 1
        'weight_decay':{
            'MountainCarContinuous-v0': dict(
                pretrain_steps = int(3.5e5),
                n_skills = 10,
                value = [1e-1, 1e-3, 1e-5]
            ),
            'HalfCheetah-v2': dict(
                pretrain_steps = int(5e5),
                n_skills = 30,
                value = [1e-1, 1e-3, 1e-5]
            ),
            'Walker2d-v2': dict(
                pretrain_steps = int(7e5),
                n_skills = 30,
                value = [1e-1, 1e-3, 1e-5]
            ),
            'Ant-v2': dict(
                pretrain_steps = int(1e6),
                n_skills = 30,
                value = [1e-1, 1e-3, 1e-5]
            ),
            
        }
        
        
    },
    {   # 2
        'dropout':{
            'MountainCarContinuous-v0': dict(
                pretrain_steps = int(3.5e5),
                n_skills = 10,
                value = [0.1, 0.5, 0.9]
            ),
            'HalfCheetah-v2': dict(
                pretrain_steps = int(5e5),
                n_skills = 30,
                value = [0.1, 0.5, 0.9]
            ),
            'Walker2d-v2': dict(
                pretrain_steps = int(7e5),
                n_skills = 30,
                value = [0.1, 0.5, 0.9]
            ),
            'Ant-v2': dict(
                pretrain_steps = int(1e6),
                n_skills = 30,
                value = [0.1, 0.5, 0.9]
            ),
        }
        
        
    },
    {   # 3
        'gp':{
            'MountainCarContinuous-v0': dict(
                pretrain_steps = int(3.5e5),
                n_skills = 10,
                value = [1, 5, 10]
            ),
            'HalfCheetah-v2': dict(
                pretrain_steps = int(5e5),
                n_skills = 30,
                value = [1, 5, 10]
            ),
            'Walker2d-v2': dict(
                pretrain_steps = int(7e5),
                n_skills = 30,
                value = [1, 5, 10]
            ),
            'Ant-v2': dict(
                pretrain_steps = int(1e6),
                n_skills = 30,
                value = [1, 5, 10]
            ),
        }
        
        
    },
    {   # 4
        'label_smoothing':{
            'MountainCarContinuous-v0': dict(
                pretrain_steps = int(3.5e5),
                n_skills = 10,
                value = [0.2]
            ),
            'HalfCheetah-v2': dict(
                pretrain_steps = int(5e5),
                n_skills = 30,
                value = [0.2]
            ),
            'Walker2d-v2': dict(
                pretrain_steps = int(7e5),
                n_skills = 30,
                value = [0.2]
            ),
            'Ant-v2': dict(
                pretrain_steps = int(1e6),
                n_skills = 30,
                value = [0.2]
            ),
        }
        
        
    },
    {   # 5
        'mixup':{
            'MountainCarContinuous-v0': dict(
                pretrain_steps = int(3.5e5),
                n_skills = 10,
                value = [True]
            ),
            'HalfCheetah-v2': dict(
                pretrain_steps = int(5e5),
                n_skills = 30,
                value = [True]
            ),
            'Walker2d-v2': dict(
                pretrain_steps = int(7e5),
                n_skills = 30,
                value = [True]
            ),
            'Ant-v2': dict(
                pretrain_steps = int(1e6),
                n_skills = 30,
                value = [True]
            ),
        }
        
        
    }
    
]


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

# PETS Hyperparameters 
pets_hyperparams = dict(
    learning_rate = 1e-3,
    batch_size = 128,
    ensemble_size = 5,
    trial_length = 200,
    population_size = 250,
    planning_horizon = 10,
    elite_ratio = 0.05,
    num_particles = 20,
    weight_decay = 5e-5,
    algorithm = "pets"
)


# Evolution Stratigies Hyperparameters after hyperparameters search
es_hyperparams = dict(
    lr = 1e-2, # learning rate 
    iterations = 100, # iterations 
    pop_size = 500, # population size
    algorithm = "es"
)

hyperparams = {
    'ppo': ppo_hyperparams,
    'sac': sac_hyperparams,
    'pets': pets_hyperparams,
    'es' : es_hyperparams
}

# Discriminator Hyperparameters
discriminator_hyperparams = dict(
    learning_rate = 3e-4,
    batch_size = 64,
    n_epochs = 1,
    weight_decay = 0, 
    dropout = None, # The dropout probability
    label_smoothing = 0.0, # usually 0.2
    gp = None, # the weight of the gradient penalty term
    mixup = False,
    gradient_clip = None,
    temperature = 1,
    parametrization = "Linear", # TODO: add this as a CMD argument MLP, Linear, Separable, Concat
    lower_bound = "ba", # ba, tuba, nwj, nce, interpolate
    log_baseline = None, 
    alpha_logit = -5., # small value of alpha => reduce the variance of nwj by introduce some nce bias 
)

default_disc_params = dict(
        learning_rate = 3e-4,
        batch_size = 64,
        n_epochs = 1,
        weight_decay = 0, 
        dropout = None, # The dropout probability
        label_smoothing = 0.0, # usually 0.2
        gp = None, # the weight of the gradient penalty term
        mixup = False,
        gradient_clip = None,
        temperature = 1,
        parametrization = "Linear", # TODO: add this as a CMD argument MLP, Linear, Separable, Concat
        lower_bound = "ba", # ba, tuba, nwj, nce, interpolate
        log_baseline = None, 
        alpha_logit = -5., # small value of alpha => reduce the variance of nwj by introduce some nce bias 
)


# Extract arguments from terminal
def cmd_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--env', type=str, default="MountainCarContinuous-v0")
    parser.add_argument("--alg", type=str, default="ppo")
    parser.add_argument("--skills", type=int, default=6)
    parser.add_argument("--presteps", type=int, default=int(1e6))
    parser.add_argument("--pm", type=str, default="MLP")
    parser.add_argument("--lb", type=str, default="ba")
    parser.add_argument("--run_all", type=bool, default=False)
    args = parser.parse_args()
    return args

# save the hyperparameters
def save_params(alg, directory):
    # save the hyperparams in a csv files
    alg_hyperparams = hyperparams[alg]
    params
    alg_hyperparams_df = pd.DataFrame(alg_hyperparams, index=[0])
    alg_hyperparams_df.to_csv(f"{directory}/{alg}_hyperparams.csv")
    
    discriminator_hyperparams_df = pd.DataFrame(discriminator_hyperparams, index=[0])
    discriminator_hyperparams_df.to_csv(f"{directory}/discriminator_hyperparams.csv")
    
    # convert the config namespace to a dictionary
    exp_params_df = pd.DataFrame(params, index=[0])
    exp_params_df.to_csv(f"{directory}/params.csv")
    
    # general configrations
    config_d = vars(conf)
    dictionary_data = config_d
    a_file = open(f"{directory}/configs.json", "w")
    json.dump(config_d, a_file)
    a_file.close()
    print(f"{alg} hyperparame`ters: {alg_hyperparams}\n" )
    print(f"discriminator hyperparameters: {discriminator_hyperparams}\n" )
    print(f"shared experiment parameters: {params}\n" )
    print(f"configurations: {config_d }\n" )
             
        
def train_all(env_params, results_df_list):
    plots_dict = {}
    columns = ['Regularization', "Environment", "Reward After Adaptation", "State Entropy", "Intrinsic Reward" , "Reward Before Adaptation (Best Skill)"]
    results_df = pd.DataFrame(columns=columns)
    print(f"{env_params}")
    alg = "sac"
    #####################
    for reg in env_params:
        for env in env_params[reg]:
            for v in env_params[reg][env]['value']:
                if env not in plots_dict:
                    plots_dict[env] = {}
                # save a timestamp
                timestamp = time.time()
                # Environment directory
                env_dir = main_exper_dir + f"env: {env}, alg:{alg}, stamp:{timestamp}/"
                os.makedirs(env_dir, exist_ok=True)
                alg_params = hyperparams[alg]
                params['n_skills'] = env_params[reg][env]['n_skills']
                params['pretrain_steps'] = env_params[reg][env]['pretrain_steps']
                print(f"stamp: {timestamp}, alg: {alg}, env: {env}, n_skills: {params['n_skills']}, pretrain_steps: {params['pretrain_steps']}")
                discriminator_hyperparams[reg] = v
                save_params(alg, env_dir)
                seed_results = []
                for i in range(len(conf.seeds)):
                    seed_everything(conf.seeds[i])
                    seed_dir = env_dir + f"seed:{conf.seeds[i]}"
                    os.makedirs(seed_dir, exist_ok=True)
                    diayn = DIAYN(params, alg_params, discriminator_hyperparams, env, alg, seed_dir, seed=conf.seeds[i], conf=conf, timestamp=timestamp, adapt_params=sac_hyperparams)
                    # pretraining step
                    pretrained_policy, discriminator = diayn.pretrain()
                    # fine-tuning step 
                    adapted_policy, best_skill = diayn.finetune()
                    # Evaluate the policy 
                    intrinsic_reward_mean, reward_beforeFinetune_mean, reward_mean, entropy_mean = evaluate(env, params['n_skills'], pretrained_policy, adapted_policy, discriminator, 
                                                                                                        discriminator_hyperparams['parametrization'], best_skill, alg)
                    # for testing
                    # intrinsic_reward_mean, reward_beforeFinetune_mean, reward_mean, entropy_mean = [np.random.randn() + np.log(10),
                    #                                                                                 np.random.randn() + 100,
                    #                                                                                 np.random.randn() + 100,
                    #                                                                                 np.random.randn() + 100
                    #                                                                                ]
                        
                    asym = asymp_perofrmance[env]
                    # save results
                    seed_results.append([intrinsic_reward_mean, reward_beforeFinetune_mean/asym, reward_mean/asym, entropy_mean])
                    label = reg_label(reg, v)
                    d = {'Regularization': label, 
                         "Reward After Adaptation": reward_mean/asym, 
                         "Intrinsic Reward": intrinsic_reward_mean, 
                         "State Entropy": entropy_mean, 
                         "Reward Before Adaptation (Best Skill)": reward_beforeFinetune_mean/asym,
                         "Environment": env[: env.index('-')] if env != "MountainCarContinuous-v0" else "MountainCar"
                        }
                    results_df = results_df.append(d, ignore_index=True)
                # add to the plot dict
                # if "algs" not in plots_dict[env]:
                #     plots_dict[env]["algs"] = []
                # if "stamps" not in plots_dict[env]:
                #     plots_dict[env]["stamps"] = []
                # if "skills" not in plots_dict[env]:
                #     plots_dict[env]["skills"] = []
                # if "pms" not in plots_dict[env]:
                #     plots_dict[env]["pms"] = []
                # if "lbs" not in plots_dict[env]:
                #     plots_dict[env]["lbs"] = []

                # plots_dict[env]["algs"].append(alg)
                # plots_dict[env]["stamps"].append(timestamp)
                # plots_dict[env]["skills"].append(params['n_skills'])
                # plots_dict[env]["pms"].append(discriminator_hyperparams['parametrization'])
                # plots_dict[env]["lbs"].append(discriminator_hyperparams['lower_bound'])
        
            save_final_results(seed_results, env_dir)
            results_df_list.append(results_df)
            # plots_d_list.append(plots_dict)
            discriminator_hyperparams[reg] = default_disc_params[reg]
    
    
def df_from_list(df_list, columns):
    results_df = pd.concat(df_list,ignore_index=True) 
    return results_df
    
    
'''
learning_rate = 3e-4,
batch_size = 64,
n_epochs = 1,
weight_decay = 0, 
dropout = None, # The dropout probability
label_smoothing = 0.0, # usually 0.2
gp = None, # the weight of the gradient penalty term
mixup = False,
gradient_clip = None,
temperature = 1,
parametrization = "Linear", # TODO: add this as a CMD argument MLP, Linear, Separable, Concat
lower_bound = "ba", # ba, tuba, nwj, nce, interpolate
log_baseline = None, 
alpha_logit
'''
def reg_label(label, value):
    if label == "weight_decay":
        return f"Weight decay_{value}"
    elif label == "dropout":
        return f"Dropout_{value}"
    elif label == "label_smoothing":
        return "Label smoothing"
    elif label == "gp":
        return f"Gradient penalty_{value}"
    elif label == "mixup":
        return "Mixup"
    
# From https://stackoverflow.com/a/47749903
def export_legend(legend, filename="legend.png"):
    fig  = legend.figure
    fig.canvas.draw()
    bbox  = legend.get_window_extent().transformed(fig.dpi_scale_trans.inverted())
    fig.savefig(filename, dpi="figure", bbox_inches=bbox)
    
if __name__ == "__main__":
    args = cmd_args()
    run_all = args.run_all
    discriminator_hyperparams['lower_bound'] = args.lb
    discriminator_hyperparams['parametrization'] = args.pm
    main_exper_dir = None
    main_exper_dir = conf.regularization_exper_dir + f"cls:{discriminator_hyperparams['parametrization']}, lb:{discriminator_hyperparams['lower_bound']}/"
    
    columns = ['Regularization', "Environment", "Reward After Adaptation", "State Entropy", "Intrinsic Reward" , "Reward Before Adaptation (Best Skill)"]
        
    if run_all:
        # for mp
        manager = mp.Manager()
        results_df_list = manager.list()
        n_processes = len(envs_mp)
        processes_list = []
        for i in range(n_processes):
            p = mp.Process(target=train_all, args=(envs_mp[i], results_df_list))
            p.start()
            processes_list.append(p)
        for p in processes_list:
            p.join()
        print(f"results_df_list: {results_df_list}")
        # for plot_dict in plots_d_list:
        #     env_name = list(plot_dict.keys())[0]
        #     algs = plot_dict[env_name]['algs']
        #     stamps = plot_dict[env_name]['stamps']
        #     skills_list = plot_dict[env_name]['skills']
        #     pms = plot_dict[env_name]['pms']
        #     lbs = plot_dict[env_name]['lbs']
        #     asym = asymp_perofrmance[env_name]
        #     plot_learning_curves(env_name, algs, stamps, skills_list, pms, lbs, asym)
        results_df = df_from_list(results_df_list, columns)
        results_df.to_csv(f"{main_exper_dir}/results_df.csv")
        print("results df")
        print(f"{results_df}")
        files_dir = f"Vis/regularization_experiment_cls:{discriminator_hyperparams['parametrization']}, lb:{discriminator_hyperparams['lower_bound']}/"
        for col in columns[2:]:
            plt.figure(figsize=conf.barplot_figsize)
            y_axis = col
            # ax = sns.barplot(x="Algorithm", y=y_axis, data=results_df, hue="Environment")
            if col == "Reward After Adaptation":
                ax = sns.barplot(x="Environment", y=y_axis, data=results_df, hue="Regularization",)
                ax.legend_.remove()
                filename = f"regularization_experiment_final_{y_axis}.png"
                files_dir = f"Vis/regularization_experiment_cls:{discriminator_hyperparams['parametrization']}, lb:{discriminator_hyperparams['lower_bound']}/"
                os.makedirs(files_dir, exist_ok=True)
                plt.savefig(files_dir + filename)
                # legend = plt.legend()
                # export_legend(legend, files_dir + f"regularization_experiment_final_{y_axis}_legend.png")
                # create a second figure for the legend
                figLegend = plt.figure(figsize = (3,3))
                # produce a legend for the objects in the other figure
                plt.figlegend(*ax.get_legend_handles_labels(), loc = 'upper left')
                plt.savefig(files_dir + f"regularization_experiment_final_{y_axis}_legend.png")
                # sns.move_legend(ax, "lower left", title='Algorithm')
            else:
                ax = sns.barplot(x="Environment", y=y_axis, data=results_df, hue="Regularization",)
                ax.legend_.remove()
                filename = f"regularization_experiment_final_{y_axis}.png"
                os.makedirs(files_dir, exist_ok=True)
                plt.savefig(files_dir + filename)
    else:
        # save a timestamp
        timestamp = time.time()
        print(f"Experiment timestamp: {timestamp}")
        params['pretrain_steps'] = args.presteps
        params['n_skills'] = args.skills
        alg_params = hyperparams[args.alg]
        if args.alg in ['ppo', 'sac', 'pets']:
                params['n_skills'] = args.skills
                params['pretrain_steps'] = args.presteps
                print(f"stamp: {timestamp}, alg: {args.alg}, env: {args.env}, n_skills: {params['n_skills']}, pretrain_steps: {params['pretrain_steps']}")
        elif args.alg == "es":
            params['n_skills'] = args.skills
            alg_params['iterations'] = args.presteps
            # alg_params['iterations_finetune'] = env_params[alg][env]['adaptation_iterations']
            print(f"stamp: {timestamp}, alg: {args.alg}, env: {args.env}, n_skills: {params['n_skills']}, pretrain_iterations: {alg_params['iterations']}")
        env_dir = main_exper_dir + f"env: {args.env}, alg:{args.alg}, stamp:{timestamp}/"
        os.makedirs(env_dir, exist_ok=True)
        save_params(args.alg, env_dir)
        seed_results = []
        for i in range(len(conf.seeds)):
            seed_everything(conf.seeds[i])
            seed_dir = env_dir + f"seed:{conf.seeds[i]}"
            os.makedirs(seed_dir, exist_ok=True)
            if args.alg in ("sac", "ppo"):
                # TODO add ES, when it is ready
                diayn = DIAYN(params, alg_params, discriminator_hyperparams, args.env, args.alg, seed_dir, seed=conf.seeds[i], conf=conf, timestamp=timestamp, adapt_params=sac_hyperparams)
                # pretraining step
                pretrained_policy, discriminator = diayn.pretrain()
                # fine-tuning step 
                adapted_policy, best_skill = diayn.finetune()
                # Evaluate the policy 
                intrinsic_reward_mean, reward_beforeFinetune_mean, reward_mean, entropy_mean = evaluate(args.env, params['n_skills'], pretrained_policy, adapted_policy, discriminator, 
                                                                                                    discriminator_hyperparams['parametrization'], best_skill, args.alg)
            elif args.alg == "pets":
                diayn = DIAYN_MB(params, alg_params, discriminator_hyperparams, args.env, args.alg, seed_dir, seed=conf.seeds[i], conf=conf, timestamp=timestamp, adapt_params=sac_hyperparams)
                # pretraining step
                pretrained_policy, discriminator = diayn.pretrain()
                print("Finished")
                # fine-tune step
                adapted_policy, best_skill = diayn.finetune()
                # Evaluate the policy 
                # Evaluate the policy 
                intrinsic_reward_mean, reward_beforeFinetune_mean, reward_mean, entropy_mean = evaluate(args.env, params['n_skills'], pretrained_policy, adapted_policy, discriminator, 
                                                                                                    discriminator_hyperparams['parametrization'], best_skill, args.alg)
            elif args.alg == "es":
                diayn = DIAYN_ES(params, alg_params, discriminator_hyperparams, args.env, "es", seed_dir, seed=conf.seeds[i], conf=conf, timestamp=timestamp, args=args, adapt_params=sac_hyperparams)
                # pretraining step
                pretrained_policy, discriminator = diayn.pretrain()
                # adaptation step
                adapted_policy, best_skill = diayn.finetune()
                # Evaluate the policy 
                intrinsic_reward_mean, reward_beforeFinetune_mean, reward_mean, entropy_mean = evaluate(args.env, params['n_skills'], pretrained_policy, adapted_policy, discriminator, 
                                                                                                    discriminator_hyperparams['parametrization'], best_skill, args.alg)
            seed_results.append([intrinsic_reward_mean, reward_beforeFinetune_mean, reward_mean, entropy_mean])
        save_final_results(seed_results, env_dir)
        
        
        
        



    

