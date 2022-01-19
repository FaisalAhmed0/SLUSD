import gym
import argparse
import os

from src.config import conf
from src.diayn import DIAYN

import torch

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import time
import os
import random

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

# set the seed
# seed = 10
# random.seed(seed)
# np.random.seed(seed)
# torch.random.manual_seed(seed)



# Asymptotic performance of the expert
asymp_perofrmance = {
    'HalfCheetah-v2': 8070.205213363435,
    'Walker2d-v2': 5966.481684037183,
    'Ant-v2': 6898.620003217143,
    "MountainCarContinuous-v0": 95.0

    }

# shared parameters
params = dict( n_skills = 6,
           pretrain_steps = int(1e5),
           finetune_steps = int(1e5),
           buffer_size = int(1e7),
           min_train_size = int(1e4)
             )


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
    learning_starts = 1000,
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
    pop_size = 2, # population size
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

# list for multip-processing
envs_mp = [
    # 1
    { # 1 dof
     'ppo':{
        'MountainCarContinuous-v0': dict( 
           pretrain_steps = int(25e6)//3, 
            n_skills = 10 
             ),
        },
    'sac':{
        'MountainCarContinuous-v0': dict( 
           pretrain_steps = int (5e5)//3,
            n_skills = 10 
             ), 
        },
    'es':{
        'MountainCarContinuous-v0': dict( 
           pretrain_iterations = 20000//3,
            n_skills = 10
             ),
        },
    'pets':{
        'MountainCarContinuous-v0': dict( 
           pretrain_steps = int (3e5)//3,
            n_skills = 10 
             ),
        },
    
    },
    # 2
    { # 6 dof
     'ppo':{
        'HalfCheetah-v2': dict( 
           pretrain_steps = int(500e6)//3,
            n_skills = 30
             ), 
        },
    'sac':{
        'HalfCheetah-v2': dict( 
           pretrain_steps = int(2e6)//3,
            n_skills = 30
             ), 
        },
    'es':{
        'HalfCheetah-v2': dict( 
           pretrain_iterations = 25000//3,
            n_skills = 30 
             ), 
        },
    'pets':{
        'HalfCheetah-v2': dict( 
           pretrain_steps = int (3.5e5)//3,
            n_skills = 30 
             ), 
        },
    },
    # 3
    { # 6 dof
     'ppo':{
        'Walker2d-v2': dict( 
           pretrain_steps = int(700e6)//3,
            n_skills = 30
             ),
        },
    'sac':{
        'Walker2d-v2': dict( 
           pretrain_steps = int(2.5e6)//3,
            n_skills = 30
             ),
        },
    'es':{
        'Walker2d-v2': dict( 
           pretrain_iterations = 25000//3,
            n_skills = 30 
             ),
        },
    'pets':{
        'Walker2d-v2': dict( 
           pretrain_steps = int (4e5)//3,
            n_skills = 30 
             ),
        },
        
    },
    # 4
    { # 8 dof
     'ppo':{
        'Ant-v2': dict( 
           pretrain_steps = int(700e6)//3,
            n_skills = 30
             ), 
        },
    'sac':{
        'Ant-v2': dict( 
           pretrain_steps = int(3e6)//3,
            n_skills = 30
             ), 
        },
    'es':{
        'Ant-v2': dict( 
           pretrain_iterations = 30000//3,
            n_skills = 30 
             ), 
        },
    'pets':{
        'Ant-v2': dict( 
           pretrain_steps = int (5e5)//3,
            n_skills = 30 
             ),
        },
    },    
    
]

# Extract arguments from terminal
def cmd_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--env', type=str, default="MountainCarContinuous-v0")
    parser.add_argument("--alg", type=str, default="ppo")
    parser.add_argument("--skills", type=int, default=6)
    parser.add_argument("--presteps", type=int, default=int(1e6))
    parser.add_argument("--pm", type=str, default="MLP")
    parser.add_argument("--lb", type=str, default="ba")
    parser.add_argument("--samples", type=int, default=10)
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
    
def plot_curve(env_name, algs, skills, pms, lbs, asym ,x=None, y=None, y_std=None, xlabel=None):
    print("Plotting the final results")
    colors = ['b', 'r', 'g', 'b']
    labels = algs
    ylabel = "Extrinsic Reward"
    plt.figure(figsize=conf.learning_curve_figsize)
    plt.title(f"{env_name[: env_name.index('-')]}")
    for i in range(len(algs)):
        color = colors[i]
        pm = pms[i]
        lb = lbs[i]
        alg = algs[i]
        print(f"Algorithm: {alg}")
        x_i = x[i]
        y_i = np.array(y[i])/asym
        y_std_i = np.array(y_std[i])/asym
        plt.plot(x_i, y_i, label=labels[i].upper(), color=color)
        plt.fill_between(x_i, (y_i-y_std_i), (y_i+y_std_i), color=color, alpha=0.3)
    plt.legend()
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    if len(algs) == 1:
        filename = f"{env_name}_scalability_exp_alg:{algs[0]}_x-axis:{xlabel}.png"
    else:
        filename = f"{env_name}_scalability_exp_all_algs_x-axis:{xlabel}.png"
    files_dir = f"Vis/{env_name}_cls:{pm}_lb:{lb}"
    os.makedirs(files_dir, exist_ok=True)
    plt.savefig(f'{files_dir}/{filename}', dpi=150) 
    
    
def train_all(env_params, plots_d_list, n_samples):
    plots_dict = {}
    print(f"{env_params}")
    for alg in env_params:
        for env in env_params[alg]:
            if env not in plots_dict:
                plots_dict[env] = {}
            # save a timestamp
            timestamp = time.time()
            env_dir = main_exper_dir + f"env: {env}, alg:{alg}, stamp:{timestamp}/"
            os.makedirs(env_dir, exist_ok=True)
            alg_params = hyperparams[alg]
            if alg in ['ppo', 'sac', 'pets']:
                params['n_skills'] = env_params[alg][env]['n_skills']
                params['pretrain_steps'] = env_params[alg][env]['pretrain_steps']
                print(f"stamp: {timestamp}, alg: {alg}, env: {env}, n_skills: {params['n_skills']}, pretrain_steps: {params['pretrain_steps']}")
            elif alg == "es":
                params['n_skills'] = env_params[alg][env]['n_skills']
                alg_params['iterations'] = env_params[alg][env]['pretrain_iterations']
                print(f"stamp: {timestamp}, alg: {alg}, env: {env}, n_skills: {params['n_skills']}, pretrain_iterations: {alg_params['iterations']}")
            # save_params(alg, env_dir)
            if (env_params[alg][env]).get('clip_range'):
                # clip_range
                alg_params['clip_range'] = (env_params[alg][env]).get('clip_range')
            seed_results_intr_reward = []
            seed_results_extr_reward = []
            for i in range(len(conf.seeds)):
                seed_everything(conf.seeds[i])
                seed_dir = env_dir + f"seed:{conf.seeds[i]}"
                os.makedirs(seed_dir, exist_ok=True)
                if alg in ("sac", "ppo"):
                    diayn = DIAYN(params, alg_params, discriminator_hyperparams, env, alg, seed_dir, seed=conf.seeds[i], conf=conf, timestamp=timestamp, adapt_params=sac_hyperparams, n_samples=n_samples, checkpoints=True)
                    pretrained_policy, discriminator, results_dict = diayn.pretrain()
                elif alg == "pets":
                    diayn = DIAYN_MB(params, alg_params, discriminator_hyperparams, env, alg, seed_dir, seed=conf.seeds[i], conf=conf, timestamp=timestamp, adapt_params=sac_hyperparams, n_samples=n_samples, checkpoints=True)
                    pretrained_policy, discriminator, results_dict = diayn.pretrain()
                elif alg == "es":
                    diayn = DIAYN_ES(params, alg_params, discriminator_hyperparams, env, "es", seed_dir, seed=conf.seeds[i], conf=conf, timestamp=timestamp, adapt_params=sac_hyperparams, n_samples=n_samples, checkpoints=True)
                    pretrained_policy, discriminator, results_dict = diayn.pretrain()
                asym = asymp_perofrmance[env]
                # shape: (n_seeds, length)
                seed_results_intr_reward.append(results_dict['intr_rewards'])
                seed_results_extr_reward.append(results_dict['extr_rewards'] )
                
            steps = results_dict['steps'] 
            intr_reward_mean = np.mean(seed_results_intr_reward, axis=0)
            intr_reward_std = np.std(seed_results_intr_reward, axis=0)
            extr_reward_mean = np.mean(seed_results_extr_reward, axis=0)
            extr_reward_std = np.std(seed_results_extr_reward, axis=0)
            
            if "algs" not in plots_dict[env]:
                plots_dict[env]["algs"] = []
            if "intr_reward_mean" not in plots_dict[env]:
                plots_dict[env]["intr_reward_mean"] = []
            if "intr_reward_std" not in plots_dict[env]:
                plots_dict[env]["intr_reward_std"] = []
            if "extr_reward_mean" not in plots_dict[env]:
                plots_dict[env]["extr_reward_mean"] = []
            if "extr_reward_std" not in plots_dict[env]:
                plots_dict[env]["extr_reward_std"] = []
            if "skills" not in plots_dict[env]:
                plots_dict[env]["skills"] = []
            if "pms" not in plots_dict[env]:
                plots_dict[env]["pms"] = []
            if "lbs" not in plots_dict[env]:
                plots_dict[env]["lbs"] = []
            if "steps" not in plots_dict[env]:
                plots_dict[env]["steps"] = []
            
            plots_dict[env]["algs"].append(alg)
            plots_dict[env]["intr_reward_mean"].append(intr_reward_mean)
            plots_dict[env]["intr_reward_std"].append(intr_reward_std)
            plots_dict[env]["extr_reward_mean"].append(extr_reward_mean)
            plots_dict[env]["extr_reward_std"].append(extr_reward_std)
            plots_dict[env]["skills"].append(params['n_skills'])
            plots_dict[env]["pms"].append(discriminator_hyperparams['parametrization'])
            plots_dict[env]["lbs"].append(discriminator_hyperparams['lower_bound'])
            plots_dict[env]["steps"].append(steps)
            
        plots_d_list.append(plots_dict)
            
    
if __name__ == "__main__":
    args = cmd_args()
    run_all = args.run_all
    discriminator_hyperparams['lower_bound'] = args.lb
    discriminator_hyperparams['parametrization'] = args.pm
    n_samples = args.samples
    main_exper_dir = conf.scalability_exper_dir + f"cls:{discriminator_hyperparams['parametrization']}, lb:{discriminator_hyperparams['lower_bound']}/"
    if run_all:
        # for mp
        manager = mp.Manager()
        plots_d_list = manager.list()
        n_processes = len(envs_mp[:1])
        processes_list = []
        for i in range(n_processes):
            p = mp.Process(target=train_all, args=(envs_mp[i], plots_d_list, n_samples))
            p.start()
            processes_list.append(p)
        for p in processes_list:
            p.join()
        for plot_dict in plots_d_list[1:]:
            print(plot_dict)
            input()
            env_name = list(plot_dict.keys())[0]
            algs = plot_dict[env_name]["algs"]
            skills_list = plot_dict[env_name]["skills"]
            pms = plot_dict[env_name]["pms"]
            lbs = plot_dict[env_name]["lbs"]
            intr_reward_mean = plot_dict[env_name]["intr_reward_mean"]
            intr_reward_std = plot_dict[env_name]["intr_reward_std"]
            extr_reward_mean = plot_dict[env_name]["extr_reward_mean"]
            extr_reward_std = plot_dict[env_name]["extr_reward_std"]
            steps = plot_dict[env_name]["steps"]
            asym = asymp_perofrmance[env_name]
            plot_curve(env_name, algs, skills_list, pms, lbs, asym, x=intr_reward_mean, y=extr_reward_mean, y_std=extr_reward_std, xlabel="Intrinsic Reward")
            plot_curve(env_name, algs, skills_list, pms, lbs, asym, x=steps, y=extr_reward_mean, y_std=extr_reward_std, xlabel="Pretraining Environment Steps")
        # save the results
        with open(f"{main_exper_dir}/results.txt", "w") as o:
            for i in plots_d_list:
                o.write(f"{i}\n")
    else:
        n_samples = args.samples
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
            print(f"stamp: {timestamp}, alg: {args.alg}, env: {args.env}, n_skills: {params['n_skills']}, pretrain_iterations: {alg_params['iterations']}")
        env_dir = main_exper_dir + f"env: {args.env}, alg:{args.alg}, stamp:{timestamp}/"
        os.makedirs(env_dir, exist_ok=True)
        save_params(args.alg, env_dir)
        seed_results_intr_reward = []
        seed_results_extr_reward = []
        for i in range(len(conf.seeds)):
            seed_everything(conf.seeds[i])
            seed_dir = env_dir + f"seed:{conf.seeds[i]}"
            os.makedirs(seed_dir, exist_ok=True)
            if args.alg in ("sac", "ppo"):
                diayn = DIAYN(params, alg_params, discriminator_hyperparams, args.env, args.alg, seed_dir, seed=conf.seeds[i], conf=conf, timestamp=timestamp, adapt_params=sac_hyperparams, n_samples=n_samples, checkpoints=True)
                pretrained_policy, discriminator, results_dict = diayn.pretrain()
            elif args.alg == "pets":
                diayn = DIAYN_MB(params, alg_params, discriminator_hyperparams, args.env, args.alg, seed_dir, seed=conf.seeds[i], conf=conf, timestamp=timestamp, adapt_params=sac_hyperparams, n_samples=n_samples, checkpoints=True)
                pretrained_policy, discriminator, results_dict = diayn.pretrain()
            elif args.alg == "es":
                diayn = DIAYN_ES(params, alg_params, discriminator_hyperparams, args.env, "es", seed_dir, seed=conf.seeds[i], conf=conf, timestamp=timestamp, adapt_params=sac_hyperparams, n_samples=n_samples, checkpoints=True)
                pretrained_policy, discriminator, results_dict = diayn.pretrain()
            asym = asymp_perofrmance[args.env]
            # shape: (n_seeds, length)
            seed_results_intr_reward.append(results_dict['intr_rewards'])
            seed_results_extr_reward.append(results_dict['extr_rewards'] )

        steps = results_dict['steps'] 
        intr_reward_mean = np.mean(seed_results_intr_reward, axis=0)
        intr_reward_std = np.std(seed_results_intr_reward, axis=0)
        extr_reward_mean = np.mean(seed_results_extr_reward, axis=0)
        extr_reward_std = np.std(seed_results_extr_reward, axis=0)
        print(f"steps: {steps}")
        print(f"steps: {intr_reward_mean}")
        print(f"steps: {extr_reward_mean}")
        input()
        plot_curve(args.env, [args.alg], [args.skills], [args.pm], [args.lb], asym, x=intr_reward_mean, y=extr_reward_mean, y_std=extr_reward_std, xlabel="Intrinsic Reward")
        plot_curve(args.env, [args.alg], [args.skills], [args.pm], [args.lb], asym, x=steps, y=extr_reward_mean, y_std=extr_reward_std, xlabel="Pretraining Environment Steps")
