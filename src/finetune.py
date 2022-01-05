import argparse
import pickle
import json

from src.config import conf
from src.diayn import DIAYN # diayn with model-free RL
from src.diayn_es import DIAYN_ES # diayn with evolution stratigies
from src.diayn_mb import DIAYN_MB # diayn with evolution model-based RL
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


# shared parameters
params = dict( n_skills = 30,
           pretrain_steps = int(10e6),
           finetune_steps = int(1e4),
           buffer_size = int(1e6),
           min_train_size = int(1e4),
             )


# list for multip-processing
envs_mp = [
    # 1
    {
     'ppo':{
        'MountainCarContinuous-v0': dict( 
           pretrain_steps = int(1e4), #int(25e6),
            n_skills = 3 #30
             ), # 1 dof
        },
    'sac':{
        'MountainCarContinuous-v0': dict( 
           pretrain_steps = int (1e4), #int(3e5),
            n_skills = 3 #30
             ), # 1 dof
        }
    },
    # 2
    {
     'ppo':{
        'Reacher-v2': dict( 
           pretrain_steps = int(1e4), #int(25e6),
            n_skills = 3, #30,
            clip_range = 0.01
             ), # 2 dof
        },
    'sac':{
        'Reacher-v2': dict( 
           pretrain_steps = int(1e4), #int(25e6),
            n_skills =3 #30,
             ), # 2 dof
        }
    },
    # 3
    {
     'ppo':{
        'Swimmer-v2': dict( 
           pretrain_steps = int(35e6),
            n_skills = 30
             ), # 2 dof
        },
    'sac':{
        'Swimmer-v2': dict( 
           pretrain_steps = int(5e5),
            n_skills = 30
             ), # 2 dof
        }
    },
    # 4
    {
     'ppo':{
        'Hopper-v2': dict( 
           pretrain_steps = int(100e6),
            n_skills = 30
             ), # 2 dof
        },
    'sac':{
        'Hopper-v2': dict( 
           pretrain_steps = int(1e6),
            n_skills = 30
             ), # 2 dof
        }
    },
    # 5
    {
     'ppo':{
        'HalfCheetah-v2': dict( 
           pretrain_steps = int(500e6),
            n_skills = 30
             ), # 2 dof
        },
    'sac':{
        'HalfCheetah-v2': dict( 
           pretrain_steps = int(2e6),
            n_skills = 30
             ), # 2 dof
        }
    },
    # 6
    {
     'ppo':{
        'Walker2d-v2': dict( 
           pretrain_steps = int(550e6),
            n_skills = 30
             ), # 2 dof
        },
    'sac':{
        'Walker2d-v2': dict( 
           pretrain_steps = int(2e6),
            n_skills = 30
             ), # 2 dof
        }
    },
    # 7
    {
     'ppo':{
        'Ant-v2': dict( 
           pretrain_steps = int(700e6),
            n_skills = 30
             ), # 2 dof
        },
    'sac':{
        'Ant-v2': dict( 
           pretrain_steps = int(3e6),
            n_skills = 30
             ), # 2 dof
        }
    },
    # 8
    {
     'ppo':{
        'Humanoid-v2': dict( 
           pretrain_steps = int(800e6),
            n_skills = 30
             ), # 2 dof
        },
    'sac':{
        'Humanoid-v2': dict( 
           pretrain_steps = int(5e6),
            n_skills = 30
             ), # 2 dof
        }
    },
    
    
]

# setting the parameters for each environment
env_params = {
    'ppo':{
        'MountainCarContinuous-v0': dict( 
           pretrain_steps = int(10e3),
            n_skills = 6
             ), # 1 dof
        'Reacher-v2': dict( 
           pretrain_steps = int(10e3),
            n_skills = 12,
            clip_range = 0.01
             ), # 2 dof
        'Swimmer-v2': dict( 
           pretrain_steps = int(200e6),
            n_skills = 12
             ), # 2 dof
        'Hopper-v2': dict( 
           pretrain_steps = int(200e6),
            n_skills = 15
             ), # 3 dof
        'HalfCheetah-v2': dict( 
           pretrain_steps = int(200e6),
            n_skills = 20
             ),# 6 dof
        'Walker2d-v2': dict( 
           pretrain_steps = int(300e6),
            n_skills = 20
             ),# 6 dof
        'Ant-v2': dict( 
           pretrain_steps = int(500e6),
            n_skills = 25
             ),# 8 dof
        'Humanoid-v2': dict( 
           pretrain_steps = int(800e6),
            n_skills = 25
             ),# 17 dof
    },
    'sac':{
        'MountainCarContinuous-v0': dict( 
           pretrain_steps = int(10e3),
            n_skills = 6
             ), # 1 dof
        'Reacher-v2': dict( 
           pretrain_steps = int(10e3),
            n_skills = 12
             ), # 2 dof
        'Swimmer-v2': dict( 
           pretrain_steps = int(5e6),
            n_skills = 12
             ), # 2 dof
        'Hopper-v2': dict( 
           pretrain_steps = int(6e6),
            n_skills = 15
             ), # 3 dof
        'HalfCheetah-v2': dict( 
           pretrain_steps = int(10e6),
            n_skills = 20
             ),# 6 dof
        'Walker2d-v2': dict( 
           pretrain_steps = int(10e6),
            n_skills = 20
             ),# 6 dof
        'Ant-v2': dict( 
           pretrain_steps = int(25e6),
            n_skills = 20
             ),# 8 dof
        'Humanoid-v2': dict( 
           pretrain_steps = int(50e6),
            n_skills = 20
             ),# 17 dof
    }
    
}


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
    n_actors = 5,
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
    batch_size = 64,
    ensemble_size = 5,
    trial_length = conf.max_steps/4,
    population_size = 500,
    planning_horizon = 15,
    elite_ratio = 0.1,
    num_particles = 20,
    weight_decay = 5e-5,
    algorithm = "pets"
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
    # config_d['seeds'] = str(config_d['seeds'])
    # config_d['barplot_figsize'] = str(config_d['barplot_figsize'])
    # config_d['learning_curve_figsize'] = str(config_d['learning_curve_figsize'])
    # print(config_d)
    # input()
    dictionary_data = config_d
    a_file = open(f"{directory}/configs.json", "w")
    json.dump(config_d, a_file)
    a_file.close()
    # input()
    # config_df = pd.DataFrame(config_d, index=[0])
    # config_df.to_csv(f"{directory}/configs.csv")
    print(f"{alg} hyperparame`ters: {alg_hyperparams}\n" )
    print(f"discriminator hyperparameters: {discriminator_hyperparams}\n" )
    print(f"shared experiment parameters: {params}\n" )
    print(f"configurations: {config_d }\n" )

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
             
        
def train_all(env_params, results_df_list, plots_d_list):
    plots_dict = {}
    columns = ['Algorithm', "Reward After Adaptation", "State Entropy", "Intrinsic Reward" , "Reward Before Adaptation", "Environment"]
    results_df = pd.DataFrame(columns=columns)
    #####################
    for alg in env_params:
        for env in env_params[alg]:
            if env not in plots_dict:
                plots_dict[env] = {}
            # save a timestamp
            timestamp = time.time()
            # Environment directory
            env_dir = main_exper_dir + f"env: {env}, alg:{alg}, stamp:{timestamp}/"
            os.makedirs(env_dir, exist_ok=True)
            params['n_skills'] = env_params[alg][env]['n_skills']
            params['pretrain_steps'] = env_params[alg][env]['pretrain_steps']
            print(f"stamp: {timestamp}, alg: {alg}, env: {env}, n_skills: {params['n_skills']}, pretrain_steps: {params['pretrain_steps']}")
            # save_params(alg, env_dir)
            alg_params = hyperparams[alg]
            if (env_params[alg][env]).get('clip_range'):
                # clip_range
                alg_params['clip_range'] = (env_params[alg][env]).get('clip_range')
            seed_results = []
            for i in range(len(conf.seeds)):
                seed_everything(conf.seeds[i])
                seed_dir = env_dir + f"seed:{conf.seeds[i]}"
                os.makedirs(seed_dir, exist_ok=True)
                if alg in ("sac", "ppo"):
                    # TODO add ES, when it is ready
                    diayn = DIAYN(params, alg_params, discriminator_hyperparams, env, alg, seed_dir, seed=conf.seeds[i], conf=conf, timestamp=timestamp)
                    # pretraining step
                    pretrained_policy, discriminator = diayn.pretrain()
                    # fine-tuning step 
                    adapted_policy, best_skill = diayn.finetune()
                    # Evaluate the policy 
                    intrinsic_reward_mean, reward_beforeFinetune_mean, reward_mean, entropy_mean = evaluate(env, params['n_skills'], pretrained_policy, adapted_policy, discriminator, 
                                                                                                        discriminator_hyperparams['parametrization'], best_skill)
                elif alg == "pets":
                    # (self, params, alg_params, discriminator_hyperparams, env="MountainCarContinuous-v0", alg="ppo", directory="./", seed=10, device="cpu", conf=None, timestamp=None, checkpoints=False, args=None, task=None):
                    diayn = DIAYN_MB(params, alg_params, discriminator_hyperparams, env, alg, seed_dir, seed=conf.seeds[i], conf=conf, timestamp=timestamp)
                    # pretraining step
                    pretrained_model, discriminator = diayn.pretrain()
                    # fine-tune step
                    adapted_policy, best_skill = diayn.finetune()
                    # Evaluate the policy 
                    # intrinsic_reward_mean, reward_beforeFinetune_mean, reward_mean, entropy_mean = pass
                
                threshold = gym.make(env).spec.reward_threshold
                # save results
                seed_results.append([intrinsic_reward_mean, reward_beforeFinetune_mean/threshold, reward_mean/threshold, entropy_mean])
                d = {'Algorithm': alg.upper(), 
                     "Reward After Adaptation": reward_mean/threshold, 
                     "Intrinsic Reward": intrinsic_reward_mean, 
                     "State Entropy": entropy_mean, 
                     "Reward Before Adaptation": reward_beforeFinetune_mean,
                     "Environment": env[: env.index('-')] if env != "MountainCarContinuous-v0" else "MountainCar"
                    }
                results_df = results_df.append(d, ignore_index=True)
            # add to the plot dict
            if "algs" not in plots_dict[env]:
                plots_dict[env]["algs"] = []
            if "stamps" not in plots_dict[env]:
                plots_dict[env]["stamps"] = []
            if "skills" not in plots_dict[env]:
                plots_dict[env]["skills"] = []
            if "pms" not in plots_dict[env]:
                plots_dict[env]["pms"] = []
            if "lbs" not in plots_dict[env]:
                plots_dict[env]["lbs"] = []
                
            plots_dict[env]["algs"].append(alg)
            plots_dict[env]["stamps"].append(timestamp)
            plots_dict[env]["skills"].append(params['n_skills'])
            plots_dict[env]["pms"].append(discriminator_hyperparams['parametrization'])
            plots_dict[env]["lbs"].append(discriminator_hyperparams['lower_bound'])
        save_final_results(seed_results, env_dir)
        results_df_list.append(results_df)
        plots_d_list.append(plots_dict)
    
    
def df_from_list(df_list, columns):
    results_df = pd.concat(df_list,ignore_index=True) 
    return results_df
    
    
if __name__ == "__main__":
    args = cmd_args()
    run_all = args.run_all
    discriminator_hyperparams['lower_bound'] = args.lb
    discriminator_hyperparams['parametrization'] = args.pm
    main_exper_dir = None
    main_exper_dir = conf.log_dir_finetune + f"cls:{discriminator_hyperparams['parametrization']}, lb:{discriminator_hyperparams['lower_bound']}/"
    
    columns = ['Algorithm', "Environment", "Reward After Adaptation", "State Entropy", "Intrinsic Reward" , "Reward Before Adaptation"]
        
    if run_all:
        # for mp
        manager = mp.Manager()
        results_df_list = manager.list()
        plots_d_list = manager.list()

        n_processes = len(envs_mp[:2])
        processes_list = []
        for i in range(n_processes):
            p = mp.Process(target=train_all, args=(envs_mp[i], results_df_list, plots_d_list))
            p.start()
            processes_list.append(p)
        for p in processes_list:
            p.join()
        print(f"results_df_list: {results_df_list}")
        print(f"plots_d_list: {plots_d_list}")
        for plot_dict in plots_d_list:
            env_name = list(plot_dict.keys())[0]
            algs = plot_dict[env_name]['algs']
            stamps = plot_dict[env_name]['stamps']
            skills_list = plot_dict[env_name]['skills']
            pms = plot_dict[env_name]['pms']
            lbs = plot_dict[env_name]['lbs']
            plot_learning_curves(env_name, algs, stamps, skills_list, pms, lbs)
        # print(results_df_list)
        results_df = df_from_list(results_df_list, columns)
        # print(results_df)
        for col in columns[2:]:
            plt.figure(figsize=conf.barplot_figsize)
            y_axis = col
            # ax = sns.barplot(x="Algorithm", y=y_axis, data=results_df, hue="Environment")
            if col == "Reward After Adaptation":
                ax = sns.barplot(x="Environment", y=y_axis, data=results_df, hue="Algorithm", hue_order=['ppo'.upper(), 'sac'.upper()])
                sns.move_legend(ax, "lower left", title='Algorithm')
            else:
                ax = sns.barplot(x="Environment", y=y_axis, data=results_df, hue="Algorithm", hue_order=['ppo'.upper(), 'sac'.upper()])
                ax.legend_.remove()
            filename = f"adaptation_experiment_final_{y_axis}.png"
            files_dir = f"Vis/adaptation_experiment_cls:{discriminator_hyperparams['parametrization']}, lb:{discriminator_hyperparams['lower_bound']}/"
            os.makedirs(files_dir, exist_ok=True)
            plt.savefig(files_dir + filename)
    else:
        # save a timestamp
        timestamp = time.time()
        print(f"Experiment timestamp: {timestamp}")
        params['pretrain_steps'] = args.presteps
        params['n_skills'] = args.skills
        env_dir = main_exper_dir + f"env: {args.env}, alg:{args.alg}, stamp:{timestamp}/"
        os.makedirs(env_dir, exist_ok=True)
        save_params(args.alg, env_dir)
        alg_params = hyperparams[args.alg]
        seed_results = []
        for i in range(len(conf.seeds)):
            seed_everything(conf.seeds[i])
            seed_dir = env_dir + f"seed:{conf.seeds[i]}"
            os.makedirs(seed_dir, exist_ok=True)
            if args.alg in ("sac", "ppo"):
                # TODO add ES, when it is ready
                diayn = DIAYN(params, alg_params, discriminator_hyperparams, args.env, args.alg, seed_dir, seed=conf.seeds[i], conf=conf, timestamp=timestamp)
                # pretraining step
                pretrained_policy, discriminator = diayn.pretrain()
                # fine-tuning step 
                adapted_policy, best_skill = diayn.finetune()
                # Evaluate the policy 
                intrinsic_reward_mean, reward_beforeFinetune_mean, reward_mean, entropy_mean = evaluate(env, params['n_skills'], pretrained_policy, adapted_policy, discriminator, 
                                                                                                    discriminator_hyperparams['parametrization'], best_skill)
            elif args.alg == "pets":
                diayn = DIAYN_MB(params, alg_params, discriminator_hyperparams, args.env, args.alg, seed_dir, seed=conf.seeds[i], conf=conf, timestamp=timestamp)
                # pretraining step
                pretrained_model, discriminator = diayn.pretrain()
                # fine-tune step
                adapted_policy, best_skill = diayn.finetune()
                # Evaluate the policy 
                # intrinsic_reward_mean, reward_beforeFinetune_mean, reward_mean, entropy_mean = pass
            seed_results.append([intrinsic_reward_mean, reward_beforeFinetune_mean, reward_mean, entropy_mean])
        save_final_results(seed_results, env_dir)
        
        
        
        



    

