
import argparse

from src.config import conf
from src.diayn import DIAYN # diayn with model-free RL
from src.diayn_es import DIAYN_ES # diayn with evolution star
from src.utils import seed_everything, evaluate_pretrained_policy_ext, evaluate_adapted_policy, evaluate_state_coverage, evaluate_pretrained_policy_intr

import torch

import numpy as np
import pandas as pd

import time
import os
import random


# shared parameters
params = dict( n_skills = 6,
           pretrain_steps = int(5e3),
           finetune_steps = int(5e3),
           buffer_size = int(1e6),
           min_train_size = int(1e4),
             )


# setting the parameters for each environment
# env_params = {
#     'ppo':{
#         'MountainCarContinuous-v0': dict( 
#            pretrain_steps = int(200e6),
#             n_skills = 6
#              ), # 1 dof
#         'Reacher-v2': dict( 
#            pretrain_steps = int(220e6),
#             n_skills = 12,
#             clip_range = 0.01
#              ), # 2 dof
#         'Swimmer-v2': dict( 
#            pretrain_steps = int(200e6),
#             n_skills = 12
#              ), # 2 dof
#         'Hopper-v2': dict( 
#            pretrain_steps = int(200e6),
#             n_skills = 15
#              ), # 3 dof
#         'HalfCheetah-v2': dict( 
#            pretrain_steps = int(200e6),
#             n_skills = 20
#              ),# 6 dof
#         'Walker2d-v2': dict( 
#            pretrain_steps = int(300e6),
#             n_skills = 20
#              ),# 6 dof
#         'Ant-v2': dict( 
#            pretrain_steps = int(500e6),
#             n_skills = 25
#              ),# 8 dof
#         'Humanoid-v2': dict( 
#            pretrain_steps = int(800e6),
#             n_skills = 25
#              ),# 17 dof
#     },
#     'sac':{
#         'MountainCarContinuous-v0': dict( 
#            pretrain_steps = int(5e6),
#             n_skills = 6
#              ), # 1 dof
#         'Reacher-v2': dict( 
#            pretrain_steps = int(5e6),
#             n_skills = 12
#              ), # 2 dof
#         'Swimmer-v2': dict( 
#            pretrain_steps = int(5e6),
#             n_skills = 12
#              ), # 2 dof
#         'Hopper-v2': dict( 
#            pretrain_steps = int(6e6),
#             n_skills = 15
#              ), # 3 dof
#         'HalfCheetah-v2': dict( 
#            pretrain_steps = int(10e6),
#             n_skills = 20
#              ),# 6 dof
#         'Walker2d-v2': dict( 
#            pretrain_steps = int(10e6),
#             n_skills = 20
#              ),# 6 dof
#         'Ant-v2': dict( 
#            pretrain_steps = int(25e6),
#             n_skills = 20
#              ),# 8 dof
#         'Humanoid-v2': dict( 
#            pretrain_steps = int(50e6),
#             n_skills = 20
#              ),# 17 dof
#     }
    
# }


'''
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
    ent_coef=0.4,
    learning_starts = 10000,
    algorithm = "sac"
)
'''
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

# Discriminator Hyperparameters
# weight_decay=0.0, label_smoothing=None, gp=None, mixup=False
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
    alpha_logit = -5. # small value of alpha => reduce the variance of nwj by introduce some nce bias
    
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
    
    alg_hyperparams_df = pd.DataFrame(alg_hyperparams, index=[0])
    alg_hyperparams_df.to_csv(f"{directory}/{alg}_hyperparams.csv")
    
    discriminator_hyperparams_df = pd.DataFrame(discriminator_hyperparams, index=[0])
    discriminator_hyperparams_df.to_csv(f"{directory}/discriminator_hyperparams.csv")
    
    # convert the config namespace to a dictionary
    exp_params_df = pd.DataFrame(params, index=[0])
    exp_params_df.to_csv(f"{directory}/params.csv")
    
    # general configrations
    config_d = vars(conf)
    config_df = pd.DataFrame(config_d, index=[0])
    config_df.to_csv(f"{directory}/configs.csv")
    print(f"{alg} hyperparameters: {alg_hyperparams}" )
    print(f"discriminator hyperparameters: {discriminator_hyperparams}" )
    print(f"shared experiment parameters: {params}" )
    print(f"configurations: {config_d }" )
    # input()

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
        
def evaluate(env, n_skills, pretrained_policy, adapted_policy, discriminator, parametrization, bestskill):
    # intrinsic reward
    intrinsic_reward_mean = np.mean([evaluate_pretrained_policy_intr(env, n_skills, pretrained_policy, discriminator, parametrization) for _ in range(10)])
    # best skill reward before adaptation
    reward_beforeFinetune_mean = np.mean([evaluate_pretrained_policy_ext(env, n_skills, pretrained_policy) for _ in range(10)])
    # reward after adaptation
    reward_mean = np.mean([evaluate_adapted_policy(env, n_skills, bestskill, adapted_policy) for _ in range(10)])
    # entropy
    entropy_mean = np.mean([evaluate_state_coverage(env, n_skills, pretrained_policy) for _ in range(10)])
    return intrinsic_reward_mean, reward_beforeFinetune_mean, reward_mean, entropy_mean

def save_final_results(all_results, env_dir):
    results_mean = np.mean(all_results, axis=0)
    results_std = np.std(all_results, axis=0)
    results_dic = dict(
                intrinsic_reward_mean = results_mean[0],
                intrinsic_reward_std = results_std[0],
                reward_beforeFinetune_mean = results_mean[1],
                reward_beforeFinetune_std = results_std[1],
                reward_mean = results_mean[2],
                reward_std = results_std[2],
                entropy_mean = results_mean[3],
                entropy_std = results_std[3],

    )
    df =pd.DataFrame(results_dic, index=[0])
    df.to_csv(f"{env_dir}/final_results.csv")
    print(df)
    
    
    
        

if __name__ == "__main__":
    seeds = [0, 10, 1234, 5, 42]
    args = cmd_args()
    run_all = args.run_all
    main_exper_dir = None
    main_exper_dir = conf.log_dir_finetune + f"cls:{discriminator_hyperparams['parametrization']}, lb:{discriminator_hyperparams['lower_bound']}/"
    discriminator_hyperparams['lower_bound'] = args.lb
    discriminator_hyperparams['parametrization'] = args.pm
    if run_all:
        for alg in env_params:
            for env in env_params[alg]:
                # save a timestamp
                timestamp = time.time()
                # Environment directory
                env_dir = main_exper_dir + f"env: {env}, alg:{alg}, stamp:{timestamp}/"
                os.makedirs(env_dir, exist_ok=True)
                params['n_skills'] = env_params[alg][env]['n_skills']
                params['pretrain_steps'] = env_params[alg][env]['pretrain_steps']
                print(f"stamp: {timestamp}, alg: {alg}, env: {env}, n_skills: {params['n_skills']}, pretrain_steps: {params['pretrain_steps']}")
                save_params(alg, env_dir)
                alg_params = hyperparams[alg]
                if (env_params[alg][env]).get('clip_range'):
                    # clip_range
                    alg_params['clip_range'] = (env_params[alg][env]).get('clip_range')
                seed_results = []
                for i in range(len(seeds)):
                    seed_everything(seeds[i])
                    seed_dir = env_dir + f"seed:{seeds[i]}"
                    os.makedirs(seed_dir, exist_ok=True)
                    # TODO add ES, when it is ready
                    diayn = DIAYN(params, alg_params, discriminator_hyperparams, env, alg, seed_dir, seed=seeds[i], conf=conf, timestamp=timestamp)
                    # pretraining step
                    pretrained_policy, discriminator = diayn.pretrain()
                    # fine-tuning step 
                    adapted_policy, best_skill = diayn.finetune()
                    # Evaluate the policy 
                    intrinsic_reward_mean, reward_beforeFinetune_mean, reward_mean, entropy_mean = evaluate(env, n_skills, pretrained_policy, adapted_policy, discriminator, 
                                                                                                            discriminator_hyperparams['parametrization'], best_skill)
                    # save results
                    seed_results.append([intrinsic_reward_mean, reward_beforeFinetune_mean, reward_mean, entropy_mean])
                save_final_results(seed_results, env_dir)
    else:
        discriminator_hyperparams['lower_bound'] = args.lb
        discriminator_hyperparams['parametrization'] = args.pm
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
        
        for i in range(len(seeds)):
            seed_everything(seeds[i])
            seed_dir = env_dir + f"seed:{seeds[i]}"
            os.makedirs(seed_dir, exist_ok=True)
            # save the exp parameters
            # create a diyan object
            diayn = DIAYN(params, alg_params, discriminator_hyperparams, args.env, args.alg, seed_dir, seed=seeds[i], conf=conf, timestamp=timestamp)
            # pretraining step
            pretrained_policy, discriminator = diayn.pretrain()
            # fine-tuning step 
            adapted_policy, bestskill = diayn.finetune()
            # save the final results
            intrinsic_reward_mean, reward_beforeFinetune_mean, reward_mean, entropy_mean = evaluate(args.env, params['n_skills'], pretrained_policy, adapted_policy, discriminator, 
                                                                                                            discriminator_hyperparams['parametrization'], bestskill)
            seed_results.append([intrinsic_reward_mean, reward_beforeFinetune_mean, reward_mean, entropy_mean])
            
        save_final_results(seed_results, env_dir)



    

