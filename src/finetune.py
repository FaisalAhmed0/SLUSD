import gym
import argparse

from stable_baselines3.common import callbacks
from stable_baselines3.common.vec_env.dummy_vec_env import DummyVecEnv
from stable_baselines3 import PPO, SAC
from stable_baselines3.common.callbacks import  EvalCallback
from stable_baselines3.common.utils import get_schedule_fn
from stable_baselines3.common.monitor import Monitor

from config import conf
from env_wrappers import SkillWrapperFinetune, RewardWrapper, SkillWrapper
from utils import record_video_finetune, best_skill
from models import Discriminator
from replayBuffers import DataBuffer
from callbacks import DiscriminatorCallback, VideoRecorderCallback

import torch
import torch.nn as nn
from torch.utils.tensorboard import SummaryWriter

import numpy as np
import pandas as pd

import time
import os
import random

# set the seed
seed = 10
random.seed(seed)
np.random.seed(seed)
torch.random.manual_seed(seed)



# shared parameters
params = dict( n_skills = 6,
           pretrain_steps = int(2e6),
           finetune_steps = int(1e5),
           buffer_size = int(1e7),
           min_train_size = int(1e4)
             )


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
    ent_coef=0.1,
    learning_starts = 10000,
    algorithm = "sac"
)

# Discriminator Hyperparameters
discriminator_hyperparams = dict(
    learning_rate = 3e-4,
    batch_size = 64,
    n_epochs = 1
)

# save a timestamp
timestamp = time.time()


# Extract arguments from terminal
def cmd_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--env', type=str, default="MountainCarContinuous-v0")
    parser.add_argument("--alg", type=str, default="ppo")
    args = parser.parse_args()
    return args

# save the hyperparameters
def save_params(args, directory):
    # save the hyperparams in a csv files
    alg_hyperparams = ppo_hyperparams if args.alg == "ppo" else sac_hyperparams
    
    alg_hyperparams_df = pd.DataFrame(alg_hyperparams, index=[0])
    alg_hyperparams_df.to_csv(f"{directory}/{args.alg}_hyperparams.csv")
    
    discriminator_hyperparams_df = pd.DataFrame(discriminator_hyperparams, index=[0])
    discriminator_hyperparams_df.to_csv(f"{directory}/discriminator_hyperparams.csv")
    
    # convert the config namespace to a dictionary
    exp_params_df = pd.DataFrame(params, index=[0])
    exp_params_df.to_csv(f"{directory}/params.csv")
    
    # general configrations
    config_d = vars(conf)
    config_df = pd.DataFrame(config_d, index=[0])
    config_df.to_csv(f"{directory}/configs.csv")
    

# perform the pretraining step with DIAYN algorithm
def pretrain(args, directory):
    # create the discirminator and the buffer
    d = Discriminator(gym.make(args.env).observation_space.shape[0], [conf.layer_size_discriminator, conf.layer_size_discriminator], params['n_skills']).to(conf.device)
    buffer = DataBuffer(params['buffer_size'], obs_shape=gym.make(args.env).observation_space.shape[0])
    
    # tensorboard summary writer
    sw = SummaryWriter(log_dir=directory, comment=f"{args.alg} Discriminator, env_name:{args.env}")
    
    if args.alg == "ppo":
        env = DummyVecEnv([
                    lambda: Monitor(RewardWrapper(SkillWrapper(gym.make(args.env), params['n_skills'], max_steps=conf.max_steps), d, params['n_skills']),  directory),
                    lambda: Monitor(RewardWrapper(SkillWrapper(gym.make(args.env), params['n_skills'], max_steps=conf.max_steps), d, params['n_skills']),  directory),
                    lambda: Monitor(RewardWrapper(SkillWrapper(gym.make(args.env), params['n_skills'], max_steps=conf.max_steps), d, params['n_skills']),  directory),
                    lambda: Monitor(RewardWrapper(SkillWrapper(gym.make(args.env), params['n_skills'], max_steps=conf.max_steps), d, params['n_skills']),  directory)])
        
        # create the model with the speicifed hyperparameters        
        model = PPO('MlpPolicy', env, verbose=1, 
                learning_rate = ppo_hyperparams['learning_rate'], 
                n_steps = ppo_hyperparams['n_steps'], 
                batch_size = ppo_hyperparams['batch_size'],
                n_epochs = ppo_hyperparams['n_epochs'],
                gamma = ppo_hyperparams['gamma'],
                gae_lambda = ppo_hyperparams['gae_lambda'],
                clip_range = ppo_hyperparams['clip_range'],
                policy_kwargs = dict(activation_fn=nn.Tanh,
                                net_arch=[dict(pi=[conf.layer_size_policy, conf.layer_size_policy], vf=[conf.layer_size_value, conf.layer_size_value])]),
                tensorboard_log = directory,
                seed = seed
                )
        
        # Create Callbacks
        video_loging_callback = VideoRecorderCallback(args.env, record_freq=5000, deterministic=d, n_z=params['n_skills'], videos_dir=conf.videos_dir +f"ppo_{args.env}_{timestamp}")
        # evaluation_callback = EvaluationCallBack(env_name, eval_freq=500, n_evals=eval_runs, log_dir=log_dir)
        discriminator_callback = DiscriminatorCallback(d, buffer, discriminator_hyperparams, sw=sw, n_skills=params['n_skills'], min_buffer_size=params['min_train_size'], save_dir=directory, on_policy=True)

        eval_env =  RewardWrapper(SkillWrapper(gym.make(args.env), params['n_skills']), d, params['n_skills'])
        eval_env = Monitor(eval_env, f"{directory}/eval_results")
        eval_callback = EvalCallback(eval_env, best_model_save_path=directory,
                                    log_path=f"{directory}/eval_results", eval_freq=1000,
                                    deterministic=True, render=False)
        # train the agent
        model.learn(total_timesteps=params['pretrain_steps'], callback=[ discriminator_callback, eval_callback], log_interval=1, tb_log_name="PPO Pretrain")
        
    elif args.alg == "sac":
        env = DummyVecEnv([lambda: Monitor(RewardWrapper(SkillWrapper(gym.make(args.env), params['n_skills'], max_steps=conf.max_steps), d, params['n_skills']), directory)])
        # create the model with the speicifed hyperparameters
        model = SAC('MlpPolicy', env, verbose=1, 
                    learning_rate = sac_hyperparams['learning_rate'], 
                    batch_size = sac_hyperparams['batch_size'],
                    gamma = sac_hyperparams['gamma'],
                    buffer_size = sac_hyperparams['buffer_size'],
                    tau = sac_hyperparams['tau'],
                    ent_coef = sac_hyperparams['ent_coef'],
                    gradient_steps = sac_hyperparams['gradient_steps'],
                    learning_starts = sac_hyperparams['learning_starts'],
                    policy_kwargs = dict(net_arch=dict(pi=[conf.layer_size_policy, conf.layer_size_policy], qf=[conf.layer_size_value, conf.layer_size_value])),
                    tensorboard_log = directory,
                    seed = seed
                    )
        
        # Create Callbacks
        video_loging_callback = VideoRecorderCallback(args.env, record_freq=1000, deterministic=d, n_z=params['n_skills'], videos_dir=conf.videos_dir + f"/sac_{args.env}_{timestamp}")
        discriminator_callback = DiscriminatorCallback(d, None, discriminator_hyperparams, sw=sw, n_skills=params['n_skills'], min_buffer_size=params['min_train_size'], save_dir=directory, on_policy=False)

        eval_env =  RewardWrapper(SkillWrapper(gym.make(args.env), params['n_skills']), d, params['n_skills'])
        eval_env = Monitor(eval_env,  f"{directory}/eval_results")

        eval_callback = EvalCallback(eval_env, best_model_save_path=directory, log_path=directory, eval_freq=1000, deterministic=True, render=False)

        # train the agent
        model.learn(total_timesteps=params['pretrain_steps'], callback=[ discriminator_callback, eval_callback], log_interval=1, tb_log_name="SAC Pretrain")

# finetune the pretrained policy on a specific task
def finetune(args, directory):
    env = DummyVecEnv([lambda: SkillWrapper(gym.make(args.env), params['n_skills'], max_steps=conf.max_steps)])
    model_dir = directory + "/best_model"
    if args.alg == "sac":
        model = SAC.load(model_dir, env=env, seed=seed)
    elif args.alg == "ppo":
        model = PPO.load(model_dir, env=env,  clip_range= get_schedule_fn(ppo_hyperparams['clip_range']), seed=seed)
    
    best_skill_index = best_skill(model, args.env,  params['n_skills'])
    # env_finetune = DummyVecEnv([lambda: SkillWrapperFinetune(gym.make(args.env), params['n_skills'], max_steps=conf.max_steps, skill=best_skill_index)])
    
    del model
    if args.alg == "sac":
        model = SAC('MlpPolicy', env, verbose=1, 
                    learning_rate = sac_hyperparams['learning_rate'], 
                    batch_size = sac_hyperparams['batch_size'],
                    gamma = sac_hyperparams['gamma'],
                    buffer_size = sac_hyperparams['buffer_size'],
                    tau = sac_hyperparams['tau'],
                    gradient_steps = sac_hyperparams['gradient_steps'],
                    learning_starts = sac_hyperparams['learning_starts'],
                    policy_kwargs = dict(net_arch=dict(pi=[conf.layer_size_policy, conf.layer_size_policy], qf=[conf.layer_size_value, conf.layer_size_value])),
                    tensorboard_log = directory,
                    )
    elif args.alg == "ppo":
        model = PPO('MlpPolicy', env, verbose=1, 
                learning_rate=ppo_hyperparams['learning_rate'], 
                n_steps=ppo_hyperparams['n_steps'], 
                batch_size=ppo_hyperparams['batch_size'],
                n_epochs=ppo_hyperparams['n_epochs'],
                gamma=ppo_hyperparams['gamma'],
                gae_lambda=ppo_hyperparams['gae_lambda'],
                clip_range=ppo_hyperparams['clip_range'],
                policy_kwargs = dict(activation_fn=nn.Tanh,
                                net_arch=[dict(pi=[conf.layer_size_policy, conf.layer_size_policy], vf=[conf.layer_size_value, conf.layer_size_value])]),
                tensorboard_log = directory,

                )

    env = DummyVecEnv([lambda: SkillWrapperFinetune( Monitor(gym.make(args.env),  f"{directory}/finetune_train_results" ), params['n_skills'],max_steps=gym.make(args.env)._max_episode_steps, skill=best_skill_index) ])
    
    eval_env = SkillWrapperFinetune(Monitor(gym.make(args.env),  f"{directory}/finetune_eval_results"), params['n_skills'], max_steps=gym.make(args.env)._max_episode_steps, skill=best_skill_index)
    eval_env = Monitor(eval_env, f"{directory}/finetune_eval_results")
    eval_callback = EvalCallback(eval_env, best_model_save_path=directory + f"/best_finetuned_model_skillIndex:{best_skill_index}",
                                log_path=directory, eval_freq=1000,
                                deterministic=True, render=False)
    if args.alg == "sac":
        model = SAC.load(model_dir, env = env, tensorboard_log = directory)
        model.learn(total_timesteps=params['finetune_steps'], callback=eval_callback, tb_log_name="SAC_FineTune")
    elif args.alg == "ppo":
        model = PPO.load(model_dir, env = env, tensorboard_log = directory, clip_range= get_schedule_fn(ppo_hyperparams['clip_range']))
        model.learn(total_timesteps=params['finetune_steps'], callback=eval_callback, tb_log_name="PPO_FineTune")
    

    # record a video of the agent
    # record_video_finetune(args.env, best_skill_index, model, params['n_skills'] ,video_length=1000, video_folder='videos_finetune/', alg=args.alg)




if __name__ == "__main__":
    print(f"Experiment timestamp: {timestamp}")
    args = cmd_args()
    # experiment directory
    exp_directory = conf.log_dir_finetune + f"{args.alg}_{args.env}_{timestamp}"
    # create a folder for the expirment
    os.makedirs(exp_directory)
    # save the exp parameters
    save_params(args, exp_directory)
    # pretraining step
    pretrain(args, exp_directory)
    # fine-tuning step 
    finetune(args, exp_directory)



    

