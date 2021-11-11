import gym
import random
from stable_baselines3.common import callbacks

import torch
import torch.nn as nn

import argparse

from torch.utils.tensorboard import SummaryWriter

import numpy as np
import matplotlib.pyplot as plt

from collections import deque
import random

from stable_baselines3 import SAC, sac
from stable_baselines3.common.callbacks import  EvalCallback
from stable_baselines3.common.vec_env import DummyVecEnv
from stable_baselines3.common.monitor import Monitor

import time

from src.models.models import Discriminator
from src.replayBuffers import DataBuffer
from src.config import conf
from src.callbacks.callbacks import DiscriminatorCallback, VideoRecorderCallback
from src.environment_wrappers.env_wrappers import RewardWrapper, SkillWrapperVideo, SkillWrapper

from typing import Any, Dict

import pandas as pd



# SAC Hyperparameters 
sac_hyperparams = dict(
    learning_rate = 3e-4,
    gamma = 0.99,
    batch_size = 128,
    buffer_size = int(1e7),
    tau = 0.01,
    gradient_steps = 1,
    ent_coef=0.1,
    learning_starts = 10000
)

# Discriminator Hyperparameters
discriminator_hyperparams = dict(
    learning_rate = 3e-4,
    batch_size = 128,
    n_epochs = 1,
)

# set the seed
seed = 1
random.seed(seed)
np.random.seed(seed)
torch.random.manual_seed(seed)

# timestamp
timestamp = time.time()

def cmd_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--env', type=str, default="MountainCar-v0")
    parser.add_argument("--r", type=str, default="False")
    args = parser.parse_args()
    return args


def run_experiment(args):
    # create the discirminator and the buffer
    d = Discriminator(gym.make(args.env).observation_space.shape[0], [conf.layer_size_discriminator, conf.layer_size_discriminator], conf.n_z).to(conf.device)

    # tensorboard summary writer
    sw = SummaryWriter(log_dir=conf.log_dir + f"/sac_{args.env}_{timestamp}", comment=f"SAC Discriminator, env_name:{args.env}")
    
    # save the hyperparams 
    sac_hyperparams_df = pd.DataFrame(sac_hyperparams, index=[0])
    sac_hyperparams_df.to_csv(conf.log_dir + f"sac_{args.env}_{timestamp}/sac_hyperparams.csv")
    discriminator_hyperparams_df = pd.DataFrame(discriminator_hyperparams, index=[0])
    discriminator_hyperparams_df.to_csv(conf.log_dir + f"sac_{args.env}_{timestamp}/discriminator_hyperparams.csv")
    

    # create the environment # monitor_dir=testing_log_dir
    env = DummyVecEnv([lambda: Monitor(RewardWrapper(SkillWrapper(gym.make(args.env), conf.n_z, max_steps=conf.max_steps), d, conf.n_z), conf.log_dir + f"/sac_{args.env}_{timestamp}")])
    
    
    
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
                tensorboard_log = conf.log_dir + f"/sac_{args.env}_{timestamp}",
                )
    

    # Create Callbacks
    # save_best_model_callback = SaveOnBestTrainingRewardCallback(check_freq=100, log_dir=log_dir, verbose=0)
    video_loging_callback = VideoRecorderCallback(args.env, record_freq=1000, deterministic=d, n_z=conf.n_z, videos_dir=conf.videos_dir + f"/sac_{args.env}_{timestamp}")
    # evaluation_callback = EvaluationCallBack(env_name, eval_freq=500, n_evals=eval_runs, log_dir=log_dir)
    discriminator_callback = DiscriminatorCallback(d, None, discriminator_hyperparams, sw=sw, n_skills=conf.n_z, min_buffer_size=conf.min_train_size, save_dir=conf.log_dir + f"/sac_{args.env}_{timestamp}", on_policy=False)

    eval_env =  RewardWrapper(SkillWrapper(gym.make(args.env), conf.n_z), d, conf.n_z)
    eval_env = Monitor(eval_env, conf.log_dir + f"/sac_{args.env}_{timestamp}" + f"/eval_results")
    
    eval_callback = EvalCallback(eval_env, best_model_save_path=conf.log_dir + f"/sac_{args.env}_{timestamp}",
                                log_path=conf.log_dir + f"/sac_{args.env}_{timestamp}", eval_freq=1000,
                                deterministic=True, render=False)


    # if args.r == "True":
    #   print("Load saved model to resume training")
    #   d.load_state_dict(torch.load(conf.log_dir + f"/sac_{args.env}" +  "/disc.pth", map_location=torch.device(conf.device)), )
      # model.load(conf.log_dir + f"/sac_{args.env}" + "/best_model")

    # train the agent
    model.learn(total_timesteps=conf.total_timesteps, log_interval=1, callback=[video_loging_callback, discriminator_callback, eval_callback])



if __name__ == "__main__":
    args = cmd_args()
    run_experiment(args)
    

