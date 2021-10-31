import gym
import random

import torch
import torch.nn as nn

import argparse

from torch.utils.tensorboard import SummaryWriter

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

from collections import deque
import random

from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import  EvalCallback
from stable_baselines3.common.vec_env import DummyVecEnv
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.env_util import make_vec_env

from models import Discriminator
from replayBuffers import DataBuffer
from config import conf
from callbacks import DiscriminatorCallback, VideoRecorderCallback
from env_wrappers import RewardWrapper, SkillWrapperVideo, SkillWrapper

from typing import Any, Dict

import time


# PPO Hyperparameters 
ppo_hyperparams = dict(
    learning_rate = 3e-4,
    gamma = 0.99,
    batch_size = 64,
    n_epochs = 10,
    gae_lambda = 0.95,
    n_steps = 2048,
    clip_range = 0.1,
    ent_coef=0.1,
)

# Discriminator Hyperparameters
discriminator_hyperparams = dict(
    learning_rate = 3e-4,
    batch_size = 64,
    n_epochs = 10,
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
    buffer = DataBuffer(conf.buffer_size, obs_shape=gym.make(args.env).observation_space.shape[0])
    
    # tensorboard summary writer
    sw = SummaryWriter(log_dir=conf.log_dir + f"ppo_{args.env}_{timestamp}", comment=f"PPO Discriminator, env_name:{args.env}")
    # save the hyperparams 
    ppo_hyperparams_df = pd.DataFrame(ppo_hyperparams, index=[0])
    ppo_hyperparams_df.to_csv(conf.log_dir + f"ppo_{args.env}_{timestamp}/ppo_hyperparams.csv")
    discriminator_hyperparams_df = pd.DataFrame(discriminator_hyperparams, index=[0])
    discriminator_hyperparams_df.to_csv(conf.log_dir + f"ppo_{args.env}_{timestamp}/discriminator_hyperparams.csv")
    # create the environment # monitor_dir=testing_log_dir
    
    
    env = DummyVecEnv([lambda: Monitor(RewardWrapper(SkillWrapper(gym.make(args.env), conf.n_z, max_steps=conf.max_steps), d, conf.n_z), conf.log_dir + f"/ppo_{args.env}_{timestamp}"),
    lambda: Monitor(RewardWrapper(SkillWrapper(gym.make(args.env), conf.n_z, max_steps=conf.max_steps), d, conf.n_z), conf.log_dir + f"/ppo_{args.env}_{timestamp}"),
    lambda: Monitor(RewardWrapper(SkillWrapper(gym.make(args.env), conf.n_z, max_steps=conf.max_steps), d, conf.n_z), conf.log_dir + f"/ppo_{args.env}_{timestamp}"),
    lambda: Monitor(RewardWrapper(SkillWrapper(gym.make(args.env), conf.n_z, max_steps=conf.max_steps), d, conf.n_z), conf.log_dir + f"/ppo_{args.env}_{timestamp}")])
    

    # env = Monitor(env, conf.log_dir + f"/ppo_{args.env}")
    # env = gym.make(env_name)


    # create the model with the speicifed hyperparameters
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
                tensorboard_log = conf.log_dir + f"/ppo_{args.env}_{timestamp}",

                )
    

    # Create Callbacks
    # save_best_model_callback = SaveOnBestTrainingRewardCallback(check_freq=100, log_dir=log_dir, verbose=0)
    video_loging_callback = VideoRecorderCallback(args.env, record_freq=5000, deterministic=d, n_z=conf.n_z, videos_dir=conf.videos_dir + f"/ppo_{args.env}_{timestamp}")
    # evaluation_callback = EvaluationCallBack(env_name, eval_freq=500, n_evals=eval_runs, log_dir=log_dir)
    discriminator_callback = DiscriminatorCallback(d, buffer, discriminator_hyperparams, sw=sw, n_skills=conf.n_z, min_buffer_size=conf.min_train_size, save_dir=conf.log_dir + f"/ppo_{args.env}_{timestamp}", on_policy=True)

    eval_env =  RewardWrapper(SkillWrapper(gym.make(args.env), conf.n_z), d, conf.n_z)
    eval_env = Monitor(eval_env, conf.log_dir + f"/ppo_{args.env}_{timestamp}" + f"/eval_results")
    # eval_env =  Monitor(RewardWrapper(SkillWrapper(gym.make(args.env), conf.n_z), d, conf.n_z), "./`")
    # eval_env = make_vec_env(lambda : RewardWrapper(SkillWrapper(gym.make(args.env), conf.n_z), d, conf.n_z))
    eval_callback = EvalCallback(eval_env, best_model_save_path=conf.log_dir + f"/ppo_{args.env}_{timestamp}",
                                log_path=conf.log_dir + f"/ppo_{args.env}_{timestamp}" + f"/eval_results", eval_freq=1000,
                                deterministic=False, render=False)


    if args.r == "True":
      print("Load saved model to resume training")
      # model_timestamp = 
      d.load_state_dict(torch.load(conf.log_dir + f"/ppo_{args.env}_{model_timestamp}" +  "/disc.pth", map_location=torch.device(conf.device)), )
      model.load(conf.log_dir + f"/ppo_{args.env}_{model_timestamp}" + "/best_model")

    # train the agent
    model.learn(total_timesteps=conf.total_timesteps, callback=[video_loging_callback, discriminator_callback, eval_callback], log_interval=1)



if __name__ == "__main__":
    args = cmd_args()
    run_experiment(args)
    


