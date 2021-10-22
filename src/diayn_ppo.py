import gym
import random

import torch
import torch.nn as nn

import argparse

from torch.utils.tensorboard import SummaryWriter

import numpy as np
import matplotlib.pyplot as plt

from collections import deque
import random

from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import  EvalCallback
from stable_baselines3.common.vec_env import DummyVecEnv

from models import Discriminator
from replayBuffers import DataBuffer
from config import conf
from callbacks import DiscriminatorCallback, VideoRecorderCallback
from env_wrappers import RewardWrapper, SkillWrapperVideo, SkillWrapper

from typing import Any, Dict


# PPO Hyperparameters 
ppo_hyperparams = dict(
    learning_rate = 3e-4,
    gamma = 0.99,
    batch_size = 128,
    n_epochs = 1,
    gae_lambda = 0.95,
    n_steps = 2000,
    clip_range = 0.2,
    ent_coef=0.1,
)

# Discriminator Hyperparameters
discriminator_hyperparams = dict(
    learning_rate = 1e-5,
    batch_size = 128,
    n_epochs = 1,
)

# set the seed
seed = 1
random.seed(seed)
np.random.seed(seed)
torch.random.manual_seed(seed)

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
    sw = SummaryWriter(log_dir=conf.log_dir + f"/ppo_{args.env}", comment=f"PPO Discriminator, env_name:{args.env}")

    # create the environment # monitor_dir=testing_log_dir
    env = DummyVecEnv([lambda: RewardWrapper(SkillWrapper(gym.make(args.env), conf.n_z, max_steps=conf.max_steps), d, conf.n_z)])
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
                tensorboard_log = conf.log_dir + f"/ppo_{args.env}",

                )
    

    # Create Callbacks
    # save_best_model_callback = SaveOnBestTrainingRewardCallback(check_freq=100, log_dir=log_dir, verbose=0)
    video_loging_callback = VideoRecorderCallback(args.env, record_freq=5000, deterministic=d, n_z=conf.n_z, videos_dir=conf.videos_dir + f"/ppo_{args.env}")
    # evaluation_callback = EvaluationCallBack(env_name, eval_freq=500, n_evals=eval_runs, log_dir=log_dir)
    discriminator_callback = DiscriminatorCallback(d, buffer, discriminator_hyperparams, sw=sw, n_skills=conf.n_z, min_buffer_size=conf.min_train_size, save_dir=conf.log_dir + f"/ppo_{args.env}", on_policy=True)

    eval_env = RewardWrapper(SkillWrapper(gym.make(args.env), conf.n_z), d, conf.n_z)
    eval_callback = EvalCallback(eval_env, best_model_save_path=conf.log_dir + f"/ppo_{args.env}",
                                log_path=conf.log_dir + f"/ppo_{args.env}", eval_freq=1000,
                                deterministic=True, render=False)


    if args.r == "True":
      print("Load saved model to resume training")
      d.load_state_dict(torch.load(conf.log_dir + f"/ppo_{args.env}" +  "/disc.pth", map_location=torch.device(conf.device)), )
      model.load(conf.log_dir + f"/ppo_{args.env}" + "/best_model")

    # train the agent
    model.learn(total_timesteps=conf.total_timesteps, callback=[video_loging_callback, discriminator_callback, eval_callback], log_interval=1)



if __name__ == "__main__":
    args = cmd_args()
    run_experiment(args)
    


