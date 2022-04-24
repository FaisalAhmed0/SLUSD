'''
This file record a video for each skill of the pretrained agent, and its performance on the final task
'''
import gym
import argparse

import omegaconf

# MBRL library components 
import mbrl
import mbrl.env.cartpole_continuous as cartpole_env
import mbrl.env.reward_fns as reward_fns
import mbrl.env.termination_fns as termination_fns
import mbrl.models as models
import mbrl.planning as planning
import mbrl.util.common as common_util
import mbrl.util as util


from stable_baselines3.common import callbacks
from stable_baselines3.common.vec_env import VecVideoRecorder, DummyVecEnv
from stable_baselines3 import PPO, SAC
from stable_baselines3.common.callbacks import  EvalCallback
from stable_baselines3.common.utils import get_schedule_fn
from stable_baselines3.common.monitor import Monitor

from src.environment_wrappers.env_wrappers import SkillWrapperFinetune, RewardWrapper, SkillWrapper, TimestepsWrapper
from src.config import conf
from src.utils import seed_everything
from src.individuals import MountainCar, Swimmer, HalfCheetah, Ant, Walker
from src.diayn_mb import Reward_func 
from src.models.models import Discriminator, SeparableCritic, ConcatCritic


import torch
import torch.nn as nn
from torch.utils.tensorboard import SummaryWriter

import numpy as np
import pandas as pd

import time
import os
import random

from gym.wrappers import Monitor
import pyglet
from pyglet import *


# make sure the same seed
# set the seed
seed = 0
seed_everything(seed)



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

experiments = [conf.log_dir_finetune, conf.scalability_exper_dir, conf.generalization_exper_dir, conf.regularization_exper_dir]

# Extract arguments from terminal
def cmd_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--envs', nargs="*", type=str, default=["MountainCarContinuous-v0"])
    parser.add_argument("--algs", nargs="*", type=str, default=["ppo"]) # pass the algorithms that correspond to each timestamp
    parser.add_argument("--skills", nargs="*", type=int, default=[6])
    parser.add_argument("--bestskills", nargs="*", type=int, default=[-1])
    parser.add_argument("--stamps", nargs="*", type=str, required=True) # pass the timestamps as a list
    parser.add_argument("--cls", nargs="*", type=str, required=True) # pass the timestamps as a list
    parser.add_argument("--lbs", nargs="*", type=str, required=True) # pass the timestamps as a list
    parser.add_argument("--exp", nargs="*", type=int, default=[0]) # pass the timestamps as a list
    parser.add_argument("--suf", nargs="*", type=str, default=['']) # pass the timestamps as a list
    args = parser.parse_args()
    return args



# A method to augment observations with the skills reperesentation
def augment_obs(obs, skill, n_skills):
    onehot = np.zeros(n_skills)
    onehot[skill] = 1
    aug_obs = np.array(list(obs) + list(onehot)).astype(np.float32)
    return torch.FloatTensor(aug_obs, device=conf.device).unsqueeze(dim=0)


# A method to return the best performing skill
@torch.no_grad()
def record_skill(model, env_name, alg, n_skills, stamp):
    print(f"skills: {n_skills}")
    display = pyglet.canvas.get_display()
    screen = display.get_screens()
    config = screen[0].get_best_config()
    pyglet.window.Window(width=1024, height=1024, display=display, config=config)
    total = []
    total_rewards = []
    for skill in range(n_skills):
        print(f"Running Skill: {skill}")
        env = TimestepsWrapper(gym.make(env_name))
        env.seed(seed)
        video_folder = f"recorded_agents/env:{env_name},alg: {alg}_skills_videos_stamp: {stamp}"
        env = Monitor(env, video_folder, resume=True,force=False, uid=f"env: {env_name}, skill: {skill}")
        obs = env.reset()
        aug_obs = augment_obs(obs, skill, n_skills)
        total_reward = 0
        done = False
        while not done:
            if alg in ("ppo", "sac"):
                action, _ = model.predict(aug_obs, deterministic=True)
            elif alg == "es":
                action = model.action(aug_obs.numpy().reshape(-1))
            elif alg == "pets":
                print(f"obs: {aug_obs.numpy().reshape(-1)}")
                # input()
                action_seq = model.plan(aug_obs.numpy().reshape(-1))
                action = action_seq[0]
            obs, reward, done, info = env.step(action)
            aug_obs = augment_obs(obs, skill, n_skills)
            total_reward += reward
        total_rewards.append(total_reward)
        env.close()
    total.append(total_rewards)
    total_mean  = np.mean(total, axis=0)
    total_std = np.std(total, axis=0)
    print(f"Total: {total}")
    print(f"Total rewards mean: {total_mean}")
    print(f"Best skill: {np.argmax(total_mean)} and best reward is: {np.max(total_mean)} with std: {total_std[np.argmax(total_mean)]}")
    return np.argmax(total_mean)



def record_skills(env_name, alg, skills, stamp, pm, lb, exper, suffix):
    main_exper_dir = exper + f"cls:{pm}, lb:{lb}/"
    env_dir = main_exper_dir + f"env: {env_name}, alg:{alg}, stamp:{stamp}_{suffix}/"
    seed_dir = env_dir + f"seed:{seed}/"
    model_dir = seed_dir + "/best_model"
    env = DummyVecEnv([lambda: SkillWrapper(gym.make(env_name),skills, max_steps=1000)])
    if alg == "sac":
        model = SAC.load(model_dir, env=env, seed=seed)
    elif alg == "ppo":
        model = PPO.load(model_dir, env=env,  clip_range= get_schedule_fn(ppo_hyperparams['clip_range']), seed=seed)
    elif alg == "es":
        env_invd = ES_env_indv(env_name, skills)
        model_dir = f"{seed_dir}/best_model.ckpt"
        env_invd.net.load_state_dict( torch.load(model_dir) )
        model = env_invd
    elif alg == "pets":
        model_dir = f"{seed_dir}/"
        model = pets_agent(env_name, model_dir, seed_dir, skills, parametrization=pm)
    best_skill_index = record_skill(model, env_name, alg, skills, stamp)
    print(f"Best skill is {best_skill_index}")


def record_learned_agent(best_skill, env_name, alg, skills, stamp, pm, lb, suffix):
    # TODO: test this
    video_folder = f"recorded_agents/env:{env_name}_alg: {alg}_finetune_videos"
    env = SkillWrapperFinetune(gym.make(env_name), skills, max_steps=gym.make(env_name)._max_episode_steps, skill=best_skill)
    env = Monitor(env, video_folder, resume=True,force=False, uid=f"env: {env}, skill: {best_skill}")
    main_exper_dir = conf.regularization_exper_dir + f"cls:{pm}, lb:{lb}/"
    env_dir = main_exper_dir + f"env: {env_name}, alg:{alg}, stamp:{stamp}_{suffix}/"
    seed_dir = env_dir + f"seed:{seed}/"
    # I stopped here
    directory =  seed_dir + f"best_finetuned_model_skillIndex:{best_skill}/" + "/best_model"
    # if alg == "sac":
    model = SAC.load(directory, seed=seed)
    # elif alg == "ppo":
        # model = PPO.load(directory, clip_range= get_schedule_fn(ppo_hyperparams['clip_range']), seed=seed)
    # elif alg == "es":
        # env_invd = ES_env_indv(env_name, skills)
        # model_dir = f"{seed_dir}/best_finetuned_model_skillIndex:{best_skill}/best_model.ckpt"
        # env_invd.net.load_state_dict( torch.load(model_dir) ) 
    # elif alg == "pets":
    #     model_dir = f"{seed_dir}/best_finetuned_model_skillIndex:{best_skill}"
    #     model = pets_agent(env_name, model_dir, skills, True, best_skill)
    obs = env.reset()
    n_skills = skills
    total_reward = 0
    done = False
    while not done:
        if alg in ("ppo", "sac"):
            action, _ = model.predict(obs, deterministic=True)
        elif alg == "es":
            action = model.action(obs.unsqueeze(0))
        elif alg == "pets":
            action_seq = agent.plan(obs)
            action = action_seq[0]
        obs, reward, done, _ = env.step(action)
        obs = obs
        total_reward += reward
    env.close()

def ES_env_indv(env_name, skills):
    if env_name == "MountainCarContinuous-v0":
        MountainCar.MountainCar.n_skills = skills
        env_indv = MountainCar.MountainCar()
    elif env_name == "Swimmer-v2":
        env_indv = Swimmer.Swimmer()
    elif env_name == "HalfCheetah-v2":
        env_indv = HalfCheetah.HalfCheetah()
    elif env_name == "Ant-v2":
        env_indv = Ant.Ant()
    return env_indv

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

# Fix This
def pets_agent(env_name, model_dir, seed_dir, n_skills, FineTune=False, best_skill=None, parametrization=None):
    # print("I am here2")
    if FineTune:
        env = SkillWrapperFinetune(gym.make(env_name), n_skills, best_skill)
    else:
        env = SkillWrapper(gym.make(env_name), n_skills)
        # load the discriminator self.directory + "/disc.pth"
        state_dim = gym.make(env_name).observation_space.shape[0]
        skill_dim = n_skills
        hidden_dims = conf.num_layers_discriminator*[conf.layer_size_discriminator]
        latent_dim = conf.latent_size
        temperature = discriminator_hyperparams['temperature']
        dropout = discriminator_hyperparams['dropout']
        if parametrization == "MLP":
            d = Discriminator(state_dim, hidden_dims, skill_dim, dropout=discriminator_hyperparams['dropout']).to(conf.device)
        elif parametrization == "Linear":
            d = nn.Linear(state_dim, skill_dim).to(conf.device)
        elif parametrization == "Separable":
            # state_dim, skill_dim, hidden_dims, latent_dim, temperature=1, dropout=None)
            d = SeparableCritic(state_dim, skill_dim, hidden_dims, latent_dim, temperature, dropout).to(conf.device)
        elif parametrization == "Concat":
            # state_dim, skill_dim, hidden_dims, temperature=1, dropout=None)
            d = ConcatCritic(state_dim, skill_dim, hidden_dims, temperature, dropout).to(conf.device)
        else:
            raise ValueError(f"{discriminator_hyperparams['parametrization']} is invalid parametrization")
        if os.path.exists(seed_dir + "/disc.pth"):
            d.load_state_dict(torch.load(seed_dir + "/disc.pth"))
        reward_fn = Reward_func(d, n_skills, parametrization)
        
    env.env.seed(seed)
    print(f"skills: {n_skills}")
    print(f"env shape: {env.observation_space.shape}")
    rng = np.random.default_rng(seed=seed)
    generator = torch.Generator(device=conf.device)
    generator.manual_seed(seed)
    obs_shape = env.observation_space.shape
    act_shape = env.action_space.shape
    term_fn = termination_fns.no_termination
    # load the pretrained policy
    ## Configrations
    # TODO: replace this with the hyperparams arguments
    trial_length = 1000
    num_trials = 1
    ensemble_size = 4
    # Everything with "???" indicates an option with a missing value.
    # Our utility functions will fill in these details using the 
    # environment information
    # TODO: replace this with the hyperparams arguments
    cfg_dict = {
        # dynamics model configuration
        "dynamics_model": {
            "_target_": "mbrl.models.GaussianMLP",
            "device": conf.device,
            "num_layers": 2,
            "ensemble_size": 4,
            "hid_size": 300,
            "in_size": "???",
            "out_size": "???",
            "deterministic": False,
            "propagation_method": "fixed_model",
            # can also configure activation function for GaussianMLP
            "activation_fn_cfg": {
                "_target_": "torch.nn.LeakyReLU",
                "negative_slope": 0.01
            }
        },
        # options for training the dynamics model
        "algorithm": {
            "learned_rewards": True,
            "target_is_delta": True,
            "normalize": True,
        },
        # these are experiment specific options
        "overrides": {
            "trial_length": 100,
            "num_steps": 1000,
            "model_batch_size": 1000,
            "validation_ratio": 0.05
        }
    }
    # TODO: replace this with the hyperparams arguments
    cfg = omegaconf.OmegaConf.create(cfg_dict)
    # Create a 1-D dynamics model for this environment
    dynamics_model = common_util.create_one_dim_tr_model(cfg, obs_shape, act_shape)

    # Create a gym-like environment to encapsulate the model
    if FineTune:
        model_env = models.ModelEnv(env, dynamics_model, term_fn, None, generator=generator)
    else:
        model_env = models.ModelEnv(env, dynamics_model, term_fn, reward_fn, generator=generator)
    model_env.dynamics_model.load(f"{model_dir}")
    agent_cfg = omegaconf.OmegaConf.create({
        # this class evaluates many trajectories and picks the best one
        "_target_": "mbrl.planning.TrajectoryOptimizerAgent",
        "planning_horizon": 5,
        "replan_freq": 1,
        "verbose": False,
        "action_lb": "???",
        "action_ub": "???",
        # this is the optimizer to generate and choose a trajectory
        "optimizer_cfg": {
            "_target_": "mbrl.planning.CEMOptimizer",
            "num_iterations": 5,
            "elite_ratio": 0.1,
            "population_size": 500,
            "alpha": 0.1,
            "device": conf.device,
            "lower_bound": "???",
            "upper_bound": "???",
            "return_mean_elites": True,
            "clipped_normal": False
        }
            })
    # CEM agent
    agent = planning.create_trajectory_optim_agent_for_model(
            model_env,
            agent_cfg,
            num_particles=10) 
    return agent
        
    
if __name__ == "__main__":
    args = cmd_args()
    n = len(args.envs)
    for i in range(n):
        exper = experiments[args.exp[i]]
        env_name = args.envs[i]
        alg = args.algs[i]
        skills = args.skills[i]
        stamp = args.stamps[i]
        pm = args.cls[i]
        lb = args.lbs[i]
        best_skill = args.bestskills[i]
        suffix = args.suf[i]
        record_skills(env_name, alg, skills, stamp, pm, lb, exper, suffix)
        if best_skill != -1:    
            record_learned_agent(best_skill, env_name, alg, skills, stamp, pm, lb, suffix)

