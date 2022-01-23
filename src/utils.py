from stable_baselines3.common.vec_env import VecVideoRecorder, DummyVecEnv
from stable_baselines3.common.utils import get_schedule_fn
from stable_baselines3 import PPO, SAC

import gym
import numpy as np
import pandas as pd

from src.environment_wrappers.env_wrappers import RewardWrapper, SkillWrapperVideo, SkillWrapperFinetune, SkillWrapper
from src.config import conf

import torch
import torch.nn as nn
import torch.nn.functional as F

import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from scipy.stats import multivariate_normal
import os

from entropy_estimators import continuous

from gym.wrappers import Monitor
import pyglet
from pyglet import *

import random

def seed_everything(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.random.manual_seed(seed)

def kde_entropy(data):
    # from https://github.com/paulbrodersen/entropy_estimators
    # compute the entropy using the k-nearest neighbour approach
    # developed by Kozachenko and Leonenko (1987):
    kozachenko = continuous.get_h(data, k=30, min_dist=0.00001)
    print(f"kozachenko entropy is {kozachenko}")
    return kozachenko

# run a random agent
def random_agent(env):
  '''
  A function to run a random agent in a gym environment
  '''
  total_reward = 0
  total_steps = 0
  done = False
  state = env.reset()
  while not done:
    # sample random action
    action = env.action_space.sample()
    state, reward, done, _ = env.step(action)
    total_reward += reward
    total_steps += 1
  print(f"Total time steps: {total_steps}, total reward: {total_reward}")
  return total_reward, total_steps



# A function to record a video of the agent interacting with the environment
def record_video(env_id, model, n_z, n_calls,video_length=1000, video_folder='videos/'):
    """
    :param env_id: (str)
    :param model: (RL model)
    :param video_length: (int)
    :param prefix: (str)
    :param video_folder: (str)
    """
    display = pyglet.canvas.get_display()
    screen = display.get_screens()
    config = screen[0].get_best_config()
    pyglet.window.Window(width=1024, height=1024, display=display, config=config)
    env = SkillWrapperVideo(gym.make(env_id), n_z)
    env = Monitor(env, video_folder, resume=True,force=False, uid=f"env: {env_id}, time step: {n_calls}, skill: {env.skill}")
    obs = env.reset()
    done = False
    while not done:
        action, _ = model.predict(obs)
        obs, reward, done, info = env.step(action)
    env.close()


  # A function to record a video of the agent interacting with the environment
def record_video_finetune(env_id, skill, model, n_z, video_length=1000, video_folder='videos/', alg="ppo"):
  """
  :param env_id: (str)
  :param model: (RL model)
  :param video_length: (int)
  :param prefix: (str)
  :param video_folder: (str)
  """
  eval_env = DummyVecEnv([lambda: (SkillWrapperVideo(gym.make(env_id), n_z))])
  eval_env = VecVideoRecorder(eval_env, video_folder=video_folder,
                              record_video_trigger=lambda step: step == 0, video_length=video_length,
                              name_prefix = f"env: {env_id}, alg: {alg}, skill: {skill}")
                              
  eval_env.skill = skill
  obs = eval_env.reset()
  for _ in range(video_length):
    action, _ = model.predict(obs, deterministic=True)
    obs, _, _, _ = eval_env.step(action)
  # Close the video recorder
  eval_env.close()



def plot_evaluation(logfile_dir, env_name=None ,save=False):
  steps = []
  rewards_mean = []
  rewards_std = []
  logs = np.loadtxt(logfile_dir)
  for log in logs:
    steps.append(int(log[0]))
    rewards_mean.append(log[1])
    rewards_std.append(log[2])
  plt.errorbar(steps, rewards_mean, yerr=rewards_std, capsize=2)
  plt.xlabel('Time step')
  plt.ylabel('Average return')
  if save:
    plt.savefig(f"Average return {env_name}")

    
# A method to augment observations with the skills reperesentation
def augment_obs(obs, skill, n_skills):
    onehot = np.zeros(n_skills)
    onehot[skill] = 1
    aug_obs = np.array(list(obs.copy()) + list(onehot)).astype(np.float32)
    return torch.FloatTensor(aug_obs).unsqueeze(dim=0)


# A method to return the best performing skill
def best_skill(model, env_name, n_skills, alg_type="rl"):
    seeds = [0, 10, 1234, 5, 42]
    env = gym.make(env_name)
    total = []
    for seed in seeds:
        rewards = []
        env.seed(seed)
        for skill in range(n_skills):
            obs = env.reset()
            aug_obs = augment_obs(obs, skill, n_skills)
            total_reward = 0
            done = False
            while not done:
                if alg_type == "rl":
                    action, _ = model.predict(aug_obs, deterministic=True)
                elif alg_type == "es":
                    action = model.action(aug_obs.unsqueeze(0))
                elif alg_type == "pets":
                    action_seq = agent.plan(aug_obs.numpy().reshape(-1))
                    action = action_seq[0]
                obs, reward, done, _ = env.step(action)
                aug_obs = augment_obs(obs, skill, n_skills)
                total_reward += reward
            rewards.append(total_reward)
        total.append(rewards)
    print(f"All seeds rewards: {total}")
    print(f"mean across seeds: {np.mean(total, axis=0)}")    
    return np.argmax(np.mean(total, axis=0))


# State coverage
@torch.no_grad()
def evaluate_state_coverage(env_name, n_skills, model, alg):
    print("Evaluating Pretrained Policy State Coverage")
    # print(f"Number of skills: {n_skills}")
    # run the model to collect the data
    data = []
    # wrap inside the skill and reward wrappers
    env = gym.make(env_name)
    for skill in range(n_skills):
        obs = env.reset()
        data.append(list(obs.reshape(-1).copy()))
        aug_obs = augment_obs(obs, skill, n_skills)
        total_reward = 0
        done = False
        while not done:
            if alg in ["ppo", "sac"]:
                action, _ = model.predict(aug_obs, deterministic=True)
            elif alg == "es":
                action = model.action(aug_obs.numpy())
            elif alg == "pets":
                action_seq = model.plan(aug_obs.numpy().reshape(-1))
                action = action_seq[0]
            obs, _, done, _ = env.step(action)
            data.append(list(obs.reshape(-1).copy()))
            aug_obs = augment_obs(obs, skill, n_skills)
            # break
        # break
    # I stoped here
    # print(f"data: {data}")
    # input()
    data = np.array(data)
    np.random.shuffle(data)
    # print("here")
    # print(data.shape)
    # print(data[:3])
    # input()
    return kde_entropy(data)


# extract the final mean and std results of an experment
def extract_results(data):
    data_mean = data.mean(axis=1)
    data_mean = np.convolve(data_mean, np.ones(10)/10, mode='valid') 
    data_std = data.std(axis=1)
    data_std = np.convolve(data_std, np.ones(10)/10, mode='valid') 
    print(f"mean: {data_mean[-1]}")
    print(f"std: {data_std[-1]}")
    return data_mean[-1]

# extract the result of the finetuning
@torch.no_grad()
def evaluate_pretrained_policy_ext(env_name, n_skills, model, alg):
    print("Evaluating Pretrained Policy in Extrensic reward")
    total_rewards = []
    for skill in range(n_skills):
        # print(f"Running Skill: {skill}")
        env = gym.make(env_name)
        obs = env.reset()
        aug_obs = augment_obs(obs, skill, n_skills)
        total_reward = 0
        done = False
        while not done:
        # for i in range(2):
            if alg in ["ppo", "sac"]:
                action, _ = model.predict(aug_obs, deterministic=True)
            elif alg == "es":
                action = model.action(aug_obs.squeeze().numpy())
            elif alg == "pets":
                action_seq = model.plan(aug_obs.numpy().reshape(-1))
                action = action_seq[0]
            obs, reward, done, info = env.step(action)
            aug_obs = augment_obs(obs, skill, n_skills)
            total_reward += reward
        print(f"Total reward: {total_reward}")
        total_rewards.append(total_reward)
        env.close()
    best_reward = np.max(total_rewards)
    return best_reward

@torch.no_grad()
def evaluate_pretrained_policy_intr(env_name, n_skills, model, d, parametrization, alg):
    print("Evaluating Pretrained Policy in Intrinsic reward")
    d.eval()
    env = RewardWrapper(SkillWrapper(gym.make(env_name), n_skills), d, n_skills, parametrization)
    done = False
    total_reward = 0
    obs = env.reset()
    while not done:
    # for i in range(2):
        if alg in ["ppo", "sac"]:
            action, _ = model.predict(obs, deterministic=True)
        elif alg == "es":
            action = model.action(obs)
        elif alg == "pets":
            action_seq = model.plan(obs)
            action = action_seq[0]
        obs, reward, done, info = env.step(action)
        total_reward += reward
    print(f"total_reward: {total_reward}")
    return total_reward
                        

@torch.no_grad()
def evaluate_adapted_policy(env_name, n_skills, bestskill, model, alg):
    print("Evaluating Adapted Policy")
    env = gym.make(env_name)
    done = False
    obs = env.reset()
    aug_obs = augment_obs(obs, bestskill, n_skills)
    total_reward = 0
    while not done:
        # if alg in ["ppo", "sac"]:
        action, _ = model.predict(aug_obs, deterministic=True)
        # elif alg == "es":
        #     action = model.action(aug_obs.unsqueeze(0))
        # elif alg == "pets":
        #     action_seq = agent.plan(aug_obs.numpy().reshape(-1))
        #     action = action_seq[0]
        obs, reward, done, info = env.step(action)
        aug_obs = augment_obs(obs, bestskill, n_skills)
        total_reward += reward
    return total_reward



# report the experiment results and save it in data frame
def report_resuts(env_name, alg, n_skills, model, data_r, data_i, timestamp, exper_directory):
    r = {}
    reward_beforeFinetune_mean, reward_beforeFinetune_std = extract_resultes_intrinsic(env_name, n_skills, model)
    r['reward_beforeFinetune_mean'] = reward_beforeFinetune_mean
    r['reward_beforeFinetune_std'] = reward_beforeFinetune_std
    reward_mean, reward_std = extract_results(data_r)
    r['reward_mean'] = reward_mean
    r['reward_std'] = reward_std
    entropy_mean, entropy_std = run_pretrained_policy(env_name, n_skills, alg, timestamp)
    r['entropy_mean'] = entropy_mean
    r['entropy_std'] = entropy_std
    intrinsic_reward_mean, intrinsic_reward_std = extract_results(data_i)
    r['intrinsic_reward_mean'] = intrinsic_reward_mean
    r['intrinsic_reward_std'] = intrinsic_reward_std
    results_df = pd.DataFrame(r, index=[0])
    results_df.to_csv(f"{exper_directory}/results.csv")
    return results_df

# evaluate the CPC based policy
torch.no_grad()
def evaluate_mi(env_name, n_skills, model, tb_sw, discriminator, timesteps, temparture, mi_estimator):
    eval_runs = 5
    batch_size = max_steps = 1000
    rewards = np.zeros(5)
    seeds = [0, 10, 1234, 5, 42]
    n_sampled_skill = 5
    i = 0
    discriminator.eval()
    for run in range(eval_runs):
        data = torch.zeros(n_sampled_skill * max_steps, gym.make(env_name).observation_space.shape[0] + n_skills)
        for i in range(n_sampled_skill):
            skill = np.random.randint(low=0, high=n_sampled_skill)
            env = SkillWrapperFinetune(gym.make(env_name), n_skills, skill, r_seed=seeds[run])
            obs = env.reset()
            data[0] = torch.tensor(obs.copy(), device=conf.device)
            for i in range(1, max_steps):
                action, _ = model.predict(obs, deterministic=True)
                obs, reward, done, _ = env.step(action)
                data[i] = torch.tensor(obs.copy(), device=conf.device)
        # print(f"length of the evaluation buffer: {len(data)}")
        idx = torch.randperm(data.shape[0])
        data = data[idx]
        env_obs = torch.clone(data[:, : -n_skills])
        skills = torch.clone(data[:, -n_skills:])
        # print(f"env_obs: {env_obs}")
        # print(f"skills: {skills}")
        # forward pass
        scores = discriminator(env_obs, skills)
        # calculate the reward
        estimator = mi_estimator.estimator_func
        eps_reward = estimator(scores, mi_estimator.estimator_type, mi_estimator.log_baseline, mi_estimator.alpha_logit)
        # print(f"log probs in diag: {log_probs[:3]}")
        # print(f"log_probs diag in eval: {log_probs.diag()[:3]}")
        # input()
        # calculate the reward
        final_return =  eps_reward.sum().item()/1
        # print(f"final return: {final_return}")
        rewards[run] = final_return
    # print(f"all seeds rewards: {rewards}")
    # print(f"mean reward: {rewards.mean()} and std: {rewards.std()}")
    tb_sw.add_scalar("eval/mean_reward", rewards.mean(), timesteps)
    # input()
    return rewards
            
            
# Interpolated lower bound I_{\alpha}
def log_interpolate(log_a, log_b, alpha_logit):
    '''
    Numerically stable interpolation in log space: log( alpha * a + (1-alpha) * b  ),
    based on this notebook by google research 
https://github.com/google-research/google-research/blob/master/vbmi/vbmi_demo.ipynb
    '''
    log_alpha =  - F.softplus(torch.tensor(-alpha_logit, device=conf.device))
    log_1_minus_alpha =  - F.softplus(torch.tensor(alpha_logit, device=conf.device))
    dim = tuple(i for i in range(log_a.dim()))
    y = torch.logsumexp( torch.stack((log_alpha + log_a, log_1_minus_alpha + log_b)) , dim=0)
    return y

def compute_log_loomean(scores):
    '''
    comute the mean of the second term of the interpolated lower bound (leave-one-out) in a numerically stable way, based on this notebook by google research 
https://github.com/google-research/google-research/blob/master/vbmi/vbmi_demo.ipynb
    '''
    max_score = torch.max(scores, dim=-1, keepdim=True)[0]
    # logsumexp minus the max
    lse_minus_max = torch.logsumexp(scores - max_score, dim=1, keepdim=True)
    d = lse_minus_max + (max_score - scores)
    d_ok = torch.not_equal(d, 0.)
    safe_d = torch.where(d_ok, d, torch.ones_like(d))
    loo_lse = scores + softplus_inverse(safe_d)
    # normailize by the batch size
    loo_lme = loo_lse - torch.log(torch.tensor(scores.shape[1], device=conf.device) - 1.)
    return loo_lme

def softplus_inverse(x):
    if not torch.is_tensor(x):
        x = torch.tensor(x, device=conf.device)
    '''
    A function that implement tge softplus invserse, this is based on tensorflow implementatiion on 
    https://github.com/tensorflow/probability/blob/v0.15.0/tensorflow_probability/python/math/generic.py#L494-L545
    '''
    # check the limit for floating point arethmatics in numpy
    threshold = np.log(np.finfo(np.float32).eps) + 2.
    is_too_small =  x < np.exp(threshold)
    is_too_large = x > -threshold
    too_small_value = torch.log(x)
    too_large_value = x
    x = torch.where(is_too_small | is_too_large, torch.ones_like(x), x)
    y = x + torch.log(-(torch.exp(-x) - 1))
    return torch.where(is_too_small, too_small_value, torch.where(is_too_large, too_large_value, y))


#### Useful function for the adaptation expeirment ####
def evaluate(env, n_skills, pretrained_policy, adapted_policy, discriminator, parametrization, bestskill, alg):
    # intrinsic reward
    intrinsic_reward_mean = np.mean([evaluate_pretrained_policy_intr(env, n_skills, pretrained_policy, discriminator, parametrization, alg) for _ in range(5)])
    # best skill reward before adaptation
    reward_beforeFinetune_mean = np.mean([evaluate_pretrained_policy_ext(env, n_skills, pretrained_policy, alg) for _ in range(5)])
    # reward after adaptation
    reward_mean = np.mean([evaluate_adapted_policy(env, n_skills, bestskill, adapted_policy, alg) for _ in range(5)])
    # entropy
    entropy_mean = np.mean([evaluate_state_coverage(env, n_skills, pretrained_policy, alg) for _ in range(5)])
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
    return df

def plot_learning_curves(env_name, algs, stamps, skills_list, pms, lbs, asym):
    colors = ['b', 'r', 'g', 'b']
    xlabel = "Environment Timesteps"
    ylabel = "Reward After Adaptation"
    legends =  algs
    # file_dir = conf.log_dir_finetune + f"{args.alg}_{args.env}_{args.stamp}/" + "eval_results/" + "evaluations.npz"
    # file_dir = conf.log_dir_finetune + f"{args.alg}_{args.env}_{args.stamp}/" + "evaluations.npz"
    print("######## Skills evaluation ########")
    plt.figure(figsize=conf.learning_curve_figsize)
    plt.title(f"{env_name[: env_name.index('-')]}")
    for i in range(len(algs)):
        skills = skills_list[i]
        pm = pms[i]
        lb = lbs[i]
        alg = algs[i]
        print(f"Algorithm: {alg}")
        plot_color = colors[i]
        stamp = stamps[i]
        # legend = alg
        main_exper_dir = conf.log_dir_finetune + f"cls:{pm}, lb:{lb}/"
        env_dir = main_exper_dir + f"env: {env_name}, alg:{alg}, stamp:{stamp}/"
        seeds_list = []
        for seed in conf.seeds:
            seed_everything(seed)
            seed_dir = env_dir + f"seed:{seed}/"
            file_dir_skills = seed_dir + "eval_results/" + "evaluations.npz"
            files = np.load(file_dir_skills)
            steps = files['timesteps']
            results = files['results']
            print(f"results.shape: {results.shape}")
            if alg in ("ppo", "sac"):
                seeds_list.append(results.mean(axis=1).reshape(-1))
            elif alg in ("es", "pets"):
                seeds_list.append(results)
            # data_mean = results.mean(axis=1)
            # data_std = results.std(axis=1)
            # print(f"mean: {data_mean[-1]}")
            # print(f"std: {data_std[-1]}")
        data_mean = np.mean(seeds_list, axis=0) / asym
        print(f"seeds list shape {np.array(seeds_list).shape}")
        # input()
        data_std = np.std(seeds_list, axis=0) / asym
        # data_mean = np.convolve(data_mean, np.ones(10)/10, mode='valid') 
        # data_std = np.convolve(data_std, np.ones(10)/10, mode='valid') 
        # steps = np.convolve(steps, np.ones(10)/10, mode='valid') 
        print(f"data shape: {data_mean.shape}")
        print(f"steps shape: {steps.shape}")
        # input()
        if len(algs) > 1:
            plt.plot(steps, data_mean, label=legends[i], color=plot_color)
            plt.legend()
        else:
            plt.plot(steps, data_mean, color=plot_color)
        plt.fill_between(steps, (data_mean-data_std), (data_mean+data_std), color=plot_color, alpha=0.3)
        plt.xlabel(xlabel)
        plt.ylabel(ylabel)
        # plt.grid(False)
        if len(algs) == 1:
            filename = f"{env_name}_pretraining_{alg}"
            files_dir = f"Vis/{env_name}_cls:{pm}_lb:{lb}"
            os.makedirs(files_dir, exist_ok=True)
            plt.savefig(f'{files_dir}/{filename}', dpi=150)    
    if len(algs) > 1:
        filename = f"{env_name}_pretraining_all_algs"
        files_dir = f"Vis/{env_name}_cls:{pm}_lb:{lb}"
        os.makedirs(files_dir, exist_ok=True)
        plt.savefig(f'{files_dir}/{filename}', dpi=150)    
    
    # loop throgh the finetuning results
    # loop throgh the finetuning results
    plt.figure(figsize=conf.learning_curve_figsize)
    plt.title(f"{env_name[: env_name.index('-')]}")
    print("######## Finetine evaluation ########")
    seeds_list = []
    for i in range(len(algs)):
        skills = skills_list[i]
        pm = pms[i]
        lb = lbs[i]
        alg = algs[i]
        print(f"Algorithm: {alg}")
        plot_color = colors[i]
        stamp = stamps[i]
        legend = alg
        main_exper_dir = conf.log_dir_finetune + f"cls:{pm}, lb:{lb}/"
        env_dir = main_exper_dir + f"env: {env_name}, alg:{alg}, stamp:{stamp}/"
        seeds_list = []
        for seed in conf.seeds:
            seed_everything(seed)
            seed_dir = env_dir + f"seed:{seed}/"
            file_dir_finetune = seed_dir + "finetune_eval_results/" + "evaluations.npz"
            files = np.load(file_dir_finetune)
            steps = files['timesteps']
            results = files['results']
            # if alg in ("ppo", "sac"):
            seeds_list.append(results.mean(axis=1).reshape(-1))
            # elif alg == "es":
            #     seeds_list.append(results)
        data_mean = np.mean(seeds_list, axis=0) / asym
        data_std = np.std(seeds_list, axis=0) / asym
        if len(algs) > 1:
            plt.plot(steps, data_mean, label=legends[i], color=plot_color)
            plt.legend()
        else:
            plt.plot(steps, data_mean, color=plot_color)
        plt.fill_between(steps, (data_mean-data_std), (data_mean+data_std), color=plot_color, alpha=0.3)
        plt.xlabel(xlabel)
        plt.ylabel(ylabel)
        # plt.grid(False)
        if len(algs) == 1:
            filename = f"{env_name}_finetune_{alg}"
            files_dir = f"Vis/{env_name}_cls:{pm}_lb:{lb}"
            os.makedirs(files_dir, exist_ok=True)
            plt.savefig(f'{files_dir}/{filename}', dpi=150)    
    if len(algs) > 1:
        filename = f"{env_name}_finetune_all_algs"
        files_dir = f"Vis/{env_name}_cls:{pm}_lb:{lb}"
        os.makedirs(files_dir, exist_ok=True)
        plt.savefig(f'{files_dir}/{filename}', dpi=150)      

#### Useful function for the adaptation expeirment ####
        
    
