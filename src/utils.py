from stable_baselines3.common.vec_env import VecVideoRecorder, DummyVecEnv

import gym
import numpy as np

from src.environment_wrappers.env_wrappers import RewardWrapper, SkillWrapperVideo, SkillWrapper
from src.config import conf

import torch

import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from scipy.stats import multivariate_normal
import os

# plot a 2D gaussian 
def plot_multinorm(data, args):
    assert data.shape[1] == 2
    # calculate the mean and the covariance
    n_points = 50j
    mean = np.mean(data, axis=0)
    cov = np.cov(data, rowvar=0)
    # print(f"The mean is: {mean}")
    # print(f"Covariance is: {cov}")
    # create the mesh
    x, y = np.mgrid[-1.2:0.6:n_points, -0.07:0.07:n_points]
    # print(x.shape)
    xy = np.column_stack([x.flat, y.flat])
    # print(xy.shape)
    # create the gaussian object
    multinorm = multivariate_normal.pdf(xy, mean, cov)
    # print(multinorm)
    multinorm = multinorm.reshape(x.shape)
    # create a figure
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.set_xlabel("Position")
    ax.set_ylabel("Velocity")
    ax.set_xlim(-1.5, 0.7)
    ax.set_ylim(-0.09, 0.09)
    ax.plot_surface(x, y, multinorm)
    entropy = multivariate_normal.entropy(mean, cov)
    print(f"Entropy is: {entropy}")
    files_dir = "Vis/MountainCar"
    os.makedirs(files_dir, exist_ok=True)
    fig.savefig(f'{files_dir}/State Distrbution for DIAYN with {args.alg} and {args.skills} skills', dpi=150)

    
    N = 60  
    X = np.linspace(-1.2, 0.6, N)
    Y = np.linspace(-0.07, 0.07, N)
    fig = plt.figure()
    ax = fig.add_subplot(111, projection=None)
    ax.set_xlabel("Position")
    ax.set_ylabel("Velocity")
    ax.set_xlim(-1.5, 0.7)
    ax.set_ylim(-0.09, 0.09)
    X, Y = np.meshgrid(X, Y)
    # Pack X and Y into a single 3-dimensional array
    pos = np.empty(X.shape + (2,))
    pos[:, :, 0] = X
    pos[:, :, 1] = Y
    Z = multivariate_normal.pdf(pos, mean, cov)
    ax.contourf(X, Y, Z, zdir='z')
    fig.savefig(f'{files_dir}/State Distrbution Contours for DIAYN with {args.alg} and {args.skills} skills', dpi=150)
    
    
    
    
    

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
  eval_env = DummyVecEnv([lambda: (SkillWrapperVideo(gym.make(env_id), n_z))])
  eval_env = VecVideoRecorder(eval_env, video_folder=video_folder,
                              record_video_trigger=lambda step: step == 0, video_length=video_length,
                              name_prefix = f"env: {env_id}, time step: {n_calls}, skill: {eval_env.envs[0].skill}")
                              
  obs = eval_env.reset()
  print("Here")
  # Start the video at step=0 and record 500 steps
  for _ in range(video_length):
    action, _ = model.predict(obs)
    obs, _, _, _ = eval_env.step(action)
  # Close the video recorder
  eval_env.close()


  # A function to record a video of the agent interacting with the environment
def record_video_finetune(env_id, skill, model, n_z, video_length=1000, video_folder='videos/', alg="ppo"):
  """
  :param env_id: (str)
  :param model: (RL model)
  :param video_length: (int)
  :param prefix: (str)
  :param video_folder: (str)
  """
  # print("Here")
  eval_env = DummyVecEnv([lambda: (SkillWrapperVideo(gym.make(env_id), n_z))])
  eval_env = VecVideoRecorder(eval_env, video_folder=video_folder,
                              record_video_trigger=lambda step: step == 0, video_length=video_length,
                              name_prefix = f"env: {env_id}, alg: {alg}, skill: {skill}")
                              
  eval_env.skill = skill
  obs = eval_env.reset()
  # Start the video at step=0 and record 500 steps
  for _ in range(video_length):
    action, _ = model.predict(obs, deterministic=True)
    obs, _, _, _ = eval_env.step(action)
    # print(obs)
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
    aug_obs = np.array(list(obs) + list(onehot)).astype(np.float32)
    return torch.FloatTensor(aug_obs).unsqueeze(dim=0)


# A method to return the best performing skill
def best_skill(model, env_name, n_skills):
    env = gym.make(env_name)
    total_rewards = []
    for skill in range(n_skills):
        obs = env.reset()
        aug_obs = augment_obs(obs, skill, n_skills)
        total_reward = 0
        done = False
        while not done:
            action, _ = model.predict(aug_obs, deterministic=True)
            obs, reward, done, _ = env.step(action)
            aug_obs = augment_obs(obs, skill, n_skills)
            total_reward += reward
        total_rewards.append(total_reward)
    return np.argmax(total_rewards)