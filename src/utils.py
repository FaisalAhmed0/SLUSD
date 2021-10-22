from stable_baselines3.common.vec_env import VecVideoRecorder, DummyVecEnv

import gym
import numpy as np
import matplotlib.pyplot as plt

from env_wrappers import SkillWrapperFinetune, SkillWrapperVideo
from config import conf


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
    print(obs)
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