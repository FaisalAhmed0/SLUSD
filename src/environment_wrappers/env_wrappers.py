import gym

import torch
import numpy as np

from src.config import conf

class SkillWrapper(gym.Wrapper):
  """
  gym wrapper that augment the state with random chosen skill index
  """
  def __init__(self, env, n_skills, max_steps=1000, ev=False, sample_freq=None):
    # Call the parent constructor, so we can access self.env later
    super(SkillWrapper, self).__init__(env)

    # skills distribution
    self.n_skills = n_skills
    self.skills_dist = torch.randint
    low, high = env.observation_space.low, env.observation_space.high
    low, high= np.concatenate((low, [0]*n_skills)), np.concatenate((high, [1]*n_skills))
    shape = env.observation_space.shape[0]
    self.observation_space = gym.spaces.Box(low=low, high=high, dtype=np.float32, shape=[shape+n_skills,])
    self.env._max_episode_steps = max_steps
    self.max_steps = max_steps
    self.ev = ev
    self.t = 0
    # if ev:
    #     self.seeds = [0, 10, 1234, 5, 42]
    #     self.i = 0

  
  def reset(self):
    """
    Reset the environment 
    """
    # if self.ev:
    #     self.i = (self.i+1) % 5
    #     self.env.seed(self.seeds[self.i])
    # pick a random skill uniformly
    self.skill = self.skills_dist(low=0, high=self.n_skills, size=(1,)).item()
    
    obs = self.env.reset()
    onehot = self.one_hot(self.skill)
    obs = np.array(list(obs) + list(onehot)).astype(np.float32)
    return obs

  def step(self, action):
    """
    :param action: ([float] or int) Action taken by the agent
    :return: (np.ndarray, float, bool, dict) observation, reward, is the episode over?, additional informations
    """
    self.t += 1
    # if self.t % 100 == 0:
    #     # print(f"New sampled skill at t={self.t}")
    #     self.skill = self.skills_dist(low=0, high=self.n_skills, size=(1,)).item()
    obs, reward, done, info = self.env.step(action)
    onehot = self.one_hot(self.skill)
    obs = np.array(list(obs.copy()) + list(onehot)).astype(np.float32)
    return obs, reward, done, info

  def one_hot(self, index):
    onehot = np.zeros(self.n_skills)
    onehot[index] = 1
    return onehot


    


class SkillWrapperVideo(gym.Wrapper):
  """
  gym wrapper that augment the state with random chosen skill index
  """
  def __init__(self, env, n_skills, max_steps=1000):
    super(SkillWrapperVideo, self).__init__(env)

    # skills distribution
    self.n_skills = n_skills
    self.skill = torch.randint(low=0, high=self.n_skills, size=(1,)).item()
    low, high = env.observation_space.low, env.observation_space.high
    low, high= np.concatenate((low, [0]*n_skills)), np.concatenate((high, [1]*n_skills))
    shape = env.observation_space.shape[0]
    self.observation_space = gym.spaces.Box(low=low, high=high, dtype=np.float32, shape=[shape+n_skills,])
    self.env._max_episode_steps = max_steps
    

  
  def reset(self):
    """
    Reset the environment 
    """
    obs = self.env.reset()
    onehot = self.one_hot(self.skill)
    obs = np.array(list(obs) + list(onehot)).astype(np.float32)
    self.t = 0
    return obs

  def step(self, action):
    """
    :param action: ([float] or int) Action taken by the agent
    :return: (np.ndarray, float, bool, dict) observation, reward, is the episode over?, additional informations
    """
    obs, reward, done, info = self.env.step(action)
    onehot = self.one_hot(self.skill)
    obs = np.array(list(obs.copy()) + list(onehot)).astype(np.float32)
    self.t += 1
    if self.t > self.env._max_episode_steps:
      done = True
    return obs, reward, done, info

  def one_hot(self, index):
    onehot = np.zeros(self.n_skills)
    onehot[index] = 1
    return onehot


    
class RewardWrapper(gym.Wrapper):
  """
  gym wrapper that augment the state with random chosen skill index
  """
  def __init__(self, env, discriminator, n_skills, parametrization="MLP", batch_size=128):
    # Call the parent constructor, so we can access self.env later
    super(RewardWrapper, self).__init__(env)

    # save the discriminator

    self.discriminator = discriminator
    self.n_skills = n_skills
    self.reward = 0
    self.t = 0
    self.parametrization = parametrization
    self.batch_size = batch_size

  def reset(self):
    self.t = 0
    return self.env.reset()
  

  def step(self, action):
    """
    :param action: ([float] or int) Action taken by the agent
    :return: (np.ndarray, float, bool, dict) observation, reward, is the episode over?, additional informations
    """
    obs, reward, done, info = self.env.step(action)

    env_obs, skill = self.split_obs(obs)
    obs_t = torch.FloatTensor(env_obs).to(conf.device)
    self.t += 1
    if self.parametrization in ["MLP", "Linear"]:
        self.discriminator.eval()
        # print(f"hereeee in timestep: {self.t}")
        reward = (torch.log_softmax(self.discriminator(obs_t).detach(), dim=-1)[int(skill)] - np.log(1/self.n_skills)).item()
    elif self.parametrization in ["Separable", "Concat"]:
        with torch.no_grad():
            self.discriminator.eval()
            skills_onehot = self.one_hot(skill)
            reward = self.discriminator(obs_t.unsqueeze(0), skills_onehot.unsqueeze(0)).mean().item()
        # print(f"reward in the skill wrapper: {reward}")
    if self.t >= self.env.max_steps:
        done = True
    else:
        done = False
    return obs, reward, done, info

  def split_obs(self, obs):
    env_obs = obs[: -self.n_skills]
    skill, = np.where( obs[-self.n_skills: ] == 1 )
    return env_obs, skill[0]

  def one_hot(self, skill):
    onehot = torch.zeros(self.n_skills)
    onehot[skill] = 1
    return onehot


class SkillWrapperFinetune(gym.Wrapper):
  """
  gym wrapper that augment the state with random chosen skill index
  """
  def __init__(self, env, n_skills, skill, r_seed=None, max_steps=1000):
    super(SkillWrapperFinetune, self).__init__(env)
    # skills distribution
    # print("Hello I am here")
    self.n_skills = n_skills
    self.skill = skill
    low, high = env.observation_space.low, env.observation_space.high
    low, high= np.concatenate((low, [0]*n_skills)), np.concatenate((high, [1]*n_skills))
    shape = env.observation_space.shape[0]
    self.observation_space = gym.spaces.Box(low=low, high=high, dtype=np.float32, shape=[shape+n_skills,])
    self.env._max_episode_steps = max_steps
    # seeds for the 5 evaluation runs
    self.manual_seed = r_seed
  
  def reset(self):
    """
    Reset the environment 
    """
    if self.manual_seed is not None:
        self.env.seed(self.manual_seed)
    obs = self.env.reset()
    # print("I am here")
    onehot = self.one_hot(self.skill)
    obs = np.array(list(obs) + list(onehot)).astype(np.float32)
    self.t = 0
    return obs

  def step(self, action):
    """
    :param action: ([float] or int) Action taken by the agent
    :return: (np.ndarray, float, bool, dict) observation, reward, is the episode over?, additional informations
    """
    obs, reward, done, info = self.env.step(action)
    onehot = self.one_hot(self.skill)
    obs = np.array(list(obs.copy()) + list(onehot)).astype(np.float32)
    self.t += 1
    if self.t > self.env._max_episode_steps:
      done = True
    return obs, reward, done, info

  def one_hot(self, index):
    onehot = np.zeros(self.n_skills)
    onehot[index] = 1
    return onehot

class TimestepsWrapper(gym.Wrapper):
  """
  gym wrapper that augment the state with random chosen skill index
  """
  def __init__(self, env, max_steps=1000):
    super(TimestepsWrapper, self).__init__(env)

    # skills distribution
    self.env = env
    self.max_steps = 1000
    self.t = 0
  
  def reset(self):
    """
    Reset the environment 
    """
    self.t = 0
    return self.env.reset()

  def step(self, action):
    """
    :param action: ([float] or int) Action taken by the agent
    :return: (np.ndarray, float, bool, dict) observation, reward, is the episode over?, additional informations
    """
    obs, reward, done, info = self.env.step(action)
    self.t += 1
    if self.t >= self.env._max_episode_steps:
      done = True
    else:
      done = False
    # print(f"Timestep:{self.t} and done: {done}")
    return obs, reward, done, info


class EvalWrapper(gym.Wrapper):
  """
  gym wrapper that augment the state with random chosen skill index
  """
  def __init__(self, env, max_steps=1000):
    super(EvalWrapper, self).__init__(env)

    # skills distribution
    self.env = env
    self.max_steps = 1000
    self.t = 0
  
  def reset(self):
    """
    Reset the environment 
    """
    self.t = 0
    return self.env.reset()

  def step(self, action):
    """
    :param action: ([float] or int) Action taken by the agent
    :return: (np.ndarray, float, bool, dict) observation, reward, is the episode over?, additional informations
    """
    obs, reward, done, info = self.env.step(action)
    self.t += 1
    if self.t >= self.env._max_episode_steps:
      done = True
    else:
      done = False
    # print(f"Timestep:{self.t} and done: {done}")
    return obs, reward, done, info
