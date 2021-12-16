import gym

import torch
import numpy as np

from src.config import conf

class SkillWrapper(gym.Wrapper):
  """
  gym wrapper that augment the state with random chosen skill index
  """
  def __init__(self, env, n_skills, max_steps=1000, ev=False):
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
    self.ev = ev
    if ev:
        self.seeds = [0, 10, 1234, 5, 42]
        self.i = 0

  
  def reset(self):
    """
    Reset the environment 
    """
    if self.ev:
        self.i = (self.i+1) % 5
        self.env.seed(self.seeds[self.i])
    # pick a random skill uniformly
    self.skill = self.skills_dist(low=0, high=self.n_skills, size=(1,)).item()
    
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
  def __init__(self, env, discriminator, n_skills, parametrization="MLP"):
    # Call the parent constructor, so we can access self.env later
    super(RewardWrapper, self).__init__(env)

    # save the discriminator

    self.discriminator = discriminator
    self.n_skills = n_skills
    self.reward = 0
    self.t = 0
    self.parametrization = parametrization
  

  def step(self, action):
    """
    :param action: ([float] or int) Action taken by the agent
    :return: (np.ndarray, float, bool, dict) observation, reward, is the episode over?, additional informations
    """
    obs, reward, done, info = self.env.step(action)

    env_obs, skill = self.split_obs(obs)
    obs_t = torch.FloatTensor(env_obs).to(conf.device)
    self.discriminator.eval()
    if self.parametrization == "MLP":
        reward = (torch.log_softmax(self.discriminator(obs_t).detach(), dim=-1)[int(skill)] - np.log(1/self.n_skills)).item()
    elif self.parametrization == "CPC":
        skills_onehot = self.one_hot(skill)
        # print("reward is")
        # print()
        # print(f"input shape: {obs_t.shape} and {skills_onehot.shape}")
        # input()
        logits = self.discriminator(obs_t.unsqueeze(0), skills_onehot.unsqueeze(0))
        probs = torch.softmax(logits, dim=-1)
        reward = ( torch.log(probs).detach() - np.log(1/self.n_skills)).item()
        print(f"reward in the skill wrapper: {reward}")
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
  def __init__(self, env, n_skills, skill, max_steps=1000):
    super(SkillWrapperFinetune, self).__init__(env)

    # skills distribution
    self.n_skills = n_skills
    self.skill = skill
    low, high = env.observation_space.low, env.observation_space.high
    low, high= np.concatenate((low, [0]*n_skills)), np.concatenate((high, [1]*n_skills))
    shape = env.observation_space.shape[0]
    self.observation_space = gym.spaces.Box(low=low, high=high, dtype=np.float32, shape=[shape+n_skills,])
    self.env._max_episode_steps = max_steps
    # seeds for the 5 evaluation runs
    self.seeds = [0, 10, 1234, 5, 42]
    self.i = 0

  
  def reset(self):
    """
    Reset the environment 
    """
    self.i = (self.i+1) % 5
    self.env.seed(self.seeds[self.i])
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