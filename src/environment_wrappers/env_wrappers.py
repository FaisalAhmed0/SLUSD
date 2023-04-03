import gym
from gym import Wrapper
from gym import spaces
import d4rl

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
    obs_t = torch.FloatTensor(env_obs, device=conf.device)
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
    onehot = torch.zeros(self.n_skills, device=conf.device)
    onehot[skill] = 1
    return onehot


class SkillWrapperFinetune(gym.Wrapper):
  """
  gym wrapper that augment the state with random chosen skill index
  """
  def __init__(self, env, n_skills, skill, seed=None, max_steps=1000, l=None):
    super(SkillWrapperFinetune, self).__init__(env)
    # skills distribution
    # print("Hello I am here")
    self.n_skills = n_skills
    self.env = env
    self.skill = skill
    low, high = env.observation_space.low, env.observation_space.high
    low, high= np.concatenate((low, [0]*n_skills)), np.concatenate((high, [1]*n_skills))
    shape = env.observation_space.shape[0]
    self.observation_space = gym.spaces.Box(low=low, high=high, dtype=np.float32, shape=[shape+n_skills,])
    self.env._max_episode_steps = max_steps
    self.env.seed(seed)
    self.l = l
    # seeds for the 5 evaluation runs
  
  def reset(self):
    """
    Reset the environment 
    """
    obs = self.env.reset()
    self.t = 0
    # print("I am here")
    onehot = self.one_hot(self.skill)
    obs = np.array(list(obs) + list(onehot)).astype(np.float32)
    # self.total = 0
    return obs

  def step(self, action):
    """
    :param action: ([float] or int) Action taken by the agent
    :return: (np.ndarray, float, bool, dict) observation, reward, is the episode over?, additional informations
    """
    self.t += 1
    obs, reward, done, info = self.env.step(action)
    # self.total += reward
    onehot = self.one_hot(self.skill)
    obs = np.array(list(obs.copy()) + list(onehot)).astype(np.float32)
    # print(self.l)
    # print(f"action: {action}")
    # print(f"obs: {obs}")
    # print(f"reward: {reward}")
    # print(f"total reward: {self.total}")
    # print(f"total steps: {self.t}")
    # input()
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


class RGBArrayAsObservationWrapper(Wrapper):
    """
    Use env.render(rgb_array) as observation
    rather than the observation environment provides
    """
    def __init__(self, env, num_skills, n_stack):
        '''
        env: Gym environment object
        num_skills: number of skill in pretraing
        n_stack: size of the frame stack
        '''
        self.vector_size = num_skills
        self.env = env
        self.num_skills = num_skills
        self.n_stack = n_stack
        
        dummy_obs = env.render("rgb_array")
        self.observation_space_img  = spaces.Box(low=0, high=255, shape=dummy_obs.repeat(n_stack, axis=0).shape, dtype=dummy_obs.dtype)
        self.observation_space_vec  = spaces.Box(low=0, high=1, shape=(num_skills, ), dtype=np.float32)
        self.observation_space = spaces.Dict(
            spaces={
                "img": self.observation_space_img,
                "vec": self.observation_space_vec
            }
        )
        
        self.action_space = self.env.action_space

    def reset(self, **kwargs):
        skill = np.random.randint(0,self.num_skills )
        self.env.reset()
        self.skill_vec = self.onehot(skill)
        img = self.env.render("rgb_array")
        self.frames_buffer = np.zeros((self.n_stack, *img.shape ))
        obs = {
            "img": self.update_frames_buffer(img),
            "vec": self.skill_vec}
        return obs

    def step(self, action):
        obs, reward, done, info = self.env.step(action)
        obs = {
            "img":self.update_frames_buffer(self.env.render("rgb_array")),
            "vec": self.skill_vec}
        return obs, reward, done, info

    def onehot(self, skill):
      skill_vec = np.zeros((self.num_skills))
      skill_vec[skill] = 1
      return skill_vec

    def update_frames_buffer(self, obs):
      self.frames_buffer[:-1] = self.frames_buffer[1:]
      self.frames_buffer[-1] = obs
      return self.frames_buffer
