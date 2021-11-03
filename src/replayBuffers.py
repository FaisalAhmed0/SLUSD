import numpy as np
import torch

# data bufffer for the discriminator
class DataBuffer():
  '''
  DataBuffer(size)
  A simple data Buffer to sotre transition during the training,
  size: the max length of the buffer
  obs_shape: size of the observation vector
  '''
  def __init__(self, size, obs_shape):
    self.size = size
    # Create the buffer as a Dequeue for efficient appending and poping
    self.observations = np.zeros((size, obs_shape))
    self.skills = np.zeros(size)
    self.max_size = size
    self.t = 0
    self.size = 0

  def add(self, obs, skill):
    self.observations[self.t] = obs
    self.skills[self.t] = skill
    self.t = (self.t + 1) % self.max_size
    if self.size < self.max_size:
      self.size += 1
    
  def sample(self, sample_size):
    assert sample_size <= self.size, "Sample size is larger than the buffer size"
    indinces = np.random.randint(0, self.size, sample_size)
    inputs = torch.FloatTensor(self.observations[indinces])
    outputs = torch.tensor(self.skills[indinces])
    return inputs, outputs

  def __len__(self):
    return self.size