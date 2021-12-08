import gym
import numpy as np

class HalfCheetahTaskWrapper(gym.Wrapper):
    def __init__(self, env, task="run_forward"):
        '''
        3 tasks:
            run_forward
            run_backward
            jump
        '''
        
        # Call the parent constructor, so we can access self.env later
        super(RewardWrapper, self).__init__(env)
        # check that the environment is HalfCheetah-v2
        assert env.spec.id == "HalfCheetah-v2", "Environment is {env.spec.id} not HalfCheetah-v2"
        self.env = env
        self.task = task
    
    def reset(self):
        obs = self.env.reset()
        # initial z coordinate
        self.initial_z = self.sim.data.qpos[1]
        return obs
  
    def step(self, action):
        x_pos_before = self.sim.data.qpos[0]
        obs, reward, done, info = self.env.step(action)
        x_pos_after = self.sim.data.qpos[0]
        z_pos = self.sim.data.qpos[1]
        v_x = (x_pos_after - x_pos_before) / self.dt # velocity in the x direction
        if self.task == "run_forward":
            reward = - (np.abs(v_x - 3) - 0.1 * np.power(action, 2).sum())
        elif self.task == "run_backward":
            reward = np.abs(v_x - 3) - 0.1 * np.power(action, 2).sum()
        elif self.task == "jump":
            reward = 15 * (z_pos - self.initial_z)  - 0.1 * np.power(action, 2).sum() 
        return obs, reward, done, info

    
# add for the Walker-2D and Ant after testing them