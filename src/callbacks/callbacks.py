import torch
import torch.optim as opt
import torch.nn.functional as F

from torch.nn.utils import clip_grad_norm_

import numpy as np

from stable_baselines3.common.callbacks import BaseCallback

from src.utils import record_video
from src.config import conf

# callback to save a video while training
class VideoRecorderCallback(BaseCallback):
    def __init__(self, eval_env_id: str, record_freq: int, n_z: int, deterministic: bool = True, videos_dir="./training_videos"):
        """
        Records a video of an agent's trajectory traversing ``eval_env`` and logs it to TensorBoard

        :param eval_env: A gym environment from which the trajectory is recorded
        :param render_freq: Render the agent's trajectory every eval_freq call of the callback.
        :param n_eval_episodes: Number of episodes to render
        :param deterministic: Whether to use deterministic or stochastic policy
        """
        super().__init__()
        self._eval_env_id = eval_env_id
        self._record_freq = record_freq
        self._deterministic = deterministic
        self.videos_dir = videos_dir
        self.n_z = n_z

    def _on_step(self) -> bool:
        if self.n_calls % self._record_freq == 0:
          record_video(self._eval_env_id, self.model, n_z=self.n_z, video_folder=self.videos_dir, n_calls=self.n_calls)            
        return True

class DiscriminatorCallback(BaseCallback):
    """
    A custom callback that derives from ``BaseCallback``.

    :param verbose: (int) Verbosity level 0: not output 1: info 2: debug
    """
    def __init__(self, discriminator, buffer, hyerparams , sw, n_skills, min_buffer_size=2048, verbose=0, save_dir="./", on_policy=False):
        super(DiscriminatorCallback, self).__init__(verbose)
        self.d = discriminator
        self.buffer = buffer
        self.hyerparams = hyerparams
        self.optimizer = opt.Adam(self.d.parameters(), lr=hyerparams['learning_rate'])
        self.epochs = hyerparams['n_epochs']
        self.batch_size = hyerparams['batch_size']
        self.criterion = F.cross_entropy
        self.sw = sw
        self.n_skills = n_skills
        self.min_buffer_size = min_buffer_size
        self.save_dir = save_dir
        self.on_policy = on_policy
        print("brkbgkl")
    def _on_step(self) -> bool:
        """
        This method will be called by the model after each call to `env.step()`.

        For child callback (of an `EventCallback`), this will be called
        when the event is triggered.

        :return: (bool) If the callback returns False, training is aborted early.
        """
        if self.on_policy:
          for obs_vec in self.locals['obs_tensor']:
            env_obs, skill = self.split_obs(obs_vec)
            obs = torch.clone(env_obs)
            skill = torch.tensor(skill)
            self.buffer.add(obs, skill)

        return True

    def split_obs(self, obs):
        if self.on_policy:
          obs = obs.cpu()
          env_obs = obs[: -self.n_skills]
          skill, = np.where( obs[-self.n_skills: ] == 1 )
          return env_obs, skill[0]
        else:
          env_obs = torch.clone(obs[:, : -self.n_skills])
          skills  =  torch.argmax(obs[:, -self.n_skills: ], dim=1)
          return env_obs, skills

    def gradient_norm(self, net):
        '''
        gradient_norm(net)
        This function calulate the gradient norm of a neural network model.
        net: network model
        '''
        total_norm = 0
        total_weights_norm = 0
        for param in net.parameters():
          param_norm = param.grad.detach().data.norm(2)
          total_weights_norm += param.detach().data.norm(2) ** 2
          total_norm += param_norm.item() ** 2
        return total_norm**0.5, total_weights_norm**0.5
          

    def _on_rollout_end(self) -> None:
        """
        This event is triggered before updating the policy.
        """
        current_buffer_size = len(self.buffer) if self.on_policy else self.locals['replay_buffer'].buffer_size if self.locals['replay_buffer'].full else self.locals['replay_buffer'].pos
        if current_buffer_size >= self.min_buffer_size:
            # initlize the average losses
            epoch_loss = 0
            for _ in range(1):
              if self.on_policy:
                inputs, targets = self.buffer.sample(self.batch_size)
              else:
                obs, _, _, _, _= self.locals['replay_buffer'].sample(self.batch_size)
                inputs, targets = self.split_obs(obs)
              outputs = self.d(inputs.to(conf.device))
              loss = self.criterion(outputs, targets.to(conf.device).to(torch.int64))
              self.optimizer.zero_grad()
              loss.backward()
              # clip_grad_norm_(model.parameters(), 0.1)
              self.optimizer.step()
              epoch_loss += loss.cpu().detach().item()
            
            grad_norms, weights_norm = self.gradient_norm(self.d)
            
            epoch_loss /= 1
            self.sw.add_scalar("discrimiator loss", epoch_loss, self.num_timesteps)
            self.sw.add_scalar("discrimiator grad norm", grad_norms, self.num_timesteps)
            self.sw.add_scalar("discrimiator wights norm", weights_norm, self.num_timesteps)
            torch.save(self.d.state_dict(), self.save_dir + "/disc.pth")





