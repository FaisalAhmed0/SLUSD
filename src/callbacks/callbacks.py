import gym
import os

import torch
import torch.nn as nn
import torch.optim as opt
import torch.nn.functional as F

from torch.nn.utils import clip_grad_norm_

import numpy as np

from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.callbacks import EvalCallback
from stable_baselines3.common.vec_env.dummy_vec_env import DummyVecEnv
from stable_baselines3 import PPO, SAC
from stable_baselines3.common.utils import get_schedule_fn
from stable_baselines3.common.monitor import Monitor


from src.environment_wrappers.env_wrappers import RewardWrapper, SkillWrapper, SkillWrapperFinetune
from src.utils import record_video, best_skill
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
            if (not self.on_policy) and self.num_timesteps%100 == 0:
                self.sw.add_scalar("discrimiator loss", epoch_loss, self.num_timesteps)
                self.sw.add_scalar("discrimiator grad norm", grad_norms, self.num_timesteps)
                self.sw.add_scalar("discrimiator wights norm", weights_norm, self.num_timesteps)
                torch.save(self.d.state_dict(), self.save_dir + "/disc.pth")
            else:
                self.sw.add_scalar("discrimiator loss", epoch_loss, self.num_timesteps)
                self.sw.add_scalar("discrimiator grad norm", grad_norms, self.num_timesteps)
                self.sw.add_scalar("discrimiator wights norm", weights_norm, self.num_timesteps)
                torch.save(self.d.state_dict(), self.save_dir + "/disc.pth")
                


class FineTuneCallback(BaseCallback):
    '''
    A callback to check the peroformance of the downstream task every N steps 
    '''
    def __init__(self, args, params, alg_params, conf, seed, alg, timestamp, n_samples=4):
        super().__init__()
        self.args = args
        self.env_name = args.env
        self.directory = None
        self.params = params
        self.alg_params = alg_params
        self.seed = seed
        self.conf = conf
        self.freq = params['pretrain_steps'] // n_samples
        # print(f"freq:{self.freq}")
        # input()
        self.timestamp = timestamp
        self.alg = alg
        
        
    def _on_step(self):
        """
        This method will be called by the model after each call to `env.step()`.

        For child callback (of an `EventCallback`), this will be called
        when the event is triggered.

        :return: (bool) If the callback returns False, training is aborted early.
        """
        # print(f"timestep: {self.num_timesteps}")
        # print(f"freq: {self.freq}")
        # print(f"rem: {self.num_timesteps % self.freq }")
        if self.num_timesteps % self.freq == 0:
            print(f"We finetune in timestep: {self.num_timesteps}")
            # input()
            self.finetune(self.num_timesteps)

        return True

    
    # finetune the pretrained policy on a specific task
    def finetune(self, steps):
        model_dir = self.conf.pretrain_steps_exper_dir + f"{self.args.env}/" + f"{self.alg}_{self.args.env}_skills:{self.params['n_skills']}_{self.timestamp}"
        self.directory = self.conf.pretrain_steps_exper_dir + f"{self.args.env}/" + f"{self.alg}_{self.args.env}_skills:{self.params['n_skills']}_pretrain_steps:{steps}_{self.timestamp}"
        os.makedirs(self.directory)
        env = DummyVecEnv([lambda: SkillWrapper(gym.make(
            self.env_name), self.params['n_skills'], max_steps=self.conf.max_steps)])
        model_dir = model_dir + "/best_model"
        if self.alg == "sac":
            model = SAC.load(model_dir, env=env, seed=self.seed)
        elif self.alg == "ppo":
            model = PPO.load(model_dir, env=env,  clip_range=get_schedule_fn(
                self.alg_params['clip_range']), seed=self.seed)

        best_skill_index = best_skill(
            model, self.env_name,  self.params['n_skills'])

        del model

        if self.alg == "sac":
            env = DummyVecEnv([lambda: SkillWrapperFinetune(Monitor(gym.make(
        self.env_name),  f"{self.directory}/finetune_train_results"), self.params['n_skills'], max_steps=gym.make(self.env_name)._max_episode_steps, skill=best_skill_index)])
            model = SAC('MlpPolicy', env, verbose=1,
                        learning_rate=self.alg_params['learning_rate'],
                        batch_size=self.alg_params['batch_size'],
                        gamma=self.alg_params['gamma'],
                        buffer_size=self.alg_params['buffer_size'],
                        tau=self.alg_params['tau'],
                        gradient_steps=self.alg_params['gradient_steps'],
                        learning_starts=self.alg_params['learning_starts'],
                        policy_kwargs=dict(net_arch=dict(pi=[self.conf.layer_size_policy, self.conf.layer_size_policy], qf=[
                                           self.conf.layer_size_value, self.conf.layer_size_value])),
                        tensorboard_log=self.directory,
                        )
        elif self.alg == "ppo":
            env = DummyVecEnv([lambda: SkillWrapperFinetune(Monitor(gym.make(
        self.env_name),  f"{self.directory}/finetune_train_results"), self.params['n_skills'], max_steps=gym.make(self.env_name)._max_episode_steps, skill=best_skill_index), 
                              lambda: SkillWrapperFinetune(Monitor(gym.make(
        self.env_name),  f"{self.directory}/finetune_train_results"), self.params['n_skills'], max_steps=gym.make(self.env_name)._max_episode_steps, skill=best_skill_index), lambda: SkillWrapperFinetune(Monitor(gym.make(
        self.env_name),  f"{self.directory}/finetune_train_results"), self.params['n_skills'], max_steps=gym.make(self.env_name)._max_episode_steps, skill=best_skill_index), lambda: SkillWrapperFinetune(Monitor(gym.make(
        self.env_name),  f"{self.directory}/finetune_train_results"), self.params['n_skills'], max_steps=gym.make(self.env_name)._max_episode_steps, skill=best_skill_index)])
            
            model = PPO('MlpPolicy', env, verbose=1,
                        learning_rate=self.alg_params['learning_rate'],
                        n_steps=self.alg_params['n_steps'],
                        batch_size=self.alg_params['batch_size'],
                        n_epochs=self.alg_params['n_epochs'],
                        gamma=self.alg_params['gamma'],
                        gae_lambda=self.alg_params['gae_lambda'],
                        clip_range=self.alg_params['clip_range'],
                        policy_kwargs=dict(activation_fn=nn.Tanh,
                                           net_arch=[dict(pi=[self.conf.layer_size_policy, self.conf.layer_size_policy], vf=[self.conf.layer_size_value, self.conf.layer_size_value])]),
                        tensorboard_log=self.directory,

                        )
        
        # env = DummyVecEnv([lambda: SkillWrapperFinetune(Monitor(gym.make(
        # self.env_name),  f"{self.directory}/finetune_train_results"), self.params['n_skills'], max_steps=gym.make(self.env_name)._max_episode_steps, skill=best_skill_index)])

        eval_env = SkillWrapperFinetune(gym.make(
            self.env_name), self.params['n_skills'], max_steps=gym.make(self.env_name)._max_episode_steps, skill=best_skill_index)
        eval_env = Monitor(eval_env, f"{self.directory}/finetune_eval_results")
        
        eval_callback = EvalCallback(eval_env, best_model_save_path=self.directory + f"/best_finetuned_model_skillIndex:{best_skill_index}",
                                    log_path=f"{self.directory}/finetune_eval_results", eval_freq=1000,
                                    deterministic=True, render=False)
        if self.alg == "sac":
            model = SAC.load(model_dir, env=env, tensorboard_log=self.directory)
            model.learn(total_timesteps=self.params['finetune_steps'],
                        callback=eval_callback, tb_log_name="SAC_FineTune")
        elif self.alg == "ppo":
            model = PPO.load(model_dir, env=env, tensorboard_log=self.directory,
                            clip_range=get_schedule_fn(self.alg_params['clip_range']))

            model.learn(total_timesteps=self.params['finetune_steps'],
                        callback=eval_callback, tb_log_name="PPO_FineTune")

        # record a video of the agent
        # record_video_finetune(self.env_name, best_skill_index, model,
        #                     self.params['n_skills'], video_length=1000, video_folder='videos_finetune/', alg=self.alg)


