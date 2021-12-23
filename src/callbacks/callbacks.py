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
from src.utils import record_video, best_skill, evaluate_cpc
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
            record_video(self._eval_env_id, self.model, n_z=self.n_z,
                         video_folder=self.videos_dir, n_calls=self.n_calls)
        return True


class DiscriminatorCallback(BaseCallback):
    """
    A custom callback that derives from ``BaseCallback``.

    :param verbose: (int) Verbosity level 0: not output 1: info 2: debug
    """
    # TODO: add smoothing parameter
    # TODO: add gradient penalties gp
    # TODO: add mixup regularization
    # TODO: add parametrization: 0 -> MLP, 1 -> dot_based CPC style

    def __init__(self, discriminator, buffer, hyerparams, sw, n_skills, min_buffer_size=2048, verbose=0, save_dir="./",
                 on_policy=False):
        super(DiscriminatorCallback, self).__init__(verbose)
        # classifier
        self.d = discriminator
        # data buffer
        self.buffer = buffer
        # optimization hyperparameters
        self.hyerparams = hyerparams
        # TODO: Added Weight decay
        self.weight_decay = hyerparams['weight_decay']
        # optimizer
        print(f"Weight decay value is {self.weight_decay}")
        self.optimizer = opt.Adam(
            self.d.parameters(), lr=hyerparams['learning_rate'], weight_decay=self.weight_decay)
        # number of epochs
        self.epochs = hyerparams['n_epochs']
        # batch size
        self.batch_size = hyerparams['batch_size']
        # use label smoothing if not None, otherwise use cross_entropy
        if (hyerparams['parametrization'] == "MLP" or hyerparams['parametrization']== "linear" ) and hyerparams['label_smoothing']:
            self.criterion = self.label_smoothedCrossEntropyLoss
        elif hyerparams['parametrization']== "MLP" or hyerparams['parametrization']== "linear":
            self.criterion = F.cross_entropy
        # tensorboard summary writer
        self.sw = sw
        # number of skills
        self.n_skills = n_skills
        # minimum size of the buffer before training
        self.min_buffer_size = min_buffer_size
        # directory to save the model parameters
        self.save_dir = save_dir
        # boolean for the algorithm type
        self.on_policy = on_policy
        # gradient penalty
        self.gp = hyerparams['gp']
        # mixup regularization
        self.mixup = hyerparams['mixup']
        # parametrization 
        self.paramerization = hyerparams['parametrization']
        # temperature
        self.temperature = self.hyerparams['temperature']

    def _on_step(self):
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
            skill, = np.where(obs[-self.n_skills:] == 1)
            return env_obs, skill[0]
        else:
            env_obs = torch.clone(obs[:, : -self.n_skills])
            skills = torch.argmax(obs[:, -self.n_skills:], dim=1)
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

    # TODO: Add a label smoothing function
    # most of the function is from https://github.com/pytorch/pytorch/issues/7455#issuecomment-513062631
    def label_smoothedCrossEntropyLoss(self, preds, targets, smoothing=0.2):
        num_classes = preds.shape[1]
        # print(f"Number of classes in the label_smoothedCrossEntropyLoss: {num_classes}")
        # print(f"targets: {targets}")
        with torch.no_grad():
            smooth_target = torch.zeros_like(preds)
            smooth_target.fill_(smoothing/(num_classes-1))
            smooth_target.scatter_(1, targets.unsqueeze(1), 1 - smoothing)
        # print(f"targets shape : {targets.shape}")
        # print(f"Actual labels: {targets[:3]}")
        # print(f"smoothed labels: {smooth_target[:3]}")
        # input()
        return torch.mean(torch.sum(- smooth_target * torch.log_softmax(preds, dim=-1), dim=-1))

    # TODO: added onehot cross entropy
    def onehot_cross_entropy(self, preds, targets):
        preds_logs = torch.log_softmax(preds, dim=-1)
        with torch.no_grad():
            onehots = torch.zeros(self.batch_size, preds.shape[1])
            print(f"onehot shape: {onehots.shape}")
            print(f"targets: {targets}")
            onehots[torch.arange(self.batch_size), targets.squeeze().to(torch.int64)] = 1
        return torch.mean(torch.sum(- onehots * preds_logs, dim=1))


    # TODO: added gradient penalty function
    def grad_penalty(self, loss, inputs):
        '''
        A function to compute the gradient penalty
        loss: the loss value
        input: the inputs used in the loss computation
        '''
        grads = torch.autograd.grad(loss, inputs, retain_graph=True, allow_unused=True)
        # print(grad)
        grad_norm = 0
        for grad in grads:
            grad_norm += torch.norm(grad)
        gradient_penalty = (grad_norm - 1).pow(2)
        return gradient_penalty

    # TODO: Add mixup r
    @torch.no_grad()
    def mixup_reg(self, x1, x2, targets=False):
        # sample the mixing factor
        m = np.random.beta(a=0.4, b=0.4)
        batch_size = self.batch_size
        if targets:
            y1 = torch.zeros((batch_size,self.n_skills))
            y1[torch.arange(batch_size), x1] = 1
            y2 = torch.zeros((batch_size,self.n_skills))
            y2[torch.arange(batch_size), x2] = 1
            return m*y1 + (1-m) * y2
        return m*x1 + (1-m) * x2

    def _on_rollout_end(self):
        """
        This event is triggered before updating the policy.
        """
        # MLP case
        if self.paramerization == "MLP" or self.paramerization == "linear":
            current_buffer_size = len(
                self.buffer) if self.on_policy else self.locals['replay_buffer'].buffer_size if self.locals['replay_buffer'].full else self.locals['replay_buffer'].pos
            if current_buffer_size >= self.min_buffer_size:
                # initlize the average losses
                epoch_loss = 0
                self.d.train()
                for _ in range(1):
                    if self.on_policy:
                        inputs, targets = self.buffer.sample(self.batch_size)
                    else:
                        obs, _, _, _, _ = self.locals['replay_buffer'].sample(
                            self.batch_size)
                        inputs, targets = self.split_obs(obs)
                    if self.gp:
                        inputs.requires_grad_(True)
                    outputs = self.d(inputs.to(conf.device))
                    loss = self.criterion(outputs/self.temperature, targets.to(
                        conf.device).to(torch.int64))
                    # TODO: add mixup
                    if self.mixup:
                        if self.on_policy:
                            inputs2, targets2 = self.buffer.sample(self.batch_size)
                        else:
                            obs, _, _, _, _ = self.locals['replay_buffer'].sample(
                            self.batch_size)
                            inputs2, targets2 = self.split_obs(obs)
                        inputs = self.mixup_reg(inputs, inputs2)
                        targets = self.mixup_reg(targets, targets2, targets=True)
                        loss = 0
                        loss = torch.mean(torch.sum(- targets * torch.log_softmax(outputs, dim=-1), dim=-1))
                    # TODO: Add gradient penalty to the loss
                    if self.gp:
                        # inputs.requires_grad_(True)
                        gp = self.grad_penalty(loss, inputs)
                        loss += (self.gp * gp)
                    self.optimizer.zero_grad()
                    loss.backward()
                    # clip_grad_norm_(model.parameters(), 0.1)
                    self.optimizer.step()
                    epoch_loss += loss.cpu().detach().item()

                grad_norms, weights_norm = self.gradient_norm(self.d)
                epoch_loss /= 1
                if (not self.on_policy) and self.num_timesteps % 100 == 0:
                    self.sw.add_scalar("discrimiator loss",
                                    epoch_loss, self.num_timesteps)
                    self.sw.add_scalar("discrimiator grad norm",
                                    grad_norms, self.num_timesteps)
                    self.sw.add_scalar("discrimiator wights norm",
                                    weights_norm, self.num_timesteps)
                    torch.save(self.d.state_dict(), self.save_dir + "/disc.pth")
                else:
                    self.sw.add_scalar("discrimiator loss",
                                    epoch_loss, self.num_timesteps)
                    self.sw.add_scalar("discrimiator grad norm",
                                    grad_norms, self.num_timesteps)
                    self.sw.add_scalar("discrimiator wights norm",
                                    weights_norm, self.num_timesteps)
                    torch.save(self.d.state_dict(), self.save_dir + "/disc.pth")
        elif self.paramerization == "CPC":
            current_buffer_size = len(
                self.buffer) if self.on_policy else self.locals['replay_buffer'].buffer_size if self.locals['replay_buffer'].full else self.locals['replay_buffer'].pos
            if current_buffer_size >= self.min_buffer_size:
                epoch_loss = 0
                self.d.train()
                for _ in range(1):
                    if self.on_policy:
                        inputs, targets = self.buffer.sample(self.batch_size)
                    else:
                        obs, _, _, _, _ = self.locals['replay_buffer'].sample(
                            self.batch_size)
                        states, skills = self.split_obs(obs)
                    # onehot encoding of the skills 
                    with torch.no_grad():
                        onehots_skills = torch.zeros(self.batch_size, self.n_skills)
                        onehots_skills[torch.arange(self.batch_size), skills] = 1
                    outputs = self.d(states.to(conf.device), onehots_skills.to(conf.device))
                    # print(f"output in the callback: {outputs}")
                    loss = torch.nn.CrossEntropyLoss()(outputs/self.temperature, target=torch.arange(self.batch_size))
                    self.optimizer.zero_grad()
                    loss.backward()
                    self.optimizer.step()
                    epoch_loss += loss.cpu().detach().item()

                grad_norms, weights_norm = self.gradient_norm(self.d)
                epoch_loss /= 1
                if (not self.on_policy) and self.num_timesteps % 100 == 0:
                    self.sw.add_scalar("discrimiator loss",
                                    epoch_loss, self.num_timesteps)
                    self.sw.add_scalar("discrimiator grad norm",
                                    grad_norms, self.num_timesteps)
                    self.sw.add_scalar("discrimiator wights norm",
                                    weights_norm, self.num_timesteps)
                    torch.save(self.d.state_dict(), self.save_dir + "/disc.pth")
                else:
                    self.sw.add_scalar("discrimiator loss",
                                    epoch_loss, self.num_timesteps)
                    self.sw.add_scalar("discrimiator grad norm",
                                    grad_norms, self.num_timesteps)
                    self.sw.add_scalar("discrimiator wights norm",
                                    weights_norm, self.num_timesteps)
                    torch.save(self.d.state_dict(), self.save_dir + "/disc.pth")



class FineTuneCallback(BaseCallback):
    '''
    A callback to check the peroformance of the downstream task every N steps 
    '''

    def __init__(self, args, params, alg_params, conf, seed, alg, timestamp, n_samples=100):
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
        model_dir = self.conf.pretrain_steps_exper_dir + \
            f"{self.args.env}/" + \
            f"{self.alg}_{self.args.env}_skills:{self.params['n_skills']}_{self.timestamp}"
        self.directory = self.conf.pretrain_steps_exper_dir + \
            f"{self.args.env}/" + \
            f"{self.alg}_{self.args.env}_skills:{self.params['n_skills']}_pretrain_steps:{steps}_{self.timestamp}"
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
            model = SAC.load(model_dir, env=env,
                             tensorboard_log=self.directory)
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

# Callback for evaluation 
class CPC_EvalCallback(BaseCallback):
    """
    A custom callback that derives from ``BaseCallback``.

    :param verbose: (int) Verbosity level 0: not output 1: info 2: debug
    """
    def __init__(self, env_name, n_skills, tb_sw, discriminator, temp, model_save_path, eval_freq=5000,verbose=0):
        super(CPC_EvalCallback, self).__init__(verbose)
        self.env_name = env_name
        self.n_skills = n_skills
        self.tb_sw = tb_sw
        self.discriminator = discriminator
        self.eval_freq = eval_freq
        self.temp = temp
        self.model_path = model_save_path
        self.best_reward = - np.inf
        self.results = []
        self.timesteps = []
        
        
    def _on_step(self):
        if self.num_timesteps % self.eval_freq == 0:
            evals_path = f"{self.model_path}/eval_results" 
            os.makedirs(evals_path, exist_ok=True)
            print(f"current timestep: {self.num_timesteps}")
            rewards = evaluate_cpc(self.env_name, self.n_skills, self.model, self.tb_sw, self.discriminator, self.num_timesteps, self.temp)
            if rewards.mean() > self.best_reward:
                self.best_reward = rewards.mean()
                self.model.save(f'{self.model_path}/best_model')
            self.timesteps.append(self.num_timesteps)
            self.results.append(rewards)
            timesteps = np.array(self.timesteps)
            results = np.array(self.results)
            np.savez(f"{evals_path}/evaluations.npz", timesteps=timesteps, results=results)
        return True

    def _on_training_end(self) -> None:
        """
        This event is triggered before exiting the `learn()` method.
        """
        pass
        # print("I am heere")
        # evaluate_cpc(self.env_name, self.n_skills, self.model, self.tb_sw, self.discriminator, self.num_timesteps)