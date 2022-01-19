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
from src.utils import record_video, best_skill, evaluate_mi, evaluate_pretrained_policy_intr, evaluate_pretrained_policy_ext
from src.mi_lower_bounds import mi_lower_bound
from src.config import conf

import matplotlib.pyplot as plt
import seaborn as sns
sns.set_theme(style="darkgrid")
sns.set(font_scale = conf.font_scale)


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
        # number of epochs
        self.epochs = hyerparams['n_epochs']
        # batch size
        self.batch_size = hyerparams['batch_size']
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
        # optimizer
        self.optimizer = opt.Adam(self.d.parameters(), lr=hyerparams['learning_rate'], weight_decay=self.weight_decay)
        # lower bound type
        self.lower_bound = hyerparams['lower_bound']
        # label smoothing
        self.label_smoothing = hyerparams['label_smoothing']
        # log baseline
        self.log_baseline = hyerparams['log_baseline']
        # alpha logit
        self.alpha_logit = hyerparams['alpha_logit']
        # gradient clip
        self.gradient_clip = hyerparams['gradient_clip']

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
                skill = torch.tensor(skill, device=conf.device)
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
    def mixup_reg(self, x1, x2, targets=False, m=None):
        # sample the mixing factor
        if m is None:
            m = np.random.beta(a=0.4, b=0.4)
        batch_size = self.batch_size
        if targets:
            y1 = torch.zeros((batch_size,self.n_skills))
            y1[torch.arange(batch_size), x1] = 1
            y2 = torch.zeros((batch_size,self.n_skills))
            y2[torch.arange(batch_size), x2] = 1
            return m*y1 + (1-m) * y2
        return m*x1 + (1-m) * x2, m

    def _on_rollout_end(self):
        """
        This event is triggered before updating the policy.
        """
        current_buffer_size = len(
                self.buffer) if self.on_policy else self.locals['replay_buffer'].buffer_size if self.locals['replay_buffer'].full else self.locals['replay_buffer'].pos
        if current_buffer_size >= self.min_buffer_size:
            epoch_loss = 0
            self.d.train()
            for epoch in range(self.epochs):
                # Extract the data
                if self.on_policy:
                    states, skills = self.buffer.sample(self.batch_size)
                else:
                    obs, _, _, _, _ = self.locals['replay_buffer'].sample(
                                    self.batch_size)
                    states, skills = self.split_obs(obs)
                # Make states require grad if gradient penalty is not none
                if self.gp:
                    states.requires_grad_(True)
                # forward pass
                if self.paramerization == "MLP" or  self.paramerization == "Linear":
                    scores = self.d(states)
                elif self.paramerization == "Separable" or  self.paramerization == "Concat":
                    # onehot encoding of the skills 
                    with torch.no_grad():
                        onehots_skills = torch.zeros(self.batch_size, self.n_skills)
                        onehots_skills[torch.arange(self.batch_size), skills] = 1
                    scores = self.d(states, onehots_skills)
                # compute the loss
                if self.lower_bound == "ba":
                    loss = nn.CrossEntropyLoss(label_smoothing=self.label_smoothing)(scores/self.temperature, skills.to(torch.long))
                elif self.lower_bound in ["tuba", "nwj", "nce", "interpolate"]:
                    mi = mi_lower_bound(scores, self.lower_bound, self.log_baseline, self.alpha_logit).mean()
                    loss = -mi
                # check if some regularizors will be used
                # Mixup regularization
                if self.mixup:
                    # sample a second batch
                    if self.on_policy:
                        states2, skills2 = self.buffer.sample(self.batch_size)
                    else:
                        obs, _, _, _, _ = self.locals['replay_buffer'].sample(
                                    self.batch_size)
                        states2, skills2 = self.split_obs(obs)
                    # compute the mixed inputs/targets
                    mixed_states, m = self.mixup_reg(states, states2)
                    mixed_skills, _ = self.mixup_reg(skills, skills2, targets=True, m=m)
                    scores = scores.detach()
                    scores = self.d(mixed_states)
                    loss = loss.detach()
                    loss = 0
                    if self.lower_bound == "ba":
                        loss = torch.mean(torch.sum(- mixed_skills * torch.log_softmax(scores/self.temperature, dim=-1), dim=-1))
                # gradient penalty
                if self.gp:
                    gp = self.grad_penalty(loss, states)
                    loss += (self.gp * gp)
                # take a gradient step
                self.optimizer.zero_grad()
                loss.backward()
                if self.gradient_clip:
                    clip_grad_norm_(self.d.parameters(), self.gradient_clip)
                self.optimizer.step()
                epoch_loss += loss.cpu().detach().item()
            epoch_loss /= self.epochs
            # logs
            grad_norms, weights_norm = self.gradient_norm(self.d)
            if (not self.on_policy) and self.num_timesteps % 100 == 0:
                self.sw.add_scalar("discrimiator loss",
                                            epoch_loss, self.num_timesteps)
                self.sw.add_scalar("discrimiator grad norm",
                                grad_norms, self.num_timesteps)
                self.sw.add_scalar("discrimiator wights norm",
                                weights_norm, self.num_timesteps)
                torch.save(self.d.state_dict(), self.save_dir + "/disc.pth")
            elif self.on_policy:
                self.sw.add_scalar("discrimiator loss",
                                            epoch_loss, self.num_timesteps)
                self.sw.add_scalar("discrimiator grad norm",
                                grad_norms, self.num_timesteps)
                self.sw.add_scalar("discrimiator wights norm",
                                weights_norm, self.num_timesteps)
                torch.save(self.d.state_dict(), self.save_dir + "/disc.pth")


class EvaluationCallback(BaseCallback):
    '''
    A callback to check the peroformance of the downstream task every N steps 
    '''

    def __init__(self, env_name, alg, discriminator, params, pm, n_samples=100):
        super().__init__()
        self.env_name = env_name
        self.params = params
        self.freq = round((params['pretrain_steps'] // n_samples)/8) * 8
        # print(f"freq:{self.freq}")
        # input()
        self.alg = alg
        self.d = discriminator
        self.skills = params['n_skills']
        self.pm = pm
        self.steps = []
        self.intr_rewards = []
        self.extr_rewards = []
        

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
        self.pretrained_policy = self.model
        if self.num_timesteps % self.freq == 0:
            print(f"Time step: {self.num_timesteps}")
            intr_reward = np.mean( [evaluate_pretrained_policy_intr(self.env_name, self.skills, self.pretrained_policy, self.d, self.pm, self.alg) for _ in range(5) ] )
            extr_reward = np.mean( [evaluate_pretrained_policy_ext(self.env_name, self.skills, self.pretrained_policy, self.alg) for _ in range(5)] )
            self.steps.append(self.num_timesteps)
            self.intr_rewards.append(intr_reward)
            self.extr_rewards.append(extr_reward)
            plt.figure()
            plt.plot(self.steps, self.extr_rewards, label=self.alg.upper())
            plt.xlabel("Pretraining Steps")
            plt.ylabel("Extrinsic Reward")
            plt.legend()
            filename = f"Scalability_Experiment_realtime_env:{self.env_name}_alg:{self.alg}_xaxis:Pretraining Steps.png"
            plt.savefig(f'{filename}', dpi=150) 
            plt.figure()
            plt.plot(self.intr_rewards, self.extr_rewards, label=self.alg.upper())
            plt.xlabel("Intrinsic Reward")
            plt.ylabel("Extrinsic Reward")
            plt.legend()
            filename = f"Scalability_Experiment_realtime_env:{self.env_name}_alg:{self.alg}_xaxis:Intrinsic Reward.png"
            plt.savefig(f'{filename}', dpi=150) 
        return True
        

# Callback for evaluation 
class MI_EvalCallback(BaseCallback):
    """
    A custom callback that derives from ``BaseCallback``.

    :param verbose: (int) Verbosity level 0: not output 1: info 2: debug
    """
    # (self.env_name, self.d, self.params, self.sw, self.discriminator_hyperparams['temperature'], self.directory, eval_freq=self.conf.eval_freq)
    def __init__(self, env_name, discriminator, params, tb_sw, discriminator_hyperparams, mi_estimator, model_save_path, eval_freq=5000, verbose=0):
        super(MI_EvalCallback, self).__init__(verbose)
        self.env_name = env_name
        self.n_skills = params['n_skills']
        self.tb_sw = tb_sw
        self.discriminator = discriminator
        self.eval_freq = eval_freq
        self.temp = discriminator_hyperparams['temperature']
        # mi_estimator is a named tuple with the following format
        # namedtuple('MI_Estimate', "estimator_func estimator_type log_baseline alpha_logit")
        self.mi_estimator = mi_estimator
        self.model_path = model_save_path
        self.best_reward = - np.inf
        self.results = []
        self.timesteps = []
        
        
    def _on_step(self):
        if self.num_timesteps % self.eval_freq == 0:
            evals_path = f"{self.model_path}/eval_results" 
            os.makedirs(evals_path, exist_ok=True)
            print(f"current timestep: {self.num_timesteps}")
            rewards = evaluate_mi(self.env_name, self.n_skills, self.model, self.tb_sw, self.discriminator, self.num_timesteps, self.temp, self.mi_estimator)
            if rewards.mean() > self.best_reward:
                self.best_reward = rewards.mean()
                self.model.save(f'{self.model_path}/best_model')
            self.timesteps.append(self.num_timesteps)
            self.results.append(rewards)
            timesteps = np.array(self.timesteps)
            # print(f"results: {self.results}")
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