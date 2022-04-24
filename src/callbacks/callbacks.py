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
from src.utils import record_video, best_skill, evaluate_mi, evaluate_pretrained_policy_intr, evaluate_pretrained_policy_ext, evaluate_adapted_policy
from src.mi_lower_bounds import mi_lower_bound
from src.config import conf

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
sns.set()
sns.set_style('whitegrid')
sns.set_context("paper", font_scale = conf.font_scale)


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
        # print("grad penalty")
        # print(gradient_penalty)
        # input()
        return gradient_penalty

    # TODO: Add mixup r
    @torch.no_grad()
    def mixup_reg(self, x1, x2, targets=False, m=None):
        # print("Mixup")
        # sample the mixing factor
        if m is None:
            m = np.random.beta(a=0.4, b=0.4)
        batch_size = self.batch_size
        if targets:
            # print(x1, x2)
            y1 = torch.zeros((batch_size,self.n_skills))
            y1[torch.arange(batch_size), x1.to(torch.long)] = 1
            y2 = torch.zeros((batch_size,self.n_skills))
            y2[torch.arange(batch_size), x2.to(torch.long)] = 1
            return m*y1 + (1-m) * y2, None
        return m * x1 + (1-m) * x2, m

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

    def __init__(self, env_name, alg, discriminator, adaptation_params, params, pm, seed, directory, tb, n_samples=100):
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
        self.seed = seed
        self.directory = directory
        self.tb = tb
        self.adapt_params = adaptation_params
        
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
        # print(f"Time step: {self.num_timesteps}")
        if self.num_timesteps % self.freq == 0 or self.num_timesteps == 8 or self.num_timesteps == 1 :
            # print(self.num_timesteps == 0)
            # print(f"Time step: {self.num_timesteps}")
            intr_reward = np.mean( [evaluate_pretrained_policy_intr(self.env_name, self.skills, self.pretrained_policy, self.d, self.pm, self.alg) for _ in range(5) ] )
            adapted_model, bestskill = self.finetune()
            extr_reward = np.mean([evaluate_adapted_policy(self.env_name, self.skills, bestskill, adapted_model, self.alg)])
            # extr_reward = np.mean( [evaluate_pretrained_policy_ext(self.env_name, self.skills, self.pretrained_policy, self.alg) for _ in range(5)] )
            self.steps.append(self.num_timesteps)
            self.intr_rewards.append(intr_reward)
            self.extr_rewards.append(extr_reward)
            self.tb.add_scalar("Extrinsic Reward (Best Skill)", extr_reward, self.num_timesteps)
            
            #####
            timesteps = np.array(self.steps)
            # print(f"results: {self.results}")
            intr_rewards = np.array(self.intr_rewards)
            extr_rewards = np.array(self.extr_rewards)
            np.savez(f"{self.directory}/scalability_evaluations.npz", timesteps=timesteps, intr_rewards=intr_rewards, extr_rewards=extr_rewards)
            #####
            
            plt.figure()
            plt.plot(self.steps, self.extr_rewards, label=self.alg.upper())
            plt.xlabel("Pretraining Steps")
            plt.ylabel("Extrinsic Reward (Best Skill)")
            plt.legend()
            plt.tight_layout()
            filename = f"{self.directory}/Scalability_Experiment_realtime_env:{self.env_name}_alg:{self.alg}_xaxis:Pretraining Steps, seed:{self.seed}.png"
            plt.savefig(f'{filename}', dpi=150) 
            plt.figure()
            plt.scatter(self.intr_rewards, self.extr_rewards, label=self.alg.upper())
            plt.xlabel("Intrinsic Reward")
            plt.ylabel("Extrinsic Reward (Best Skill)")
            plt.legend()
            plt.tight_layout()
            filename = f"{self.directory}/Scalability_Experiment_realtime_env:{self.env_name}_alg:{self.alg}_xaxis:Intrinsic Reward, seed:{self.seed}.png"
            plt.savefig(f'{filename}', dpi=150) 
            
        return True
    
    def finetune(self):
        # pick the best skill
        bestskill = best_skill(
            self.model, self.env_name,  self.skills)
        
        # create the adaptation environment
        env, eval_env = self.adaptation_environment(bestskill)
        
        eval_callback = EvalCallback(eval_env, best_model_save_path=self.directory + f"/best_finetuned_model_skillIndex:{bestskill}",
                                    log_path=f"{self.directory}/finetune_eval_results", eval_freq=25000,
                                    deterministic=True, render=False)
        
        
        # create the adapration model
        self.adaptation_model = SAC('MlpPolicy', env, verbose=1,
                    learning_rate=self.adapt_params['learning_rate'],
                    batch_size=self.adapt_params['batch_size'],
                    gamma=self.adapt_params['gamma'],
                    buffer_size=self.adapt_params['buffer_size'],
                    tau=self.adapt_params['tau'],
                    ent_coef=self.adapt_params['ent_coef'],
                    gradient_steps=self.adapt_params['gradient_steps'],
                    learning_starts=self.adapt_params['learning_starts'],
                    policy_kwargs=dict(net_arch=dict(pi=[conf.layer_size_policy, conf.layer_size_policy], qf=[
                                       conf.layer_size_q, conf.layer_size_q]), n_critics=2),
                    tensorboard_log=self.directory,
                    seed=self.seed
                    )
        if self.alg == "sac":
            sac_model = self.model
            
            self.adaptation_model.actor.latent_pi.load_state_dict(sac_model.actor.latent_pi.state_dict())
            self.adaptation_model.actor.mu.load_state_dict(sac_model.actor.mu.state_dict())
            self.adaptation_model.actor.log_std.load_state_dict(sac_model.actor.log_std.state_dict())
            self.adaptation_model.actor.optimizer = opt.Adam(self.adaptation_model.actor.parameters(), lr=self.adapt_params['learning_rate'])
            
            self.adaptation_model.critic.qf0.load_state_dict(sac_model.critic.qf0.state_dict())
            self.adaptation_model.critic.qf1.load_state_dict(sac_model.critic.qf1.state_dict())
            self.adaptation_model.critic_target.load_state_dict(self.adaptation_model.critic.state_dict())
            self.adaptation_model.critic.optimizer = opt.Adam(self.adaptation_model.critic.parameters(), lr=self.adapt_params['learning_rate'])
            
            self.adaptation_model.learn(total_timesteps=self.params['finetune_steps'],
                        callback=eval_callback, tb_log_name="SAC_FineTune", d=None, mi_estimator=None)
            
            return self.adaptation_model, bestskill
        
        elif self.alg == "ppo":
            ppo_model = self.model
            ppo_actor = ppo_model.policy
            # initlialize the adaptation policy with the pretrained model
            self.adaptation_model.actor.latent_pi.load_state_dict(ppo_actor.mlp_extractor.policy_net.state_dict())
            self.adaptation_model.actor.mu.load_state_dict(ppo_actor.action_net.state_dict())
            self.adaptation_model.actor.log_std.load_state_dict(nn.Linear(in_features=conf.layer_size_policy, out_features=gym.make(self.env_name).action_space.shape[0], bias=True).state_dict())
            self.adaptation_model.actor.optimizer = opt.Adam(self.adaptation_model.actor.parameters(), lr=self.adapt_params['learning_rate'])
            # initlialize the adaptation critic with the discriminator weights
            layers = [nn.Linear(gym.make(self.env_name).observation_space.shape[0] + self.params['n_skills']  + gym.make(self.env_name).action_space.shape[0], conf.layer_size_discriminator)]
            for l in range(len(ppo_model.policy.mlp_extractor.value_net)):
                if l != 0:
                    layers.append(ppo_model.policy.mlp_extractor.value_net[l])
            layers.append(nn.Linear(conf.layer_size_discriminator, 1))
            seq = nn.Sequential(*layers)
            # print(seq)
            # input()
            self.adaptation_model.critic.qf0.load_state_dict(seq.state_dict())
            self.adaptation_model.critic.qf1.load_state_dict(seq.state_dict())
            self.adaptation_model.critic_target.load_state_dict(self.adaptation_model.critic.state_dict())
            self.adaptation_model.critic.optimizer = opt.Adam(self.adaptation_model.critic.parameters(), lr=self.adapt_params['learning_rate'])
            
            self.adaptation_model.learn(total_timesteps=self.params['finetune_steps'],
                        callback=eval_callback, tb_log_name="PPO_FineTune", d=None, mi_estimator=None)
            
            return self.adaptation_model, bestskill
        
    def adaptation_environment(self, best_skill_index):
        '''
        Create the adaptation environment according to the task reward
        '''
        env = DummyVecEnv([lambda: SkillWrapperFinetune(Monitor(gym.make(
    self.env_name),  f"{self.directory}/finetune_train_results"), self.params['n_skills'], r_seed=None,max_steps=gym.make(self.env_name)._max_episode_steps, skill=best_skill_index)])
        eval_env = SkillWrapperFinetune(gym.make(
            self.env_name), self.params['n_skills'], max_steps=gym.make(self.env_name)._max_episode_steps, r_seed=None, skill=best_skill_index)
            
        return env, eval_env
            
            
            
        
        
        
        

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