import gym

from stable_baselines3.common.vec_env.dummy_vec_env import DummyVecEnv
from stable_baselines3 import PPO, SAC
from stable_baselines3.common.callbacks import EvalCallback
from stable_baselines3.common.utils import get_schedule_fn
from stable_baselines3.common.monitor import Monitor

from src.environment_wrappers.env_wrappers import RewardWrapper, SkillWrapper, SkillWrapperFinetune
from src.environment_wrappers.tasks_wrappers import HalfCheetahTaskWrapper
from src.utils import record_video_finetune, best_skill
from src.mi_lower_bounds import mi_lower_bound
from src.models.models import Discriminator, SeparableCritic, ConcatCritic
from src.replayBuffers import DataBuffer
from src.callbacks.callbacks import DiscriminatorCallback, VideoRecorderCallback, FineTuneCallback, MI_EvalCallback

import torch
import torch.nn as nn
from torch.utils.tensorboard import SummaryWriter
import copy

import time

from collections import namedtuple


class DIAYN():
    def __init__(self, params, alg_params, discriminator_hyperparams, env="MountainCarContinuous-v0", alg="ppo", directory="./", seed=10, conf=None, timestamp=None, checkpoints=False, args=None, task=None, adapt_params=None):
        # create the discirminator
        state_dim = gym.make(env).observation_space.shape[0]
        skill_dim = params['n_skills']
        hidden_dims = conf.num_layers_discriminator*[conf.layer_size_discriminator]
        latent_dim = conf.latent_size
        temperature = discriminator_hyperparams['temperature']
        dropout = discriminator_hyperparams['dropout']
        if discriminator_hyperparams['parametrization'] == "MLP":
            self.d = Discriminator(state_dim, hidden_dims, skill_dim, dropout=discriminator_hyperparams['dropout']).to(conf.device)
        elif discriminator_hyperparams['parametrization'] == "Linear":
            self.d = nn.Linear(state_dim, skill_dim).to(conf.device)
        elif discriminator_hyperparams['parametrization'] == "Separable":
            # state_dim, skill_dim, hidden_dims, latent_dim, temperature=1, dropout=None)
            self.d = SeparableCritic(state_dim, skill_dim, hidden_dims, latent_dim, temperature, dropout).to(conf.device)
        elif discriminator_hyperparams['parametrization'] == "Concat":
            # state_dim, skill_dim, hidden_dims, temperature=1, dropout=None)
            self.d = ConcatCritic(state_dim, skill_dim, hidden_dims, temperature, dropout).to(conf.device)
        else:
            raise ValueError(f"{discriminator_hyperparams['parametrization']} is invalid parametrization")
        # Create the data buffer for the discriminator
        self.buffer = DataBuffer(params['buffer_size'], obs_shape=state_dim)
            
        # tensorboard summary writer
        self.sw = SummaryWriter(
            log_dir=directory, comment=f"{alg} Discriminator, env_name:{env}")

        # save some attributes
        self.alg = alg
        self.params = params
        self.alg_params = alg_params
        self.discriminator_hyperparams = discriminator_hyperparams
        self.env_name = env
        self.directory = directory
        self.seed = seed
        self.timestamp = timestamp
        self.conf = conf
        self.checkpoints = checkpoints
        self.args = args
        self.task = task
        self.parametrization = discriminator_hyperparams['parametrization']
        self.adapt_params = adapt_params

    def pretrain(self):
        if self.alg == "ppo":
            env = DummyVecEnv([lambda: Monitor(RewardWrapper(SkillWrapper(gym.make(
                    self.env_name), self.params['n_skills'], max_steps=self.conf.max_steps), self.d, self.params['n_skills'], parametrization=self.parametrization),  self.directory)]*self.alg_params['n_actors'])

            # create the model with the speicifed hyperparameters
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
                        seed=self.seed
                        )
            
            discriminator_callback = DiscriminatorCallback(self.d, self.buffer, self.discriminator_hyperparams,
                                                           sw=self.sw, n_skills=self.params['n_skills'], min_buffer_size=self.params['min_train_size'], save_dir=self.directory, on_policy=True)

            # TODO: Replace this when you implement the other MI lower bounds with on policy algorithms
            eval_env = RewardWrapper(SkillWrapper(gym.make(
                self.env_name), self.params['n_skills'], ev=True), self.d, self.params['n_skills'], parametrization=self.parametrization)
            eval_env = Monitor(eval_env, f"{self.directory}/eval_results")
            eval_callback = EvalCallback(eval_env, best_model_save_path=self.directory,
                                         log_path=f"{self.directory}/eval_results", eval_freq=self.conf.eval_freq,
                                         deterministic=True, render=False, n_eval_episodes=self.conf.eval_runs)
            # create the callback list
            if self.checkpoints:
                fineune_callback = FineTuneCallback(self.args, self.params, self.alg_params, self.conf, self.seed, self.alg, self.timestamp)
                callbacks = [discriminator_callback, eval_callback, fineune_callback]
            else:
                callbacks = [discriminator_callback, eval_callback]
            # train the agent
            model.learn(total_timesteps=self.params['pretrain_steps'], callback=callbacks, log_interval=3, tb_log_name="PPO Pretrain")
        elif self.alg == "sac":
            env = DummyVecEnv([lambda: Monitor(RewardWrapper(SkillWrapper(gym.make(self.env_name), self.params['n_skills'], max_steps=self.conf.max_steps),
                                                             self.d, self.params['n_skills'], parametrization=self.parametrization), self.directory)])
            # create the model with the speicifed hyperparameters
            model = SAC('MlpPolicy', env, verbose=1,
                        learning_rate=self.alg_params['learning_rate'],
                        batch_size=self.alg_params['batch_size'],
                        gamma=self.alg_params['gamma'],
                        buffer_size=self.alg_params['buffer_size'],
                        tau=self.alg_params['tau'],
                        ent_coef=self.alg_params['ent_coef'],
                        gradient_steps=self.alg_params['gradient_steps'],
                        learning_starts=self.alg_params['learning_starts'],
                        policy_kwargs=dict(net_arch=dict(pi=[2*self.conf.layer_size_policy, 2*self.conf.layer_size_policy], qf=[
                                           self.conf.layer_size_q, self.conf.layer_size_q])),
                        tensorboard_log=self.directory,
                        seed=self.seed
                        )

            discriminator_callback = DiscriminatorCallback(self.d, None, self.discriminator_hyperparams, sw=self.sw,
                                                           n_skills=self.params['n_skills'], min_buffer_size=self.params['min_train_size'], save_dir=self.directory, on_policy=False)
            if self.parametrization in ["MLP", "Linear"]:
                eval_env = RewardWrapper(SkillWrapper(gym.make(
                    self.env_name), self.params['n_skills'], ev=True), self.d, self.params['n_skills'], parametrization=self.parametrization)
                eval_env = Monitor(eval_env,  f"{self.directory}/eval_results")

                eval_callback = EvalCallback(eval_env, best_model_save_path=self.directory,
                                             log_path=f"{self.directory}/eval_results", eval_freq=self.conf.eval_freq, deterministic=True, render=False)
            elif self.parametrization in ["Separable", "Concat"]:
                MI_estimator = namedtuple('MI_estimator', "estimator_func estimator_type log_baseline alpha_logit")
                mi_estimate = MI_estimator(mi_lower_bound, self.discriminator_hyperparams['lower_bound'], self.discriminator_hyperparams['log_baseline'], self.discriminator_hyperparams['alpha_logit'])
                # (env_name, discriminator, params, tb_sw, discriminator_hyperparams, mi_estimator, model_save_path, eval_freq=5000, verbose=0)
                eval_callback = MI_EvalCallback(self.env_name, self.d, self.params, self.sw, self.discriminator_hyperparams, mi_estimate, self.directory, eval_freq=self.conf.eval_freq*self.alg_params['n_actors'])
            
            # create the callback list
            if self.checkpoints:
                fineune_callback = FineTuneCallback(self.args, self.params, self.alg_params, self.conf, self.seed, self.alg, self.timestamp)
                callbacks = [discriminator_callback, eval_callback, fineune_callback]
            else:
                callbacks = [discriminator_callback, eval_callback]

            # train the agent
            if self.parametrization in ["Separable", "Concat"]:
                model.learn(total_timesteps=self.params['pretrain_steps'], callback=callbacks, log_interval=3, tb_log_name="SAC Pretrain", d=self.d, mi_estimator=mi_estimate)
            elif self.parametrization in ["MLP", "Linear"]:
                model.learn(total_timesteps=self.params['pretrain_steps'], callback=callbacks, log_interval=3, tb_log_name="SAC Pretrain")
            else:
                raise ValueError(f"{discriminator_hyperparams['parametrization']} is invalid parametrization")
        return model, self.d

    # finetune the pretrained policy on a specific task
    def finetune(self):
        # For the generalization experiment
        if self.task:
            # TODO: add other environments
            if self.env_name == "HalfCheetah-v2":
                env = HalfCheetahTaskWrapper(gym.make("HalfCheetah-v2"), task=self.task)
                env = DummyVecEnv([lambda: SkillWrapper(env, self.params['n_skills'], max_steps=self.conf.max_steps)])
        else:
            env = DummyVecEnv([lambda: SkillWrapper(gym.make(
                self.env_name), self.params['n_skills'], max_steps=self.conf.max_steps)])
            
        model_dir = self.directory + "/best_model"

        # Extract the model and extract the best skill
        if self.alg == "sac":
            model = SAC.load(model_dir, env=env, seed=self.seed)
        elif self.alg == "ppo":
            model = PPO.load(model_dir, env=env,  clip_range=get_schedule_fn(
                self.alg_params['clip_range']), seed=self.seed)

        best_skill_index = best_skill(
            model, self.env_name,  self.params['n_skills'])

        del model
        # define the environment for the adaptation model
        if self.task:
            if self.env_name == "HalfCheetah-v2":
                env = HalfCheetahTaskWrapper(gym.make("HalfCheetah-v2"), task=self.task)
                env = DummyVecEnv([lambda: SkillWrapperFinetune(Monitor(env,  f"{self.directory}/finetune_train_results"), self.params['n_skills'], max_steps=gym.make(self.env_name)._max_episode_steps, skill=best_skill_index)])
        else:
            env = DummyVecEnv([lambda: SkillWrapperFinetune(Monitor(gym.make(
    self.env_name),  f"{self.directory}/finetune_train_results"), self.params['n_skills'], r_seed=None,max_steps=gym.make(self.env_name)._max_episode_steps, skill=best_skill_index)])
        
        adaptation_model = SAC('MlpPolicy', env, verbose=1,
                    learning_rate=self.adapt_params['learning_rate'],
                    batch_size=self.adapt_params['batch_size'],
                    gamma=self.adapt_params['gamma'],
                    buffer_size=self.adapt_params['buffer_size'],
                    tau=self.adapt_params['tau'],
                    ent_coef=self.adapt_params['ent_coef'],
                    gradient_steps=self.adapt_params['gradient_steps'],
                    learning_starts=self.adapt_params['learning_starts'],
                    policy_kwargs=dict(net_arch=dict(pi=[2*self.conf.layer_size_policy, 2*self.conf.layer_size_policy], qf=[
                                       self.conf.layer_size_q, self.conf.layer_size_q])),
                    tensorboard_log=self.directory,
                    seed=self.seed
                    )

        eval_env = SkillWrapperFinetune(gym.make(
            self.env_name), self.params['n_skills'], max_steps=gym.make(self.env_name)._max_episode_steps, r_seed=None, skill=best_skill_index)
        eval_env = Monitor(eval_env, f"{self.directory}/finetune_eval_results")
        
        eval_callback = EvalCallback(eval_env, best_model_save_path=self.directory + f"/best_finetuned_model_skillIndex:{best_skill_index}",
                                    log_path=f"{self.directory}/finetune_eval_results", eval_freq=self.conf.eval_freq,
                                    deterministic=True, render=False)
        # if the model for pretraining is SAC just continue
        if self.alg == "sac":
            sac_model = SAC.load(model_dir, env=env, tensorboard_log=self.directory)
            adaptation_model = sac_model
            # load the discriminator
            d = copy.deepcopy(self.d)
            d.layers[0] = nn.Linear(gym.make(self.env_name).observation_space.shape[0]+ self.params['n_skills']  +1, self.conf.layer_size_discriminator)
            d.head = nn.Sequential(*d.layers)
            d.output = nn.Linear(self.conf.layer_size_discriminator, 1)
            adaptation_model.critic.qf0 = d
            adaptation_model.critic.qf1 = copy.deepcopy(d)
            adaptation_model.critic_target.qf0 = copy.deepcopy(d)
            adaptation_model.critic_target.qf1 = copy.deepcopy(d)
            adaptation_model.learn(total_timesteps=self.params['finetune_steps'],
                        callback=eval_callback, tb_log_name="SAC_FineTune", d=None, mi_estimator=None)

        # if the model for the prratrining is PPO load the discrimunator and actor from ppo into sac models
        elif self.alg == "ppo":
            ppo_model = PPO.load(model_dir, env=env, tensorboard_log=self.directory,
                            clip_range=get_schedule_fn(self.alg_params['clip_range']))
            # load the policy from ppo into sac
            ppo_actor = ppo_model.policy.mlp_extractor.policy_net
            
            adaptation_model.actor.latent_pi = ppo_actor
            adaptation_model.actor.mu = nn.Linear(in_features=self.conf.layer_size_policy, out_features=gym.make(self.env_name).action_space.shape[0], bias=True)
            adaptation_model.actor.log_std = nn.Linear(in_features=self.conf.layer_size_policy, out_features=gym.make(self.env_name).action_space.shape[0], bias=True)
            # load the discriminator
            d = copy.deepcopy(self.d)
            d.layers[0] = nn.Linear(gym.make(self.env_name).observation_space.shape[0]+ self.params['n_skills']  +1, self.conf.layer_size_discriminator)
            d.head = nn.Sequential(*d.layers)
            d.output = nn.Linear(self.conf.layer_size_discriminator, 1)
            adaptation_model.critic.qf0 = d
            adaptation_model.critic.qf1 = copy.deepcopy(d)
            adaptation_model.critic_target.qf0 = copy.deepcopy(d)
            adaptation_model.critic_target.qf1 = copy.deepcopy(d)
            adaptation_model.learn(total_timesteps=self.params['finetune_steps'],
                        callback=eval_callback, tb_log_name="PPO_FineTune", d=None, mi_estimator=None)
        return adaptation_model, best_skill_index