import gym

from stable_baselines3.common.vec_env.dummy_vec_env import DummyVecEnv
from stable_baselines3 import PPO, SAC
from stable_baselines3.common.callbacks import EvalCallback
from stable_baselines3.common.utils import get_schedule_fn
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.distributions import DiagGaussianDistribution

from src.environment_wrappers.env_wrappers import RewardWrapper, SkillWrapper, SkillWrapperFinetune
from src.environment_wrappers.tasks_wrappers import HalfCheetahTaskWrapper, WalkerTaskWrapper, AntTaskWrapper
from src.utils import best_skill
from src.mi_lower_bounds import mi_lower_bound
from src.models.models import Discriminator, SeparableCritic, ConcatCritic
from src.replayBuffers import DataBuffer
from src.callbacks.callbacks import DiscriminatorCallback, VideoRecorderCallback, EvaluationCallback, MI_EvalCallback
from src.config import conf

import torch
import torch.nn as nn
import torch.optim as opt
from torch.utils.tensorboard import SummaryWriter
import copy

import time

from collections import namedtuple

import matplotlib.pyplot as plt
import seaborn as sns
sns.set_theme(style="darkgrid")
sns.set(font_scale = conf.font_scale)


class DIAYN():
    '''
    An implementation of DIAYN with a model-free algorithm (PPO, SAC) as the optimization backbone
    '''
    def __init__(self, params, alg_params, discriminator_hyperparams, env="MountainCarContinuous-v0", alg="ppo", directory="./", seed=10, conf=None, timestamp=None, checkpoints=False, args=None, task=None, adapt_params=None, n_samples=None):
        # create the discirminator
        state_dim = gym.make(env).observation_space.shape[0]
        skill_dim = params['n_skills']
        hidden_dims = conf.num_layers_discriminator*[conf.layer_size_discriminator]
        latent_dim = conf.latent_size
        temperature = discriminator_hyperparams['temperature']
        dropout = discriminator_hyperparams['dropout']
        if discriminator_hyperparams['parametrization'] == "MLP":
            self.d = Discriminator(state_dim, hidden_dims, skill_dim, dropout=discriminator_hyperparams['dropout']).to(conf.device)
            self.d_adapt = Discriminator(state_dim, hidden_dims, skill_dim, dropout=discriminator_hyperparams['dropout']).to(conf.device)
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
        self.n_samples = n_samples

    def pretrain(self):
        '''
        Pretraining phase with an intrinsic reward
        '''
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
                        policy_kwargs=dict(activation_fn=nn.ReLU,
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
                                         log_path=f"{self.directory}/eval_results", eval_freq=5000,
                                         deterministic=True, render=False, n_eval_episodes=self.conf.eval_runs)
            # create the callback list
            if self.checkpoints:
                # env_name, alg, discriminator, params, pm, n_samples=100)
                fineune_callback = EvaluationCallback(self.env_name, self.alg, self.d, self.params, self.parametrization, n_samples=self.n_samples)
                callbacks = [discriminator_callback, eval_callback, fineune_callback]
            else:
                callbacks = [discriminator_callback, eval_callback]
            # train the agent
            model.learn(total_timesteps=self.params['pretrain_steps'], callback=callbacks, log_interval=3, tb_log_name="PPO Pretrain")
            # for testing
            # model.learn(total_timesteps=4500, callback=callbacks, log_interval=3, tb_log_name="PPO Pretrain")
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
                        policy_kwargs=dict(net_arch=dict(pi=[self.conf.layer_size_policy, self.conf.layer_size_policy], qf=[
                                           self.conf.layer_size_q, self.conf.layer_size_q]), n_critics=2),
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
                # env_name, alg, discriminator, params, pm, n_samples=100)
                fineune_callback = EvaluationCallback(self.env_name, self.alg, self.d, self.params, self.parametrization, n_samples=self.n_samples)
                callbacks = [discriminator_callback, eval_callback, fineune_callback]
            else:
                callbacks = [discriminator_callback, eval_callback]
                

            # train the agent
            if self.parametrization in ["Separable", "Concat"]:
                model.learn(total_timesteps=self.params['pretrain_steps'], callback=callbacks, log_interval=3, tb_log_name="SAC Pretrain", d=self.d, mi_estimator=mi_estimate)
            elif self.parametrization in ["MLP", "Linear"]:
                model.learn(total_timesteps=self.params['pretrain_steps'], callback=callbacks, log_interval=3, tb_log_name="SAC Pretrain")
                 # for testing
                # model.learn(total_timesteps=2000, callback=callbacks, log_interval=3, tb_log_name="SAC Pretrain")
            else:
                raise ValueError(f"{discriminator_hyperparams['parametrization']} is invalid parametrization")
        if self.checkpoints:
            results = {
                'steps': fineune_callback.steps,
                'intr_rewards': fineune_callback.intr_rewards,
                'extr_rewards': fineune_callback.extr_rewards,
            }
            return model, self.d, results
        return model, self.d

    # finetune the pretrained policy on a specific task
    def finetune(self):
        '''
        Adapt the pretrained model with task reward using SAC
        '''
        # For the generalization experiment
        if self.task:
            # TODO: add other environments
            if self.env_name == "HalfCheetah-v2":
                env = HalfCheetahTaskWrapper(gym.make("HalfCheetah-v2"), task=self.task)
                env = DummyVecEnv([lambda: SkillWrapper(env, self.params['n_skills'], max_steps=self.conf.max_steps)])
            # WalkerTaskWrapper, AntTaskWrapper
            elif self.env_name == "Walker2d-v2":
                env = WalkerTaskWrapper(gym.make("Walker2d-v2"), task=self.task)
                env = DummyVecEnv([lambda: SkillWrapper(env, self.params['n_skills'], max_steps=self.conf.max_steps)])
            elif self.env_name == "Ant-v2":
                env = AntTaskWrapper(gym.make("Ant-v2"), task=self.task)
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
        # best_skill_index = 0

        del model
        # define the environment for the adaptation model
        env, eval_env = self.adaptation_environment(best_skill_index)
        
        self.adaptation_model = SAC('MlpPolicy', env, verbose=1,
                    learning_rate=self.adapt_params['learning_rate'],
                    batch_size=self.adapt_params['batch_size'],
                    gamma=self.adapt_params['gamma'],
                    buffer_size=self.adapt_params['buffer_size'],
                    tau=self.adapt_params['tau'],
                    ent_coef=self.adapt_params['ent_coef'],
                    gradient_steps=self.adapt_params['gradient_steps'],
                    learning_starts=self.adapt_params['learning_starts'],
                    policy_kwargs=dict(net_arch=dict(pi=[self.conf.layer_size_policy, self.conf.layer_size_policy], qf=[
                                       self.conf.layer_size_q, self.conf.layer_size_q]), n_critics=2),
                    tensorboard_log=self.directory,
                    seed=self.seed
                    )

        eval_env = Monitor(eval_env, f"{self.directory}/finetune_eval_results")
        
        
        eval_callback = EvalCallback(eval_env, best_model_save_path=self.directory + f"/best_finetuned_model_skillIndex:{best_skill_index}",
                                    log_path=f"{self.directory}/finetune_eval_results", eval_freq=self.conf.eval_freq,
                                    deterministic=True, render=False)
        
        # if the model for pretraining is SAC just load the discriminator
        if self.alg == "sac":
            sac_model = SAC.load(model_dir, env=env, tensorboard_log=self.directory)
            # self.adaptation_model = sac_model
            self.adaptation_model.actor.latent_pi.load_state_dict(sac_model.actor.latent_pi.state_dict())
            self.adaptation_model.actor.mu.load_state_dict(sac_model.actor.mu.state_dict())
            self.adaptation_model.actor.log_std.load_state_dict(sac_model.actor.log_std.state_dict())
            self.adaptation_model.actor.optimizer = opt.Adam(self.adaptation_model.actor.parameters(), lr=self.adapt_params['learning_rate'])
            # # load the discriminator
            # self.d.layers[0] = nn.Linear(gym.make(self.env_name).observation_space.shape[0] + self.params['n_skills']  + gym.make(self.env_name).action_space.shape[0], self.conf.layer_size_discriminator)
            # self.d.layers[-1] = nn.Linear(self.conf.layer_size_discriminator, 1)
            # seq = nn.Sequential(*self.d.layers)
            # # # print(d)
            # self.d.eval()
            self.adaptation_model.critic.qf0.load_state_dict(sac_model.critic.qf0.state_dict())
            self.adaptation_model.critic.qf1.load_state_dict(sac_model.critic.qf1.state_dict())
            self.adaptation_model.critic_target.load_state_dict(self.adaptation_model.critic.state_dict())
            self.adaptation_model.critic.optimizer = opt.Adam(self.adaptation_model.critic.parameters(), lr=self.adapt_params['learning_rate'])
            self.adaptation_model.learn(total_timesteps=self.params['finetune_steps'],
                        callback=eval_callback, tb_log_name="SAC_FineTune", d=None, mi_estimator=None)

        # if the model for the prratrining is PPO load the discrimunator and actor from ppo into sac models
        elif self.alg == "ppo":
            self.ppo_model = PPO.load(model_dir, env=env, tensorboard_log=self.directory,
                            clip_range=get_schedule_fn(self.alg_params['clip_range']))
            # adaptation_model.actor.action_dist = DiagGaussianDistribution(env.action_space.shape[0])
            # load the policy ppo 
            ppo_actor = self.ppo_model.policy
            self.load_state_dicts(ppo_actor)
            self.adaptation_model.learn(total_timesteps=self.params['finetune_steps'],
                        callback=eval_callback, tb_log_name="PPO_FineTune", d=None, mi_estimator=None)
        return self.adaptation_model, best_skill_index     
    
    def load_state_dicts(self, ppo_actor):
        '''
        load the pretrained model parameters into the adaptation model
        '''
        # initlialize the adaptation policy with the pretrained model
        self.adaptation_model.actor.latent_pi.load_state_dict(ppo_actor.mlp_extractor.policy_net.state_dict())
        self.adaptation_model.actor.mu.load_state_dict(ppo_actor.action_net.state_dict())
        self.adaptation_model.actor.log_std.load_state_dict(nn.Linear(in_features=self.conf.layer_size_policy, out_features=gym.make(self.env_name).action_space.shape[0], bias=True).state_dict())
        self.adaptation_model.actor.optimizer = opt.Adam(self.adaptation_model.actor.parameters(), lr=self.adapt_params['learning_rate'])
        
        layers = [nn.Linear(gym.make(self.env_name).observation_space.shape[0] + self.params['n_skills']  + gym.make(self.env_name).action_space.shape[0], self.conf.layer_size_discriminator)]
        for l in range(len(self.ppo_model.policy.mlp_extractor.value_net)):
            if l != 0:
                layers.append(self.ppo_model.policy.mlp_extractor.value_net[l])
        layers.append(nn.Linear(self.conf.layer_size_discriminator, 1))
        seq = nn.Sequential(*layers)
        print(seq)
        # input()
        self.adaptation_model.critic.qf0.load_state_dict(seq.state_dict())
        self.adaptation_model.critic.qf1.load_state_dict(seq.state_dict())
        self.adaptation_model.critic_target.load_state_dict(self.adaptation_model.critic.state_dict())
        self.adaptation_model.critic.optimizer = opt.Adam(self.adaptation_model.critic.parameters(), lr=self.adapt_params['learning_rate'])
        
    def adaptation_environment(self, best_skill_index):
        '''
        Adaptation environment according to the task reward
        '''
        if self.task:
            # print(f"There is a task which is {self.task}")
            # input()
            if self.env_name == "HalfCheetah-v2":
                # print(f"environment: {self.env_name}")
                # input()
                env = HalfCheetahTaskWrapper(gym.make("HalfCheetah-v2"), task=self.task)
                env = DummyVecEnv([lambda: SkillWrapperFinetune(Monitor(env,  f"{self.directory}/finetune_train_results"), self.params['n_skills'], max_steps=gym.make(self.env_name)._max_episode_steps, skill=best_skill_index)])
                eval_env = HalfCheetahTaskWrapper(gym.make(self.env_name), task=self.task)
                eval_env = SkillWrapperFinetune(eval_env, self.params['n_skills'], max_steps=gym.make(self.env_name)._max_episode_steps, r_seed=None, skill=best_skill_index)
                
            # # WalkerTaskWrapper, AntTaskWrapper
            elif self.env_name == "Walker2d-v2":
                # print(f"environment: {self.env_name}")
                # input()
                env = WalkerTaskWrapper(gym.make("Walker2d-v2"), task=self.task)
                env = DummyVecEnv([lambda: SkillWrapperFinetune(Monitor(env,  f"{self.directory}/finetune_train_results"), self.params['n_skills'], max_steps=gym.make(self.env_name)._max_episode_steps, skill=best_skill_index)])
                eval_env = WalkerTaskWrapper(gym.make(self.env_name), task=self.task)
                eval_env = SkillWrapperFinetune(eval_env, self.params['n_skills'], max_steps=gym.make(self.env_name)._max_episode_steps, r_seed=None, skill=best_skill_index)
                
            elif self.env_name == "Ant-v2":
                # print(f"environment: {self.env_name}")
                # input()
                env = AntTaskWrapper(gym.make("Ant-v2"), task=self.task)
                env = DummyVecEnv([lambda: SkillWrapperFinetune(Monitor(env,  f"{self.directory}/finetune_train_results"), self.params['n_skills'], max_steps=gym.make(self.env_name)._max_episode_steps, skill=best_skill_index)])
                eval_env = AntTaskWrapper(gym.make(self.env_name), task=self.task)
                eval_env = SkillWrapperFinetune(eval_env, self.params['n_skills'], max_steps=gym.make(self.env_name)._max_episode_steps, r_seed=None, skill=best_skill_index)
                
        else:
            # print(f"Just ")
            env = DummyVecEnv([lambda: SkillWrapperFinetune(Monitor(gym.make(
    self.env_name),  f"{self.directory}/finetune_train_results"), self.params['n_skills'], r_seed=None,max_steps=gym.make(self.env_name)._max_episode_steps, skill=best_skill_index)])
            eval_env = SkillWrapperFinetune(gym.make(
            self.env_name), self.params['n_skills'], max_steps=gym.make(self.env_name)._max_episode_steps, r_seed=None, skill=best_skill_index)
            
        return env, eval_env
        