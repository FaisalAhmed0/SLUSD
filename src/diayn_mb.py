'''
This file contains an implementation of DIAYN with PETS (a model-based RL algorithm)
'''
import matplotlib as mpl
import matplotlib.pyplot as plt
import omegaconf

# MBRL library components 
import mbrl
import mbrl.env.cartpole_continuous as cartpole_env
import mbrl.env.reward_fns as reward_fns
import mbrl.env.termination_fns as termination_fns
import mbrl.models as models
import mbrl.planning as planning
import mbrl.util.common as common_util
import mbrl.util as util

import os

import torch
import torch.multiprocessing as mp
torch.set_num_threads(1)
torch.multiprocessing.set_sharing_strategy('file_system')
import copy

import gym

from stable_baselines3.common.vec_env.dummy_vec_env import DummyVecEnv
from stable_baselines3 import SAC
from stable_baselines3.common.callbacks import EvalCallback
from stable_baselines3.common.monitor import Monitor

from src.environment_wrappers.env_wrappers import RewardWrapper, SkillWrapper, SkillWrapperFinetune
from src.environment_wrappers.tasks_wrappers import HalfCheetahTaskWrapper
from src.utils import record_video_finetune, best_skill, evaluate_pretrained_policy_intr, evaluate_pretrained_policy_ext
from src.mi_lower_bounds import mi_lower_bound
from src.models.models import Discriminator, SeparableCritic, ConcatCritic
from src.replayBuffers import DataBuffer
from src.callbacks.callbacks import DiscriminatorCallback, VideoRecorderCallback, MI_EvalCallback
from src.config import conf

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as opt
import numpy as np


from torch.utils.tensorboard import SummaryWriter

import time

from collections import namedtuple


class DIAYN_MB():
    def __init__(self, params, alg_params, discriminator_hyperparams, env="MountainCarContinuous-v0", alg="pets", directory="./", seed=10, device="cpu", conf=None, timestamp=None, checkpoints=False, args=None, task=None, adapt_params=None, n_samples=None):
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
            log_dir=f"{directory}/PETS_PreTrain", comment=f"{alg} Discriminator, env_name:{env}, PreTrain")

        self.sw_adapt = SummaryWriter(
            log_dir=f"{directory}/PETS_FineTune", comment=f"{alg} Discriminator, env_name:{env}, FineTune")

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
        # Extract the discriminator hyperparameters
        self.weight_decay = discriminator_hyperparams['weight_decay']
        # number of epochs
        self.epochs = discriminator_hyperparams['n_epochs']
        # batch size
        self.batch_size = discriminator_hyperparams['batch_size']
        # number of skills
        self.n_skills = skill_dim
        # minimum size of the buffer before training
        self.min_buffer_size = params['min_train_size']
        # gradient penalty
        self.gp = discriminator_hyperparams['gp']
        # mixup regularization
        self.mixup = discriminator_hyperparams['mixup']
        # parametrization 
        self.paramerization = discriminator_hyperparams['parametrization']
        # temperature
        self.temperature = discriminator_hyperparams['temperature']
        # optimizer
        self.optimizer = opt.Adam(self.d.parameters(), lr=discriminator_hyperparams['learning_rate'], weight_decay=self.weight_decay)
        # lower bound type
        self.lower_bound = discriminator_hyperparams['lower_bound']
        # label smoothing
        self.label_smoothing = discriminator_hyperparams['label_smoothing']
        # log baseline
        self.log_baseline = discriminator_hyperparams['log_baseline']
        # alpha logit
        self.alpha_logit = discriminator_hyperparams['alpha_logit']
        # gradient clip
        self.gradient_clip = discriminator_hyperparams['gradient_clip']
        # hyperparameters for the adaptation algorithm
        self.adapt_params = adapt_params
        # checkpoints for the scalability experiment
        self.checkpoints = checkpoints
        # number of checkpoints
        self.n_samples = n_samples
        
    def pretrain(self):
        self.timesteps = []
        self.results = []
        # create MBRL lib gym-like env wrapper
        env_name = self.env_name
        seed = self.seed
        env = RewardWrapper(SkillWrapper(gym.make(env_name), self.params['n_skills']), self.d, self.n_skills, self.paramerization)
        env.env.seed(seed)
        rng = np.random.default_rng(seed=seed)
        generator = torch.Generator(device=self.conf.device)
        generator.manual_seed(seed)
        obs_shape = env.observation_space.shape
        act_shape = env.action_space.shape
        reward_fn = Reward_func(self.d, self.n_skills, self.paramerization)
        term_fn = termination_fns.no_termination
        ## Configrations
        # TODO: replace this with the hyperparams arguments
        trial_length = self.alg_params['trial_length'] 
        self.trial_length = trial_length
        num_trials = int(self.params['pretrain_steps'] // trial_length)
        print(f"trial_length:{trial_length}")
        print(f"pretraining steps: {self.params['pretrain_steps']}")
        print(f"num_trials: {num_trials}")
        # input()
        ensemble_size = self.alg_params['ensemble_size'] 
        # Everything with "???" indicates an option with a missing value.
        # Our utility functions will fill in these details using the 
        # environment information
        # TODO: replace this with the hyperparams arguments
        cfg_dict = {
            # dynamics model configuration
            "dynamics_model": {
                "_target_": "mbrl.models.GaussianMLP",
                "device": conf.device,
                "num_layers": 2,
                "ensemble_size": ensemble_size,
                "hid_size": conf.layer_size_policy,
                "in_size": "???",
                "out_size": "???",
                "deterministic": False,
                "propagation_method": "fixed_model",
                # can also configure activation function for GaussianMLP
                "activation_fn_cfg": {
                    "_target_": "torch.nn.LeakyReLU",
                    "negative_slope": 0.01
                }
            },
            # options for training the dynamics model
            "algorithm": {
                "learned_rewards": False,
                "target_is_delta": True,
                "normalize": False,
            },
            # these are experiment specific options
            "overrides": {
                "trial_length": trial_length,
                "num_steps": self.params['pretrain_steps'],
                "model_batch_size": self.alg_params['batch_size'],
                "validation_ratio": 0.05
            }
        }
        # TODO: replace this with the hyperparams arguments
        cfg = omegaconf.OmegaConf.create(cfg_dict)
        # Create a 1-D dynamics model for this environment
        dynamics_model = common_util.create_one_dim_tr_model(cfg, obs_shape, act_shape)

        # Create a gym-like environment to encapsulate the model
        model_env = models.ModelEnv(env, dynamics_model, term_fn, reward_fn, generator=generator)
        # Create the replay buffer
        replay_buffer = common_util.create_replay_buffer(cfg, obs_shape, act_shape, rng=rng)
        # collect a random trajectory
        common_util.rollout_agent_trajectories(
            env,
            5 * trial_length, # initial exploration steps
            planning.RandomAgent(env),
            {}, # keyword arguments to pass to agent.act()
            replay_buffer=replay_buffer,
            trial_length=trial_length
        )
        timesteps = 5 * trial_length
        print("# samples stored", replay_buffer.num_stored)
        # CEM (Cross-Entropy Method) agent
        # TODO: replace this with the hyperparams arguments
        agent_cfg = omegaconf.OmegaConf.create({
        # this class evaluates many trajectories and picks the best one
        "_target_": "mbrl.planning.TrajectoryOptimizerAgent",
        "planning_horizon": self.alg_params['planning_horizon'],
        "replan_freq": 1,
        "verbose": False,
        "action_lb": "???",
        "action_ub": "???",
        # this is the optimizer to generate and choose a trajectory
        "optimizer_cfg": {
            "_target_": "mbrl.planning.CEMOptimizer",
            "num_iterations": 5,
            "elite_ratio": self.alg_params['elite_ratio'],
            "population_size": self.alg_params['population_size'],
            "alpha": 0.1,
            "device": conf.device,
            "lower_bound": "???",
            "upper_bound": "???",
            "return_mean_elites": True,
            "clipped_normal": False
        }
        })
        # TODO: replace this with the hyperparams arguments
        agent = planning.create_trajectory_optim_agent_for_model(
            model_env,
            agent_cfg,
            num_particles=self.alg_params['num_particles'],
        )
        # TODO: Add tensorboard
        # Main Training loop
        # Create a trainer for the model
        # TODO: replace this with the hyperparams arguments
        model_trainer = models.ModelTrainer(dynamics_model, optim_lr=self.alg_params['learning_rate'], weight_decay=self.alg_params['weight_decay'])
        # number of trials is the number of iterations 
        best_eval_reward = -np.inf
        print(f"Trails: {num_trials}")
        timesteps_list = []
        # for scalability experiment
        if self.checkpoints:
            steps_l = []
            intr_rewards = []
            extr_rewards = []
            freq = self.params['pretrain_steps'] // self.n_samples
            print(f"checkpoints freq: {freq}")
        for trial in range(num_trials):
          # initilization
            obs = env.reset()    
            agent.reset()
            done = False
            total_reward = 0.0
            steps_trial = 0
            while not done:
                print(f"Trial: {trial}, step: {steps_trial}")
                # --------------- Model Training -----------------
                if steps_trial == 0:
                    dynamics_model.update_normalizer(replay_buffer.get_all())  # update normalizer stats

                    dataset_train, dataset_val = common_util.get_basic_buffer_iterators(
                        replay_buffer,
                        batch_size=cfg.overrides.model_batch_size,
                        val_ratio=cfg.overrides.validation_ratio,
                        ensemble_size=ensemble_size,
                        shuffle_each_epoch=True,
                        bootstrap_permutes=False,  # build bootstrap dataset using sampling with replacement
                    )
                    # train the model dynamics
                    model_trainer.train(
                        dataset_train, 
                        dataset_val=dataset_val, 
                        num_epochs=10, 
                        patience=50, 
                        # callback=train_callback,
                        silent=True)
                    
                    # add data to the discriminator buffer
                    env_obs, skill = self.split_obs(obs)
                    obs_t = torch.tensor(env_obs)
                    skill = torch.tensor(skill, device=conf.device)
                    self.buffer.add(obs_t, skill)
                    
                # update the discriminator
                self.update_discriminator(timesteps)

                # --- Doing env step using the agent and adding to model dataset ---
                next_obs, reward, done, _ = common_util.step_env_and_add_to_buffer(
                    env, obs, agent, {}, replay_buffer)
                env_obs, skill = self.split_obs(next_obs)
                obs_t = torch.tensor(env_obs)
                skill = torch.tensor(skill, device=conf.device)
                self.buffer.add(obs_t, skill)
                
                # if timesteps % (conf.eval_freq * 5) == 0:
                if timesteps % (conf.eval_freq) == 0:
                    print("In Evaluation")
                    rewards = self.evaluate_policy(self.env_name, model_env)
                    if rewards > best_eval_reward:
                        best_eval_reward = rewards
                        print(f"New best reward: {best_eval_reward}")
                        # save the best model
                        model_env.dynamics_model.save(f"{self.directory}")
                    self.timesteps.append(timesteps)
                    self.results.append(rewards)
                    timesteps_np = np.array(self.timesteps)
                    results_np = np.array(self.results)
                    os.makedirs(f"{self.directory}/eval_results", exist_ok=True)
                    np.savez(f"{self.directory}/eval_results/evaluations.npz", timesteps=timesteps_np, results=results_np)
                    # add to tensorboard
                    self.sw.add_scalar("eval/mean_reward", rewards, timesteps)
                    print(f"TimeStep: {timesteps}, Evaluation reward: {rewards}")
                    
                # if checkpoints is true add to the logs
                if self.checkpoints and (timesteps%freq == 0):
                    print("Checkpoint")
                    pretrained_policy = agent
                    intr_reward = np.mean( [evaluate_pretrained_policy_intr(self.env_name, self.n_skills, pretrained_policy, self.d, self.paramerization, "pets") for _ in range(3) ] )
                    extr_reward = np.mean( [evaluate_pretrained_policy_ext(self.env_name, self.n_skills, pretrained_policy, "pets") for _ in range(3)] )
                    steps_l.append(timesteps)
                    intr_rewards.append(intr_reward)
                    extr_rewards.append(extr_reward)
                    plt.figure()
                    plt.plot(steps_l, extr_rewards, label="pets".upper())
                    plt.xlabel("Pretraining Steps")
                    plt.ylabel("Extrinsic Reward")
                    plt.legend()
                    plt.tight_layout()
                    filename = f"Scalability_Experiment_realtime_env:{self.env_name}_alg:pets_xaxis:Pretraining Steps.png"
                    plt.savefig(f'{filename}', dpi=150) 
                    plt.figure()
                    plt.plot(intr_rewards, extr_rewards, label="pets".upper())
                    plt.xlabel("Intrinsic Reward")
                    plt.ylabel("Extrinsic Reward")
                    plt.legend()
                    plt.tight_layout()
                    filename = f"Scalability_Experiment_realtime_env:{self.env_name}_alg:pets_xaxis:Intrinsic Reward.png"
                    plt.savefig(f'{filename}', dpi=150) 
                    
                timesteps += 1
                obs = next_obs
                total_reward += reward
                steps_trial += 1
                print(int(steps_trial) == int(trial_length))
                if int(steps_trial) == int(trial_length):
                    self.sw.add_scalar("rollout/ep_rew_mean", total_reward, timesteps)
                    print(f"TimeStep: {timesteps}, Rollout reward: {total_reward}")
                    break
            
            print("broke")        
        self.model_env = model_env
        self.agent = agent
        if self.checkpoints:
            results = {
                'steps': steps_l,
                'intr_rewards': intr_rewards,
                'extr_rewards': extr_rewards
            }
            return agent, self.d, results
        return agent, self.d
    
    def best_skill(self):
        total_rewards = []
        for skill in range(self.n_skills):
            env = SkillWrapperFinetune(gym.make(self.env_name), self.params['n_skills'], skill)
            env.env.seed(self.seed)
            total_reward = 0
            obs = env.reset()
            for i in range(self.alg_params['trial_length']):
                action_seq = self.agent.plan(obs)
                obs, reward, done, _ = env.step(action_seq[0])
                total_reward += reward
            total_rewards.append(total_reward)
        return total_rewards
            
    def finetune(self):
        self.timesteps = []
        self.results = []
        # create MBRL lib gym-like env wrapper
        env_name = self.env_name
        seed = self.seed
        best_skill = np.argmax( np.mean([self.best_skill() for _ in range(3)], axis=0))

        env = DummyVecEnv([lambda: SkillWrapperFinetune(Monitor(gym.make(
        self.env_name),  f"{self.directory}/finetune_train_results"), self.params['n_skills'], r_seed=None,max_steps=gym.make(self.env_name)._max_episode_steps, skill=best_skill)])

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
            self.env_name), self.params['n_skills'], max_steps=gym.make(self.env_name)._max_episode_steps, r_seed=None, skill=best_skill)
        eval_env = Monitor(eval_env, f"{self.directory}/finetune_eval_results")

        eval_callback = EvalCallback(eval_env, best_model_save_path=self.directory + f"/best_finetuned_model_skillIndex:{best_skill}",
                                    log_path=f"{self.directory}/finetune_eval_results", eval_freq=self.conf.eval_freq,
                                    deterministic=True, render=False)

        env = DummyVecEnv([lambda: SkillWrapperFinetune(Monitor(gym.make(
    self.env_name),  f"{self.directory}/finetune_train_results"), self.params['n_skills'], r_seed=None,max_steps=gym.make(self.env_name)._max_episode_steps, skill=best_skill)])

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
                                       self.conf.layer_size_q, self.conf.layer_size_q]), clip_mean=None),
                    tensorboard_log=self.directory,
                    seed=self.seed
                    )

        eval_env = SkillWrapperFinetune(gym.make(
            self.env_name), self.params['n_skills'], max_steps=gym.make(self.env_name)._max_episode_steps, r_seed=None, skill=best_skill)
        eval_env = Monitor(eval_env, f"{self.directory}/finetune_eval_results")

        eval_callback = EvalCallback(eval_env, best_model_save_path=self.directory + f"/best_finetuned_model_skillIndex:{best_skill}",
                                    log_path=f"{self.directory}/finetune_eval_results", eval_freq=self.conf.eval_freq,
                                    deterministic=True, render=False)
        # load the policy weights from the environment model
        model = self.load_env_model_for_adapt(self.model_env)
        print(model)
        print(adaptation_model.actor.latent_pi)
        adaptation_model.actor.latent_pi = model
        adaptation_model.actor.mu = nn.Linear(conf.layer_size_policy, env.action_space.shape[0])
        adaptation_model.actor.log_std = nn.Linear(conf.layer_size_policy, env.action_space.shape[0])
        adaptation_model.policy.actor.optimizer = opt.Adam(adaptation_model.actor.parameters(), lr=self.adapt_params['learning_starts'])
        # load the discriminator
        d = copy.deepcopy(self.d)
        d.layers[0] = nn.Linear(gym.make(self.env_name).observation_space.shape[0]+self.params['n_skills'] +1, self.conf.layer_size_discriminator)
        d.head = nn.Sequential(*d.layers)
        d.output = nn.Linear(self.conf.layer_size_discriminator, 1)
        adaptation_model.critic.qf0 = d
        adaptation_model.critic.qf1 = copy.deepcopy(d)
        adaptation_model.critic_target.qf0 = copy.deepcopy(d)
        adaptation_model.critic_target.qf1 = copy.deepcopy(d)
        adaptation_model.critic.optimizer = opt.Adam(adaptation_model.critic.parameters(), lr=self.adapt_params['learning_starts'])
        adaptation_model.learn(total_timesteps=self.params['finetune_steps'],
                    callback=eval_callback, tb_log_name="PETS_FineTune", d=None, mi_estimator=None)

        return adaptation_model, best_skill



    def load_env_model_for_adapt(self, model_env):
        # pick a model randomly from the ensamble 
        random_model = np.random.randint(low=0, high=self.alg_params['ensemble_size'])
        layers = []
        act = nn.ReLU()
        for i in range(2):
            state_dict = {}
            model_layer_weight = model_env.dynamics_model.model.state_dict()[f"hidden_layers.{i}.0.weight"][random_model]
            state_dict["weight"] = model_layer_weight.T
            model_layer_bias = model_env.dynamics_model.model.state_dict()[f"hidden_layers.{i}.0.bias"][random_model]
            state_dict["bias"] = model_layer_bias.reshape(-1)
            layer = nn.Linear(model_layer_weight.shape[0], model_layer_weight.shape[1])
            layer.load_state_dict(state_dict)
            layers.append(layer)
            layers.append(act)
        layers[0] = nn.Linear(gym.make(self.env_name).observation_space.shape[0]+self.params["n_skills"], conf.layer_size_policy)
        model = nn.Sequential(*layers)
        return model
        
    
    # common_util.rollout_model_env()
    def evaluate_policy(self, env_name, model_env):
        print("In evaluate_policy")
        # cfg for the evaluation agent
        agent_cfg = omegaconf.OmegaConf.create({
        # this class evaluates many trajectories and picks the best one
        "_target_": "mbrl.planning.TrajectoryOptimizerAgent",
        "planning_horizon": self.alg_params['planning_horizon'],
        "replan_freq": 1,
        "verbose": False,
        "action_lb": "???",
        "action_ub": "???",
        # this is the optimizer to generate and choose a trajectory
        "optimizer_cfg": {
            "_target_": "mbrl.planning.CEMOptimizer",
            "num_iterations": 5,
            "elite_ratio": self.alg_params['elite_ratio'],
            "population_size": self.alg_params['population_size'],
            "alpha": 0.1,
            "device": conf.device,
            "lower_bound": "???",
            "upper_bound": "???",
            "return_mean_elites": True,
            "clipped_normal": False
        }
        })
        # CEM agent
        agent = planning.create_trajectory_optim_agent_for_model(
                model_env,
                agent_cfg,
                num_particles=self.alg_params['num_particles']) 
        
        env = RewardWrapper(SkillWrapper(gym.make(env_name), self.params['n_skills']), self.d, self.n_skills, self.paramerization)
        total_reward = 0
        obs = env.reset()
        for i in range(self.trial_length):
            # print(f"Step: {i}")
            action_seq = agent.plan(obs)
            obs, reward, done, _ = env.step(action_seq[0])
            total_reward += reward
        print("Finished evaluate_policy")
        return total_reward
    

        
        
    def split_obs(self, obs):
        obs = obs
        env_obs = obs[: -self.n_skills]
        skill, = np.where(obs[-self.n_skills:] == 1)
        return env_obs, skill[0]
    
    
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
    
    def update_discriminator(self, timesteps):
        """
        This event is triggered before updating the policy.
        """
        current_buffer_size = len(self.buffer)
        print(f"current_buffer_size: {current_buffer_size}")
        if current_buffer_size >= self.min_buffer_size:
            epoch_loss = 0
            self.d.train()
            for epoch in range(self.epochs):
                # Extract the data
                states, skills = self.buffer.sample(self.batch_size)
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
                    states2, skills2 = self.buffer.sample(self.batch_size)
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
            self.sw.add_scalar("discrimiator loss",
                                        epoch_loss, timesteps)
            self.sw.add_scalar("discrimiator grad norm",
                            grad_norms, timesteps)
            self.sw.add_scalar("discrimiator wights norm",
                            weights_norm, timesteps)
            torch.save(self.d.state_dict(), self.directory + "/disc.pth")
                
                
                
                
class Reward_func():
    def __init__(self, d, skills,parametrization="MLP"):
        self.d = d
        self.n_skills = skills
        self.parametrization = parametrization
        
    @ torch.no_grad()
    def __call__(self, action, next_obs):
        self.d.eval()
        self.batch_size = action.shape[0]
        # print(f"obs shape: {next_obs.shape}")
        # input()
        env_obs, skills = self.split_obs(next_obs)
        # print(f"skill: {skills}")
        # print(f"prediction shape: {}")
        if self.parametrization in ["MLP", "Linear"]:
            reward = (torch.log_softmax(self.d(env_obs.to(next_obs.device)).detach(), dim=-1)[torch.arange(self.batch_size), skills] - torch.log(torch.tensor(self.n_skills)).to(conf.device))
            # print(f"reward shape: {reward.shape}")
            # input()
            return reward.reshape(-1, 1)
        elif self.parametrization in ["Separable", "Concat"]:
            skills_onehot = self.one_hot(skills)
            reward = self.d( env_obs, skills_onehot )
            return reward.reshape(-1, 1)
            
            
            
    def one_hot(self, skills):
        onehot = torch.zeros(self.batch_size, self.n_skills)
        onehot[torch.arange(self.batch_size), skills] = 1
        return onehot
        
    def split_obs(self, obs):
        env_obs = obs[:, :-self.n_skills]
        skills = np.argmax(obs[:, -self.n_skills:].cpu().numpy(), axis=-1)
        return env_obs, skills
    
    
# class AdaptationReward():
#     def __init__(self, env_name):
#         self.env_name = env_name
        
#     def __call__(self, action, next_obs):
#         if env_name == "MountainCarContinuous-v0":
#             return self.MC_reward(action, next_obs)
#         elif env_name == "Reacher-v2":
#             return self.Reacher_reward(action, next_obs)
#         elif env_name == "Swimmer-v2":
#             return self.Swimmer_reward(action, next_obs)
#         elif env_name == "Hopper-v2":
#             return self.Hopper_reward(action, next_obs)
#         elif env_name == "HalfCheetah-v2":
#             return self.HalfCheetah_reward(action, next_obs)
#         elif env_name == "Walker2d-v2":
#             return self.Walker_reward(action, next_obs)
#         elif env_name == "Ant-v2":
#             return self.Ant_reward(action, next_obs)
#         elif env_name == "Humanoid-v2":
#             return self.Humanoid_reward(action, next_obs)
    
#     def MC_reward(self, action, next_obs):
#         position = next_obs[0]
#         velocity = next_obs[1]
#         goal_position =  0.45
#         goal_velocity = 0
#         # Convert a possible numpy bool to a Python bool.
#         done = bool(position >= goal_position and velocity >= goal_velocity)
#         reward = 0
#         if done:
#             reward = 100.0
#         reward -= math.pow(action[0], 2) * 0.1
#         return reward

#     def Reacher_reward(self, action, next_obs):
#         pass
#     def Swimmer_reward(self, action, next_obs):
#         pass
#     def Hopper_reward(self, action, next_obs):
#         pass
#     def HalfCheetah_reward(self, action, next_obs):
#         pass
#     def Walker_reward(self, action, next_obs):
#         pass
#     def Ant_reward(self, action, next_obs):
#         pass
#     def Humanoid_reward(self, action, next_obs):
#         pass
    
    
    
    
    

# Termination Functions
class Termination():
    def __init__(self, env_name):
        self.env_name = env_name
        
    def __call__(self, action, next_obs):
        if env_name == "MountainCarContinuous-v0":
            return self.MC(action, next_obs)
        elif env_name == "HalfCheetah-v2":
            return self.HalfCheetah(action, next_obs)
        elif env_name == "Walker2d-v2":
            return self.Walker(action, next_obs)
        elif env_name == "Ant-v2":
            return self.Ant(action, next_obs)
    
    def MC(self, action, next_obs):
        position = next_obs[:, 0]
        velocity = next_obs[:, 1]
        goal_position =  0.45
        goal_velocity = 0
        done = (torch.logical_and(position >= goal_position, velocity >= goal_velocity))
        return done.reshape(-1, 1)
    def HalfCheetah(self, action, next_obs):
        return termination_fns.no_termination
    def Walker(self, action, next_obs):
        return termination_fns.walker2d
    def Ant(self, action, next_obs):
        return termination_fns.ant

