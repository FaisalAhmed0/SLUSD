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

import torch
import torch.multiprocessing as mp
torch.set_num_threads(1)
torch.multiprocessing.set_sharing_strategy('file_system')

import gym

from src.environment_wrappers.env_wrappers import RewardWrapper, SkillWrapper, SkillWrapperFinetune
from src.environment_wrappers.tasks_wrappers import HalfCheetahTaskWrapper
from src.utils import record_video_finetune, best_skill
from src.mi_lower_bounds import mi_lower_bound
from src.models.models import Discriminator, SeparableCritic, ConcatCritic
from src.replayBuffers import DataBuffer
from src.callbacks.callbacks import DiscriminatorCallback, VideoRecorderCallback, FineTuneCallback, MI_EvalCallback
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
    def __init__(self, params, alg_params, discriminator_hyperparams, env="MountainCarContinuous-v0", alg="ppo", directory="./", seed=10, device="cpu", conf=None, timestamp=None, checkpoints=False, args=None, task=None):
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
        
    def pretrain(self):
        '''
        pets_hyperparams = dict(
        learning_rate = 1e-3,
        batch_size = 64,
        ensemble_size = 5,
        trial_length = conf.max_steps/4,
        planning_horizon = 15,
        elite_ratio = 0.1,
        num_particles = 20,
        weight_decay = 5e-5
        algorithm = "pets"
        '''
        self.timesteps = []
        self.results = []
        # create MBRL lib gym-like env wrapper
        env_name = self.env_name
        seed = self.seed
        env = SkillWrapper(gym.make(env_name), self.params['n_skills'])
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
        num_trials = int(self.params['pretrain_steps'] // trial_length)
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
                "num_layers": 3,
                "ensemble_size": ensemble_size,
                "hid_size": 500,
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
                "normalize": True,
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
                        num_epochs=50, 
                        patience=50, 
                        # callback=train_callback,
                        silent=True)
                    
                    # add data to the discriminator buffer
                    env_obs, skill = self.split_obs(obs)
                    obs_t = torch.tensor(env_obs)
                    skill = torch.tensor(skill, device=conf.device)
                    self.buffer.add(obs_t, skill)
                    
                    # update the discriminator
                    self.update_discriminator()

                # --- Doing env step using the agent and adding to model dataset ---
                next_obs, reward, done, _ = common_util.step_env_and_add_to_buffer(
                    env, obs, agent, {}, replay_buffer)
                
                if timesteps % (conf.eval_freq * 5) == 0:
                    rewards = self.evaluate_policy(self.env_name, model_env)
                    self.timesteps.append(timesteps)
                    self.results.append(rewards)
                    timesteps = np.array(timesteps)
                    results = np.array(self.results)
                    np.savez(f"{self.directory}/eval_results/evaluations.npz", timesteps=timesteps, results=results)
                    # add to tensorboard
                    self.sw.add_scalar("eval/mean_reward", rewards.mean(), timesteps)
                    print(f"TimeStep: {timesteps}, Evaluation reward: {rewards.mean()}")
                timesteps += 1
                
                    
                obs = next_obs
                total_reward += reward
                steps_trial += 1

                if steps_trial == trial_length:
                    self.sw.add_scalar("rllout/train_reward", total_reward, timesteps)
                    print(f"TimeStep: {timesteps}, Rollout reward: {total_reward}")
                    break
    
    
    
    # common_util.rollout_model_env()
    def evaluate_policy(self, env_name, model_env):
        # cfg for the evaluation agent
        agent_cfg = omegaconf.OmegaConf.create({
        # this class evaluates many trajectories and picks the best one
        "_target_": "mbrl.planning.TrajectoryOptimizerAgent",
        "planning_horizon": 10,
        "replan_freq": 1,
        "verbose": False,
        "action_lb": "???",
        "action_ub": "???",
        # this is the optimizer to generate and choose a trajectory
        "optimizer_cfg": {
            "_target_": "mbrl.planning.CEMOptimizer",
            "num_iterations": 5,
            "elite_ratio": 0.1,
            "population_size": 500,
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
        
        env = SkillWrapper(gym.make(env_name), self.params['n_skills'])
        total_reward = 0
        obs = env.reset()
        for i in range(1000):
            action_seq = agent.plan(obs)
            obs, reward, done, _ = env.step(action_seq[0])
            total_reward += reward
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
    
    def update_discriminator(self):
        """
        This event is triggered before updating the policy.
        """
        current_buffer_size = len(self.buffer) 
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
                                        epoch_loss, self.num_timesteps)
            self.sw.add_scalar("discrimiator grad norm",
                            grad_norms, self.num_timesteps)
            self.sw.add_scalar("discrimiator wights norm",
                            weights_norm, self.num_timesteps)
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