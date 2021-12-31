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

import gym

from src.environment_wrappers.env_wrappers import RewardWrapper, SkillWrapper, SkillWrapperFinetune
from src.environment_wrappers.tasks_wrappers import HalfCheetahTaskWrapper
from src.utils import record_video_finetune, best_skill
from src.mi_lower_bounds import mi_lower_bound
from src.models.models import Discriminator, SeparableCritic, ConcatCritic
from src.replayBuffers import DataBuffer
from src.callbacks.callbacks import DiscriminatorCallback, VideoRecorderCallback, FineTuneCallback, MI_EvalCallback

import torch
import torch.nn as nn
import torch.nn.functional as F
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
            self.d = Discriminator(state_dim, hidden_dims, skill_dim, dropout=discriminator_hyperparams['dropout']).to(device)
        elif discriminator_hyperparams['parametrization'] == "Linear":
            self.d = nn.Linear(state_dim, skill_dim).to(device)
        elif discriminator_hyperparams['parametrization'] == "Separable":
            # state_dim, skill_dim, hidden_dims, latent_dim, temperature=1, dropout=None)
            self.d = SeparableCritic(state_dim, skill_dim, hidden_dims, latent_dim, temperature, dropout).to(device)
        elif discriminator_hyperparams['parametrization'] == "Concat":
            # state_dim, skill_dim, hidden_dims, temperature=1, dropout=None)
            self.d = ConcatCritic(state_dim, skill_dim, hidden_dims, temperature, dropout).to(device)
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
        
    def pretrain(self):
        # create MBRL lib gym-like env wrapper
        env_name = self.env_name
        seed = self.seed
        env = SkillWrapper(gym.make(env_name), self.params['n_skills'])
        env.env.seed(seed)
        rng = np.random.default_rng(seed=0)
        generator = torch.Generator(device=self.conf.device)
        generator.manual_seed(seed)
        obs_shape = env.observation_space.shape
        act_shape = env.action_space.shape
        reward_fn = # TODO: pass the reward class here
        term_fn = termination_fns.no_termination
        ## Configrations
        # TODO: replace this with the hyperparams arguments
        trial_length = 1000
        num_trials = 100
        ensemble_size = 5
        # Everything with "???" indicates an option with a missing value.
        # Our utility functions will fill in these details using the 
        # environment information
        # TODO: replace this with the hyperparams arguments
        cfg_dict = {
            # dynamics model configuration
            "dynamics_model": {
                "_target_": "mbrl.models.GaussianMLP",
                "device": device,
                "num_layers": 3,
                "ensemble_size": ensemble_size,
                "hid_size": 200,
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
                "num_steps": num_trials * trial_length,
                "model_batch_size": 32,
                "validation_ratio": 0.05
            }
        }
        # TODO: replace this with the hyperparams arguments
        cfg = omegaconf.OmegaConf.create(cfg_dict)
        # Create a 1-D dynamics model for this environment
        dynamics_model = common_util.create_one_dim_tr_model(cfg, obs_shape, act_shape)

        # Create a gym-like environment to encapsulate the model
        model_env = models.ModelEnv(env, dynamics_model, term_fn, r, generator=generator)
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
        print("# samples stored", replay_buffer.num_stored)
        # CEM (Cross-Entropy Method) agent
        # TODO: replace this with the hyperparams arguments
        agent_cfg = omegaconf.OmegaConf.create({
        # this class evaluates many trajectories and picks the best one
        "_target_": "mbrl.planning.TrajectoryOptimizerAgent",
        "planning_horizon": 15,
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
            "device": device,
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
            num_particles=20
        )
        # TODO: Add tensorboard
        # Main Training loop
        # Create a trainer for the model
        # TODO: replace this with the hyperparams arguments
        model_trainer = models.ModelTrainer(dynamics_model, optim_lr=1e-3, weight_decay=5e-5)
        # number of trials is the number of iterations 
        for trial in range(num_trials):
          # initilization
            obs = env.reset()    
            agent.reset()
            done = False
            total_reward = 0.0
            steps_trial = 0
            update_axes(axs, env.render(mode="rgb_array"), ax_text, trial, steps_trial, all_rewards)
            while not done:
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
                        callback=train_callback,
                        silent=True)

                # --- Doing env step using the agent and adding to model dataset ---
                next_obs, reward, done, _ = common_util.step_env_and_add_to_buffer(
                    env, obs, agent, {}, replay_buffer)

                update_axes(
                    axs, env.render(mode="rgb_array"), ax_text, trial, steps_trial, all_rewards)

                obs = next_obs
                total_reward += reward
                steps_trial += 1

                if steps_trial == trial_length:
                    break
        
        
        