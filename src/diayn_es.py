'''
Skills discovery with Evolution Straigies.
'''
import gym

import numpy as np

import torch
torch.set_num_threads(1) # fix the hang up bug
torch.multiprocessing.set_sharing_strategy('file_system')
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as opt
from torch.utils.tensorboard import SummaryWriter
import copy

from stable_baselines3.common.vec_env.dummy_vec_env import DummyVecEnv
from stable_baselines3 import SAC
from stable_baselines3.common.callbacks import EvalCallback
from stable_baselines3.common.monitor import Monitor

from src.environment_wrappers.env_wrappers import RewardWrapper, SkillWrapper, SkillWrapperFinetune
from src.environment_wrappers.tasks_wrappers import HalfCheetahTaskWrapper, WalkerTaskWrapper, AntTaskWrapper
from src.utils import  best_skill, evaluate_pretrained_policy_intr, evaluate_pretrained_policy_ext
from src.models.models import Discriminator, MLP_policy
from src.replayBuffers import DataBuffer
from src.callbacks.callbacks import DiscriminatorCallback, VideoRecorderCallback
from src.individuals import MountainCar, Swimmer, HalfCheetah, Ant, Walker
from src.individuals.population import NormalPopulation
from src.config import conf

from evostrat import compute_centered_ranks
from torch.multiprocessing import Pool
from torch.distributions import MultivariateNormal

import time
import tqdm

import os

import matplotlib.pyplot as plt
import seaborn as sns
sns.set_theme(style="darkgrid")
sns.set(font_scale = conf.font_scale)


class DIAYN_ES():
    '''
    An implementation of DIAYN with a black-box optimization algorithm (Evolution Stratigies) as the optimization backbone
    '''
    def __init__(self, params, es_params, discriminator_hyperparams, env="MountainCarContinuous-v0", alg="es", directory="./", seed=10, device="cpu", conf=None, timestamp=None, checkpoints=False, args=None, task=None, adapt_params=None, n_samples=None):
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
        # replay buffer
        self.buffer = DataBuffer(params['buffer_size'], obs_shape=gym.make(
            env).observation_space.shape[0])
        
        self.optimizer = opt.Adam(self.d.parameters(), lr=discriminator_hyperparams['learning_rate'])

        # tensorboard summary writers
        # tensorboard summary writer
        self.sw = SummaryWriter(
            log_dir=f"{directory}/ES_PreTrain", comment=f"{alg}, env_name:{env}, PreTrain")
        self.sw_adapt = SummaryWriter(
            log_dir=f"{directory}/ES_FineTune", comment=f"{alg}, env_name:{env}, FineTune")
        
        # save some attributes
        # general shared parameters
        self.params = params 
        # Evolution Stratigies hyperparameters
        self.es_params = es_params 
        # Discriminator hyperparameters
        self.discriminator_hyperparams = discriminator_hyperparams 
        # gym environment name
        self.env_name = env 
        # experiment directory
        self.directory = directory 
        # random seed
        self.seed = seed 
        # unique timesamp
        self.timestamp = timestamp 
        # configuration
        self.conf = conf 
        # if not none finetune with a fixed freq. and save the result, this for the scaling experiment
        self.checkpoints = checkpoints 
        # cmd args
        self.args = args 
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
        # hyeperparameters for the adaptation algorithm
        self.adapt_params = adapt_params
        # checkpoint
        self.checkpoints = checkpoints
        # number of checkpoints
        self.n_samples = n_samples
        # task label
        self.task = task
        
    # pretraining step with intrinsic reward    
    def pretrain(self):
        '''
        Pretraining phase with an intrinsic reward
        '''
        # create the indiviual object according to the environment
        if self.env_name == "MountainCarContinuous-v0":
            MountainCar.MountainCar.d = self.d
            MountainCar.MountainCar.n_skills = self.params['n_skills']
            MountainCar.MountainCar.skill = None
            MountainCar.MountainCar.paramerization = self.paramerization
            env_indv = MountainCar.MountainCar()
            param_shapes = {k: v.shape for k, v in MountainCar.MountainCar().get_params().items()}
            population = NormalPopulation(param_shapes, env_indv.from_params, std=0.1)
        elif self.env_name == "Swimmer-v2":
            Swimmer.Swimmer.d = self.d
            Swimmer.Swimmer.n_skills = self.params['n_skills']
            Swimmer.Swimmer.paramerization = self.paramerization
            Swimmer.Swimmer.skill = None
            env_indv = Swimmer.Swimmer()
            # set up the population
            param_shapes = {k: v.shape for k, v in Swimmer.Swimmer().get_params().items()}
            population = NormalPopulation(param_shapes, env_indv.from_params, std=0.1)
        elif self.env_name == "Walker2d-v2":
            Walker.Walker.d = self.d
            Walker.Walker.n_skills = self.params['n_skills']
            Walker.Walker.paramerization = self.paramerization
            Walker.Walker.skill = None
            env_indv = Walker.Walker()
            # set up the population
            param_shapes = {k: v.shape for k, v in Walker.Walker().get_params().items()}
            population = NormalPopulation(param_shapes, env_indv.from_params, std=0.1)
        elif self.env_name == "HalfCheetah-v2":
            HalfCheetah.HalfCheetah.d = self.d
            HalfCheetah.HalfCheetah.n_skills = self.params['n_skills']
            HalfCheetah.HalfCheetah.paramerization = self.paramerization
            HalfCheetah.HalfCheetah.skill = None
            env_indv = HalfCheetah.HalfCheetah()
            # set up the population
            param_shapes = {k: v.shape for k, v in HalfCheetah.HalfCheetah().get_params().items()}
            population = NormalPopulation(param_shapes, env_indv.from_params, std=0.1)
        elif self.env_name == "Ant-v2":
            Ant.Ant.d = self.d
            Ant.Ant.n_skills = self.params['n_skills']
            Ant.Ant.paramerization = self.paramerization
            Ant.Ant.skill = None
            env_indv = Ant.Ant()
            # set up the population
            param_shapes = {k: v.shape for k, v in Ant.Ant().get_params().items()}
            population = NormalPopulation(param_shapes, env_indv.from_params, std=0.1)
        else:
            raise ValueError(f'Environment {self.env_name} is not implemented')
        # extract the hyperparameters
        iterations = self.es_params['iterations']
        lr = self.es_params['lr'] # learning rate
        pop_size = self.es_params['pop_size'] # population size        
        # set up the training loop
        optim = opt.Adam(population.parameters(), lr=lr)
        pbar = tqdm.tqdm(range(iterations))
        pool = Pool()
        log_freq = 1
        self.timesteps = 0 
        best_fit = -np.inf
        results = []
        timesteps = []
        # for scalability experiment
        if self.checkpoints:
            steps_l = []
            intr_rewards = []
            extr_rewards = []
            freq = iterations // self.n_samples
            print(f"checkpoints freq: {freq}")
            indv = self.env_indv_from_name()
            params = population.param_means
            pretrained_policy = indv.from_params(params)
            intr_reward = np.mean( [evaluate_pretrained_policy_intr(self.env_name, self.n_skills, pretrained_policy, self.d, self.paramerization, "es") for _ in range(10) ] )
            extr_reward = np.mean( [evaluate_pretrained_policy_ext(self.env_name, self.n_skills, pretrained_policy, "es") for _ in range(10)] )
            steps_l.append(0)
            intr_rewards.append(intr_reward)
            extr_rewards.append(extr_reward)
            plt.figure()
            plt.plot(steps_l, extr_rewards, label="es".upper())
            plt.xlabel("Pretraining Steps")
            plt.ylabel("Extrinsic Reward")
            plt.legend()
            plt.tight_layout()
            filename = f"Scalability_Experiment_realtime_env:{self.env_name}_alg:es_xaxis:Pretraining Steps.png"
            plt.savefig(f'{filename}', dpi=150) 
            plt.figure()
            plt.plot(intr_rewards, extr_rewards, label="es".upper())
            plt.xlabel("Intrinsic Reward")
            plt.ylabel("Extrinsic Reward")
            plt.legend()
            plt.tight_layout()
            filename = f"Scalability_Experiment_realtime_env:{self.env_name}_alg:es_xaxis:Intrinsic Reward.png"
            plt.savefig(f'{filename}', dpi=150) 
        for i in pbar:
            optim.zero_grad()
            # update the policy
            raw_fit, data, steps = population.fitness_grads(pop_size, pool, compute_centered_ranks)
            self.timesteps += steps
            for d in data:
                self.buffer.add(d[0], d[1])
            optim.step()
            # TODO: update the discriminator
            d_loss, d_grad_norm, d_weight_norm = None, None, None
            if len(self.buffer) > self.conf.min_train_size:
                # for i in range(10):
                d_loss, d_grad_norm, d_weight_norm = self.update_d()
            pbar.set_description("fit avg (training): %0.3f, std: %0.3f" % (raw_fit.mean().item(), raw_fit.std().item()))
            if (i+1) % log_freq == 0:
                evals_path = f"{self.directory}/eval_results" 
                os.makedirs(evals_path, exist_ok=True)
                eval_fit = self.evaluate_policy(population, env_indv, False)
                # Save the model if there is an improvement.
                if eval_fit.mean().item() > best_fit:
                    best_fit = eval_fit.mean().item()
                    print(f"New best fit: {best_fit}")
                    torch.save(population.param_means, f"{self.directory}/best_model.ckpt") 
                    pbar.set_description("fit avg (evaluation): %0.3f, std: %0.3f" % (eval_fit.mean().item(), eval_fit.std().item()))
                # log all values
                timesteps.append(self.timesteps)
                results.append(eval_fit.mean().item())
                timesteps_np = np.array(timesteps)
                # print(f"results: {self.results}")
                results_np = np.array(results)
                np.savez(f"{evals_path}/evaluations.npz", timesteps=timesteps_np, results=results_np)
                self.sw.add_scalar("rollout/ep_rew_mean", raw_fit.mean().item(), self.timesteps)
                self.sw.add_scalar("eval/mean_reward", eval_fit.mean(), self.timesteps)
                # print the logs
                printed_logs = f'''
                Iteration: {i}
                Timesteps: {self.timesteps}
                Training reward: {raw_fit.mean().item()}
                Evaluation reward: {eval_fit.mean()}
                Discriminator loss: {d_loss}
                Discriminator grad norm: {d_grad_norm}
                Discriminator weights norm: {d_weight_norm}
                '''
                print(printed_logs)
            # if checkpoints is true add to the logs
            if self.checkpoints and (i%freq == 0):
                indv = self.env_indv_from_name()
                params = population.param_means
                pretrained_policy = indv.from_params(params)
                intr_reward = np.mean( [evaluate_pretrained_policy_intr(self.env_name, self.n_skills, pretrained_policy, self.d, self.paramerization, "es") for _ in range(10) ] )
                extr_reward = np.mean( [evaluate_pretrained_policy_ext(self.env_name, self.n_skills, pretrained_policy, "es") for _ in range(10)] )
                steps_l.append(self.timesteps)
                intr_rewards.append(intr_reward)
                extr_rewards.append(extr_reward)
                plt.figure()
                plt.plot(steps_l, extr_rewards, label="es".upper())
                plt.xlabel("Pretraining Steps")
                plt.ylabel("Extrinsic Reward")
                plt.legend()
                plt.tight_layout()
                filename = f"Scalability_Experiment_realtime_env:{self.env_name}_alg:es_xaxis:Pretraining Steps.png"
                plt.savefig(f'{filename}', dpi=150) 
                plt.figure()
                plt.plot(intr_rewards, extr_rewards, label="es".upper())
                plt.xlabel("Intrinsic Reward")
                plt.ylabel("Extrinsic Reward")
                plt.legend()
                plt.tight_layout()
                filename = f"Scalability_Experiment_realtime_env:{self.env_name}_alg:es_xaxis:Intrinsic Reward.png"
                plt.savefig(f'{filename}', dpi=150) 
                
        pool.close()
        if self.checkpoints:
            results = {
                'steps': steps_l,
                'intr_rewards': intr_rewards,
                'extr_rewards': extr_rewards
            }
            return env_indv, self.d, results
        return env_indv, self.d
    
    
    def env_indv_from_name(self):
        if self.env_name == "MountainCarContinuous-v0":
            MountainCar.MountainCar.n_skills = self.params['n_skills']
            env_indv = MountainCar.MountainCar()
        elif self.env_name == "Walker2d-v2":
            Walker.Walker.n_skills = self.params['n_skills']
            env_indv = Walker.Walker()
        elif self.env_name == "HalfCheetah-v2":
            HalfCheetah.HalfCheetah.n_skills = self.params['n_skills']
            env_indv = HalfCheetah.HalfCheetah()
        elif self.env_name == "Ant-v2":
            Ant.Ant.n_skills = self.params['n_skills']
            env_indv = Ant.Ant()
        else:
            raise ValueError(f'Environment {self.env_name} is not implemented')
        return env_indv
        
        
    # Finetune on a pretrained policy
    def finetune(self):
        self.d.eval()
        '''
        Adapt the pretrained model with task reward using SAC
        '''
        # create the indiviual object according to the environment
        if self.env_name == "MountainCarContinuous-v0":
            MountainCar.MountainCar.n_skills = self.params['n_skills']
            env_indv = MountainCar.MountainCar()
            # set up the population
            param_shapes = {k: v.shape for k, v in MountainCar.MountainCar().get_params().items()}
            population = NormalPopulation(param_shapes, env_indv.from_params, std=0.1)
        elif self.env_name == "Walker2d-v2":
            Walker.Walker.n_skills = self.params['n_skills']
            if self.task:
                Walker.Walker.task = self.task
            env_indv = Walker.Walker()
            # set up the population
            param_shapes = {k: v.shape for k, v in Walker.Walker().get_params().items()}
            population = NormalPopulation(param_shapes, env_indv.from_params, std=0.1)
        elif self.env_name == "HalfCheetah-v2":
            HalfCheetah.HalfCheetah.n_skills = self.params['n_skills']
            if self.task:
                HalfCheetah.HalfCheetah.task = self.task
            env_indv = HalfCheetah.HalfCheetah()
            # set up the population
            param_shapes = {k: v.shape for k, v in HalfCheetah.HalfCheetah().get_params().items()}
            population = NormalPopulation(param_shapes, env_indv.from_params, std=0.1)
        elif self.env_name == "Ant-v2":
            Ant.Ant.n_skills = self.params['n_skills']
            if self.task:
                Ant.Ant.task = self.task
            env_indv = Ant.Ant()
            # set up the population
            param_shapes = {k: v.shape for k, v in Ant.Ant().get_params().items()}
            population = NormalPopulation(param_shapes, env_indv.from_params, std=0.1)
        else:
            raise ValueError(f'Environment {self.env_name} is not implemented')
        # define the model directory and load it.
        model_dir = f"{self.directory}/""best_model.ckpt"
        state_dim = gym.make(self.env_name).observation_space.shape[0]
        action_dim = gym.make(self.env_name).action_space.shape[0]
        model = MLP_policy(state_dim + self.n_skills, [self.conf.layer_size_policy, self.conf.layer_size_policy], action_dim)
        model.load_state_dict( torch.load(model_dir) )
        env_indv.net = model
        # extraact the best skill
        bestskill = best_skill(env_indv, self.env_name,  self.params['n_skills'], alg_type="es")
        # create the SAC adaptation model
        env, eval_env = self.adaptation_environment(bestskill)

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
        
        eval_env = Monitor(eval_env, f"{self.directory}/finetune_eval_results")

        eval_callback = EvalCallback(eval_env, best_model_save_path=self.directory + f"/best_finetuned_model_skillIndex:{bestskill}",
                                    log_path=f"{self.directory}/finetune_eval_results", eval_freq=self.conf.eval_freq,
                                    deterministic=True, render=False)

        env = DummyVecEnv([lambda: SkillWrapperFinetune(Monitor(gym.make(
    self.env_name),  f"{self.directory}/finetune_train_results"), self.params['n_skills'], r_seed=None,max_steps=gym.make(self.env_name)._max_episode_steps, skill=bestskill)])

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
                                       self.conf.layer_size_q, self.conf.layer_size_q]),  n_critics=1),
                    tensorboard_log=self.directory,
                    seed=self.seed
                    )

        eval_env = SkillWrapperFinetune(gym.make(
            self.env_name), self.params['n_skills'], max_steps=gym.make(self.env_name)._max_episode_steps, r_seed=None, skill=bestskill)
        eval_env = Monitor(eval_env, f"{self.directory}/finetune_eval_results")

        eval_callback = EvalCallback(eval_env, best_model_save_path=self.directory + f"/best_finetuned_model_skillIndex:{bestskill}",
                                    log_path=f"{self.directory}/finetune_eval_results", eval_freq=self.conf.eval_freq,
                                    deterministic=True, render=False)

        self.load_state_dicts(model)
        self.adaptation_model.learn(total_timesteps=self.params['finetune_steps'],
                    callback=eval_callback, tb_log_name="ES_FineTune", d=None, mi_estimator=None)

        return self.adaptation_model, bestskill
    
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
    
    def load_state_dicts(self, es_model):
        '''
        load the pretrained model parameters into the adaptation model
        '''
        # initlialize the adaptation policy with the pretrained model
        self.adaptation_model.actor.latent_pi.load_state_dict(es_model.model.state_dict())
        self.adaptation_model.actor.mu.load_state_dict(es_model.mean.state_dict())
        self.adaptation_model.actor.log_std.load_state_dict(es_model.log_var.state_dict())
        # initlialize the adaptation critic with the discriminator weights
        self.d.layers[0] = nn.Linear(gym.make(self.env_name).observation_space.shape[0] + self.params['n_skills']  + gym.make(self.env_name).action_space.shape[0], self.conf.layer_size_discriminator)
        self.d.layers[-1] = nn.Linear(self.conf.layer_size_discriminator, 1)
        seq = nn.Sequential(*self.d.layers)
        # print(d)
        self.adaptation_model.critic.qf0.load_state_dict(seq.state_dict())
        self.adaptation_model.critic_target.load_state_dict(self.adaptation_model.critic.state_dict())
        
    
    def update_d(self):
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
            grad_norms, weights_norm = self.norms(self.d)
            self.sw.add_scalar("discrimiator loss",
                                        epoch_loss, self.timesteps)
            self.sw.add_scalar("discrimiator grad norm",
                            grad_norms, self.timesteps)
            self.sw.add_scalar("discrimiator wights norm",
                            weights_norm, self.timesteps)
            torch.save(self.d.state_dict(), self.directory + "/disc.pth")
        return epoch_loss, grad_norms, weights_norm
    
    def evaluate_policy(self, population, indv,finetune=False, skill=None):
        # set the seeds
        runs = 10
        params = population.param_means
        model = indv.from_params(params)
        # create the environments
        if finetune:
            env = SkillWrapperFinetune(gym.make(self.env_name), self.params['n_skills'], max_steps=self.conf.max_steps, skill=skill)
        else:
            env = RewardWrapper(SkillWrapper(gym.make(self.env_name), self.params['n_skills'], max_steps=self.conf.max_steps), self.d, self.params['n_skills'])
        # run the evaluation loop
        final_returns = []
        for run in range(runs):
            obs = env.reset()
            done = False
            total_reward = 0
            while not done:
                action = model.action(obs)
                obs, reward, done, _ = env.step(action) 
                total_reward += reward
            final_returns.append(total_reward)
        return np.array(final_returns)
        
    def norms(self, net):
        '''
        norms(net)
        This function calulate the gradient and weight norms of a neural network model.
        net: network model
        '''
        total_norm = 0
        total_weights_norm = 0
        for param in net.parameters():
            param_norm = param.grad.detach().data.norm(2)
            total_weights_norm += param.detach().data.norm(2) ** 2
            total_norm += param_norm.item() ** 2
        return total_norm**0.5, total_weights_norm**0.5
        
        
        
        