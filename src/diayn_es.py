'''
Skills discovery with Evolution Straigies.
'''
import gym

import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as opt
from torch.utils.tensorboard import SummaryWriter

from src.environment_wrappers.env_wrappers import RewardWrapper, SkillWrapper, SkillWrapperFinetune
from src.utils import record_video_finetune, best_skill
from src.models.models import Discriminator
from src.replayBuffers import DataBuffer
from src.callbacks.callbacks import DiscriminatorCallback, VideoRecorderCallback, FineTuneCallback
from src.individuals import MountainCar, Swimmer, Hopper, HalfCheetah, Ant

import time


class DIAYN_ES():
    
    def __init__(self, params, es_params, discriminator_hyperparams, env="MountainCarContinuous-v0", directory="./", seed=10, device="cpu", conf=None, timestamp=None, checkpoints=False, args=None, paramerization=0):
        ### Discriminator ###
        # check the parametrization
        if paramerization == 0:
            self.d = Discriminator(gym.make(env).observation_space.shape[0], [
                                   conf.layer_size_discriminator, conf.layer_size_discriminator], params['n_skills']).to(device)
        elif paramerization == 1:
            pass # TODO: add the CPC style discriminator
        # replay buffer
        self.buffer = DataBuffer(params['buffer_size'], obs_shape=gym.make(
            env).observation_space.shape[0])
        self.optimizer = opt.Adam(self.d.parameters(), lr=discriminator_hyperparams['learning_rate'])
        self.criterion =  F.cross_entropy

        # tensorboard summary writer for the discriminator
        # print(f"Sumary writer comment is: {alg} Discriminator, env_name:{env}, weight_decay:{discriminator_hyperparams['weight_decay']}, dropout:{discriminator_hyperparams['dropout']}, label_smoothing:{discriminator_hyperparams['label_smoothing']}, gradient_penalty:{discriminator_hyperparams['gp']}, mixup:{discriminator_hyperparams['mixup']} ")
        self.sw = SummaryWriter(
            log_dir=directory, filename_suffix=f"{alg} Discriminator, env_name:{env}, weight_decay:{discriminator_hyperparams['weight_decay']}, dropout:{discriminator_hyperparams['dropout']}, label_smoothing:{discriminator_hyperparams['label_smoothing']}, gradient_penalty:{discriminator_hyperparams['gp']}, mixup:{discriminator_hyperparams['mixup']}")
        
        # TODO: create a summary writer for the ES algorithm
        self.sw_policy = SummaryWriter( log_dir=f"{directory}/ES", comment=f"ES, env_name:{env}" )
        
        # TODO: create a summary writer for the ES algorithm
        self.sw_policy_finetune = SummaryWriter( log_dir=f"{directory}/ES", comment=f"ES_finetune, env_name:{env}" )

        # save some attributes
        self.params = params # general shared parameters
        self.es_params = es_params # Evolution Stratigies hyperparameters
        self.discriminator_hyperparams = discriminator_hyperparams # Discriminator hyperparameters
        self.env_name = env # gym environment name
        self.directory = directory # experiment directory
        self.seed = seed # random seed
        self.timestamp = timestamp # unique timesamp
        self.conf = conf # configuration
        self.checkpoints = checkpoints # if not none finetune with a fixed freq. and save the result, this for the scaling experiment
        self.args = args # cmd args
        self.paramerization = paramerization # discriminator parametrization if 0 use MLP, else if 1 use CPC style discriminator
        
    # pretraining step with intrinsic reward    
    def pretrain(self):
        # create the indiviual object according to the environment
        if self.env_name == "MountainCarContinuous-v0":
            env_indv = MountainCar()
            # set up the population
            param_shapes = {k: v.shape for k, v in MountainCar().get_params().items()}
            population = NormalPopulation(param_shapes, env_indv.from_params, std=0.1)
        elif self.env_name == "Swimmer-v2":
            env_indv = Swimmer()
            # set up the population
            param_shapes = {k: v.shape for k, v in Swimmer().get_params().items()}
            population = NormalPopulation(param_shapes, env_indv.from_params, std=0.1)
        elif self.env_name == "Hopper-v2":
            env_indv = Hopper()
            # set up the population
            param_shapes = {k: v.shape for k, v in Hopper().get_params().items()}
            population = NormalPopulation(param_shapes, env_indv.from_params, std=0.1)
        elif self.env_name == "HalfCheetah-v2":
            env_indv = HalfCheetah()
            # set up the population
            param_shapes = {k: v.shape for k, v in HalfCheetah().get_params().items()}
            population = NormalPopulation(param_shapes, env_indv.from_params, std=0.1)
        elif self.env_name == "Ant-v2":
            env_indv = Ant()
            # set up the population
            param_shapes = {k: v.shape for k, v in Ant().get_params().items()}
            population = NormalPopulation(param_shapes, env_indv.from_params, std=0.1)
        else:
            raise ValueError(f'Environment {self.env_name} is not implemented')
        # set the discriminator
        env_indv.set_discriminator(self.d)
        # set the number of skills
        env_indv.set_n_skills(self.params['n_skills'])
        # set the data buffer
        env_indv.set_dataBuffer(self.buffer)
        # TODO: extract the hyperparameters
        iterations = self.es_params['iterations']
        lr = self.es_params['lr'] # learning rate
        pop_size = self.es_params['pop_size'] # population size
        
        # set up the training loop
        optim = opt.Adam(population.parameters(), lr=lr)
        pbar = tqdm.tqdm(range(iterations))
        pool = Pool()
        log_freq = 10
        # TODO: add tensorflow logs
        # TODO: add the discriminator update with the network update
        best_fit = -np.inf
        for i in pbar:
            optim.zero_grad()
            # update the policy
            raw_fit = population.fitness_grads(pop_size, pool, compute_centered_ranks)
            optim.step()
            # TODO: update the discriminator
            d_loss, d_grad_norm, d_weight_norm = self.update_d()
            pbar.set_description("fit avg (training): %0.3f, std: %0.3f" % (raw_fit.mean().item(), raw_fit.std().item()))
            if (i+1) % evaluate_freq == 0:
                eval_fit = self.evaluate_policy()
                # Save the model if there is an improvement.
                if eval_fit.mean().item() > best_fit:
                    best_fit = eval_fit.mean().item()
                    print(f"New best fit: {best_fit}")
                    torch.save(env_indv.net.state_dict(), "best_model.ckpt") 
                    pbar.set_description("fit avg (evaluation): %0.3f, std: %0.3f" % (eval_fit.mean().item(), eval_fit.std().item()))
                # log all values
                self.sw_policy.add_scalar("train_reward", raw_fit.mean().item(), i)
                self.sw_policy.add_scalar("eval_reward", eval_fit.mean(), i)
                self.sw.add_scalar("discriminator_loss", d_loss, i)
                self.sw.add_scalar("discriminator_grad_norm", d_grad_norm, i)
                self.sw.add_scalar("discriminator_grad_norm", d_weight_norm, i)
                # print the logs
                printed_logs = f'''
                Iteration: {i}
                Training reward: {raw_fit.mean().item()}
                Evaluation reward: {eval_fit.mean()}
                Discriminator loss: {d_loss}
                Discriminator grad norm: {d_grad_norm}
                Discriminator weights norm: {d_weight_norm}
                '''
                print(printed_logs)
        pool.close()
        
    # Finetune on a pretrained policy
    def finetine(self):
        # create the indiviual object according to the environment
        if self.env_name == "MountainCarContinuous-v0":
            env_indv = MountainCar()
            # set up the population
            param_shapes = {k: v.shape for k, v in MountainCar().get_params().items()}
            population = NormalPopulation(param_shapes, env_indv.from_params, std=0.1)
        elif self.env_name == "Swimmer-v2":
            env_indv = Swimmer()
            # set up the population
            param_shapes = {k: v.shape for k, v in Swimmer().get_params().items()}
            population = NormalPopulation(param_shapes, env_indv.from_params, std=0.1)
        elif self.env_name == "Hopper-v2":
            env_indv = Hopper()
            # set up the population
            param_shapes = {k: v.shape for k, v in Hopper().get_params().items()}
            population = NormalPopulation(param_shapes, env_indv.from_params, std=0.1)
        elif self.env_name == "HalfCheetah-v2":
            env_indv = HalfCheetah()
            # set up the population
            param_shapes = {k: v.shape for k, v in HalfCheetah().get_params().items()}
            population = NormalPopulation(param_shapes, env_indv.from_params, std=0.1)
        elif self.env_name == "Ant-v2":
            env_indv = Ant()
            # set up the population
            param_shapes = {k: v.shape for k, v in Ant().get_params().items()}
            population = NormalPopulation(param_shapes, env_indv.from_params, std=0.1)
        else:
            raise ValueError(f'Environment {self.env_name} is not implemented')
        # define the model directory and load it.
        model_dir = f"{self.directory}/""best_model.ckpt"
        env_indv.net.load_state_dict(torch.load(model_dir))
        
        # extraact the best skill
        bestskill = best_skill(env_indv, self.env_name,  self.params['n_skills'], alg_type="es")
        # set the best skill
        env_indv.set_skill(bestskill)
        
        # training loop
        # TODO: extract the hyperparameters
        iterations = self.es_params['iterations_finetune']
        lr = self.es_params['lr'] # learning rate
        pop_size = self.es_params['pop_size'] # population size
        
        # set up the training loop
        optim = opt.Adam(population.parameters(), lr=lr)
        pbar = tqdm.tqdm(range(iterations))
        pool = Pool()
        log_freq = 10
        best_fit = -np.inf
        for i in pbar:
            optim.zero_grad()
            # update the policy
            raw_fit = population.fitness_grads(pop_size, pool, compute_centered_ranks)
            optim.step()
            pbar.set_description("fit avg (training): %0.3f, std: %0.3f" % (raw_fit.mean().item(), raw_fit.std().item()))
            if (i+1) % evaluate_freq == 0:
                eval_fit = self.evaluate_policy()
                if eval_fit.mean().item() > best_fit:
                    best_fit = eval_fit.mean().item()
                    print(f"New best fit: {best_fit}")
                    torch.save(env_indv.net.state_dict(), "best_model.ckpt") 
                    pbar.set_description("fit avg (evaluation): %0.3f, std: %0.3f" % (eval_fit.mean(), eval_fit.std()))
                # log all values
                self.sw_policy_finetune.add_scalar("train_reward", raw_fit.mean().item(), i)
                self.sw_policy_finetune.add_scalar("eval_reward", eval_fit.mean(), i)
                printed_logs = f'''
                Iteration: {i}
                Training reward: {raw_fit.mean().item()}
                Evaluation reward: {eval_fit.mean()}
                Discriminator loss: {d_loss}
                Discriminator grad norm: {d_grad_norm}
                Discriminator weights norm: {d_weight_norm}
                '''
                print(printed_logs)
        pool.close()        
    
    # perform a gradient step to train the discriminator
    def update_d(self):
        # sample data from the replay buffer
        inputs, targets = self.buffer.sample(self.discriminator_hyperparams['batch_size'])
        # forward pass
        outputs = self.d(inputs)
        # compute the loss
        loss = self.criterion(outputs, targets.to(conf.device).to(torch.int64))
        # optimize
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        # compute the weights and gradient norms
        grad_norm, weight_norm = self.norms(self.d)
        return loss.item(), grad_norm, weight_norm 
    
    
    def evaluate_policy(self, model, finetune=False, skill=None):
        # set the seeds
        seeds = [0, 10, 1234, 5, 42]
        # create the environments
        if finetune:
            env = SkillWrapperFinetune(gym.make(self.env_name), self.params['n_skills'], max_steps=self.conf.max_steps, skill=self.skill)
        else:
            env = RewardWrapper(SkillWrapper(gym.make(self.env_name), self.params['n_skills'],, max_steps=self.conf.max_steps), self.d, self.params['n_skills'])
        # run the evaluation loop
        final_returns = []
        for seed in seeds:
            env.seed(seed)
            obs = env.reset()
            done = False
            total_reward = 0
            while not done:
                action = Env.action(obs)
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
        
        
        
        