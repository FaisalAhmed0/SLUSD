'''
Skills discovery with Evolution Straigies.
'''
import gym

from src.environment_wrappers.env_wrappers import RewardWrapper, SkillWrapper, SkillWrapperFinetune
from src.utils import record_video_finetune, best_skill
from src.models.models import Discriminator
from src.replayBuffers import DataBuffer
from src.callbacks.callbacks import DiscriminatorCallback, VideoRecorderCallback, FineTuneCallback
from src.individuals import MountainCar, Swimmer, Hopper, HalfCheetah, Ant

import torch.nn as nn
from torch.utils.tensorboard import SummaryWriter

import time


class DIAYN_ES():
    
    def __init__(self, params, es_params, discriminator_hyperparams, env="MountainCarContinuous-v0", directory="./", seed=10, device="cpu", conf=None, timestamp=None, checkpoints=False, args=None):
        self.d = Discriminator(gym.make(env).observation_space.shape[0], [
                               conf.layer_size_discriminator, conf.layer_size_discriminator], params['n_skills']).to(device)
        self.buffer = DataBuffer(params['buffer_size'], obs_shape=gym.make(
            env).observation_space.shape[0])

        # tensorboard summary writer
        '''
        learning_rate = 3e-4,
        batch_size = 64,
        n_epochs = 1,
        weight_decay = 1e-1,
        dropout = 0,
        label_smoothing = None,
        gp = None,
        mixup = False
        '''
        # print(f"Sumary writer comment is: {alg} Discriminator, env_name:{env}, weight_decay:{discriminator_hyperparams['weight_decay']}, dropout:{discriminator_hyperparams['dropout']}, label_smoothing:{discriminator_hyperparams['label_smoothing']}, gradient_penalty:{discriminator_hyperparams['gp']}, mixup:{discriminator_hyperparams['mixup']} ")
        self.sw = SummaryWriter(
            log_dir=directory, filename_suffix=f"{alg} Discriminator, env_name:{env}, weight_decay:{discriminator_hyperparams['weight_decay']}, dropout:{discriminator_hyperparams['dropout']}, label_smoothing:{discriminator_hyperparams['label_smoothing']}, gradient_penalty:{discriminator_hyperparams['gp']}, mixup:{discriminator_hyperparams['mixup']}")
        
        # TODO: create a summary writer for the ES algorithm

        # save some attributes
        self.params = params
        self.es_params = es_params
        self.discriminator_hyperparams = discriminator_hyperparams
        self.env_name = env
        self.directory = directory
        self.seed = seed
        self.timestamp = timestamp
        self.conf = conf
        self.checkpoints = checkpoints
        self.args = args
        
    def pretrain(self):
        if self.env_name == "MountainCarContinuous-v0":
            env_indv = MountainCar()
        elif self.env_name == "Swimmer-v2":
            env_indv = Swimmer()
        elif self.env_name == "Hopper-v2":
            env_indv = Hopper()
        elif self.env_name == "HalfCheetah-v2":
            env_indv = HalfCheetah()
        elif self.env_name == "Ant-v2":
            env_indv = Ant()
        else:
            raise ValueError(f'Environment {self.env_name} is not implemented')
        # set the discriminator
        env_indv.set_discriminator(self.d)
        # set the number of skills
        env_indv.set_n_skills(self.params['n_skills'])
        # set the data buffer
        env_indv.set_dataBuffer(self.buffer)
        # TODO: extract the hyperparameters
        
        # set up the population
        param_shapes = {k: v.shape for k, v in LunarLander().get_params().items()}
        population = NormalPopulation(param_shapes, Env.from_params, std=0.1)
        # set up the training loop
        optim = opt.Adam(population.parameters(), lr=learning_rate)
        pbar = tqdm.tqdm(range(iterations))
        pool = Pool()
        # TODO: add tensorflow logs
        # TODO: add the discriminator update with the network update
        best_fit = -np.inf
        for _ in pbar:
            optim.zero_grad()
            raw_fit = population.fitness_grads(pop_size, pool, compute_centered_ranks)
            optim.step()
            pbar.set_description("fit avg: %0.3f, std: %0.3f" % (raw_fit.mean().item(), raw_fit.std().item()))
            if raw_fit.mean().item() > best_fit:
                best_fit = raw_fit.mean().item()
                print(f"New best fit: {best_fit}")
                torch.save(Env.net.state_dict(), "best_model.ckpt") 
        pool.close()
        
    def finetine(self):
        pass
    def update_d(self):
        pass
        
        
        
        