import gym

from stable_baselines3.common.vec_env.dummy_vec_env import DummyVecEnv
from stable_baselines3 import PPO, SAC
from stable_baselines3.common.callbacks import EvalCallback
from stable_baselines3.common.utils import get_schedule_fn
from stable_baselines3.common.monitor import Monitor

from src.environment_wrappers.env_wrappers import RewardWrapper, SkillWrapper, SkillWrapperFinetune
from src.environment_wrappers.tasks_wrappers import HalfCheetahTaskWrapper
from src.utils import record_video_finetune, best_skill
from src.models.models import Discriminator, Discriminator_CPC
from src.replayBuffers import DataBuffer
from src.callbacks.callbacks import DiscriminatorCallback, VideoRecorderCallback, FineTuneCallback

import torch.nn as nn
from torch.utils.tensorboard import SummaryWriter

import time


class DIAYN():
    def __init__(self, params, alg_params, discriminator_hyperparams, env="MountainCarContinuous-v0", alg="ppo", directory="./", seed=10, device="cpu", conf=None, timestamp=None, checkpoints=False, args=None, task=None):
        # create the discirminator and the buffer
        if discriminator_hyperparams['parametrization'] == "MLP":
            self.d = Discriminator(gym.make(env).observation_space.shape[0], [
                                   conf.layer_size_discriminator, conf.layer_size_discriminator], params['n_skills'], dropout=discriminator_hyperparams['dropout']).to(device)
            self.buffer = DataBuffer(params['buffer_size'], obs_shape=gym.make(
                env).observation_space.shape[0])
        elif discriminator_hyperparams['parametrization'] == "CPC":
            # Discriminator_CPC(10, 6, [100, 100], 32)
            self.d = Discriminator_CPC(gym.make(env).observation_space.shape[0], params['n_skills'], [conf.layer_size_discriminator, conf.layer_size_discriminator], conf.latent_size).to(device)
        elif discriminator_hyperparams['parametrization'] == "linear":
            self.d = nn.Linear(gym.make(env).observation_space.shape[0], params['n_skills'])
            self.buffer = DataBuffer(params['buffer_size'], obs_shape=gym.make(
                env).observation_space.shape[0])
            print(f"linear disriminator: {self.d}")
            # input()
            
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
        print(f"Sumary writer comment is: {alg} Discriminator, env_name:{env}, weight_decay:{discriminator_hyperparams['weight_decay']}, dropout:{discriminator_hyperparams['dropout']}, label_smoothing:{discriminator_hyperparams['label_smoothing']}, gradient_penalty:{discriminator_hyperparams['gp']}, mixup:{discriminator_hyperparams['mixup']} ")
        self.sw = SummaryWriter(
            log_dir=directory, filename_suffix=f"{alg} Discriminator, env_name:{env}, weight_decay:{discriminator_hyperparams['weight_decay']}, dropout:{discriminator_hyperparams['dropout']}, label_smoothing:{discriminator_hyperparams['label_smoothing']}, gradient_penalty:{discriminator_hyperparams['gp']}, mixup:{discriminator_hyperparams['mixup']}")

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
        if self.alg == "ppo":
            env = DummyVecEnv([lambda: Monitor(RewardWrapper(SkillWrapper(gym.make(
                    self.env_name), self.params['n_skills'], max_steps=self.conf.max_steps), self.d, self.params['n_skills']),  self.directory)]*self.alg_params['n_actors'])
            # env = DummyVecEnv([
            #     lambda: Monitor(RewardWrapper(SkillWrapper(gym.make(
            #         self.env_name), self.params['n_skills'], max_steps=self.conf.max_steps), self.d, self.params['n_skills'], parametrization=self.parametrization),  self.directory),
            #     lambda: Monitor(RewardWrapper(SkillWrapper(gym.make(
            #         self.env_name), self.params['n_skills'], max_steps=self.conf.max_steps), self.d, self.params['n_skills'], parametrization=self.parametrization),  self.directory),
            #     lambda: Monitor(RewardWrapper(SkillWrapper(gym.make(
            #         self.env_name), self.params['n_skills'], max_steps=self.conf.max_steps), self.d, self.params['n_skills'], parametrization=self.parametrization),  self.directory),
            #     lambda: Monitor(RewardWrapper(SkillWrapper(gym.make(self.env_name), self.params['n_skills'], max_steps=self.conf.max_steps), self.d, self.params['n_skills'], parametrization=self.parametrization),  self.directory)])

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
                                           net_arch=[self.conf.layer_size_shared, self.conf.layer_size_shared,dict(pi=[self.conf.layer_size_policy], vf=[self.conf.layer_size_value])]),
                        tensorboard_log=self.directory,
                        seed=self.seed
                        )
            # Create Callbacks
            # video_loging_callback = VideoRecorderCallback(self.env_name, record_freq=5000, deterministic=self.d,
            #                                               n_z=self.params['n_skills'], videos_dir=self.conf.videos_dir + f"ppo_{self.env_name}_{self.timestamp}")
            # evaluation_callback = EvaluationCallBack(env_name, eval_freq=500, n_evals=eval_runs, log_dir=log_dir)
            discriminator_callback = DiscriminatorCallback(self.d, self.buffer, self.discriminator_hyperparams,
                                                           sw=self.sw, n_skills=self.params['n_skills'], min_buffer_size=self.params['min_train_size'], save_dir=self.directory, on_policy=True)

            eval_env = RewardWrapper(SkillWrapper(gym.make(
                self.env_name), self.params['n_skills'], ev=True), self.d, self.params['n_skills'], parametrization=self.parametrization)
            eval_env = Monitor(eval_env, f"{self.directory}/eval_results")
            eval_callback = EvalCallback(eval_env, best_model_save_path=self.directory,
                                         log_path=f"{self.directory}/eval_results", eval_freq=5000,
                                         deterministic=True, render=False)
            # create the callback list
            if self.checkpoints:
                # (self, args, params, alg_params, conf, seed, alg, timestamp)
                # print("will add the finetune callback")
                # input()
                fineune_callback = FineTuneCallback(self.args, self.params, self.alg_params, self.conf, self.seed, self.alg, self.timestamp)
                callbacks = [discriminator_callback, eval_callback, fineune_callback]
            else:
                callbacks = [discriminator_callback, eval_callback]
            # train the agent
            model.learn(total_timesteps=self.params['pretrain_steps'], callback=callbacks, log_interval=1, tb_log_name="PPO Pretrain")
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

            # Create Callbacks
            # video_loging_callback = VideoRecorderCallback(self.env_name, record_freq=5000, deterministic=self.d,
            #                                               n_z=self.params['n_skills'], videos_dir=self.conf.videos_dir + f"/sac_{self.env_name}_{self.timestamp}")
            discriminator_callback = DiscriminatorCallback(self.d, None, self.discriminator_hyperparams, sw=self.sw,
                                                           n_skills=self.params['n_skills'], min_buffer_size=self.params['min_train_size'], save_dir=self.directory, on_policy=False)

            eval_env = RewardWrapper(SkillWrapper(gym.make(
                self.env_name), self.params['n_skills'], ev=True), self.d, self.params['n_skills'], parametrization=self.parametrization)
            eval_env = Monitor(eval_env,  f"{self.directory}/eval_results")

            eval_callback = EvalCallback(eval_env, best_model_save_path=self.directory,
                                         log_path=f"{self.directory}/eval_results", eval_freq=5000, deterministic=True, render=False)
            
            # create the callback list
            if self.checkpoints:
                # (self, args, params, alg_params, conf, seed, alg, timestamp)
                # print("will add the finetune callback")
                # input()
                fineune_callback = FineTuneCallback(self.args, self.params, self.alg_params, self.conf, self.seed, self.alg, self.timestamp)
                callbacks = [discriminator_callback, eval_callback, fineune_callback]
            else:
                callbacks = [discriminator_callback, eval_callback]

            # train the agent
            if self.parametrization == "CPC":
                model.learn(total_timesteps=self.params['pretrain_steps'], callback=callbacks, log_interval=1, tb_log_name="SAC Pretrain", d=self.d)
            else:
                model.learn(total_timesteps=self.params['pretrain_steps'], callback=callbacks, log_interval=1, tb_log_name="SAC Pretrain")
        return model

    # finetune the pretrained policy on a specific task
    def finetune(self):
        if self.task:
            # TODO: add other environments
            if self.env_name == "HalfCheetah-v2":
                env = HalfCheetahTaskWrapper(gym.make("HalfCheetah-v2"), task=self.task)
                env = DummyVecEnv([lambda: SkillWrapper(env, self.params['n_skills'], max_steps=self.conf.max_steps)])
        else:
            env = DummyVecEnv([lambda: SkillWrapper(gym.make(
                self.env_name), self.params['n_skills'], max_steps=self.conf.max_steps)])
            
        model_dir = self.directory + "/best_model"
        if self.alg == "sac":
            model = SAC.load(model_dir, env=env, seed=self.seed)
        elif self.alg == "ppo":
            model = PPO.load(model_dir, env=env,  clip_range=get_schedule_fn(
                self.alg_params['clip_range']), seed=self.seed)

        best_skill_index = best_skill(
            model, self.env_name,  self.params['n_skills'])

        del model

        if self.alg == "sac":
            if self.task:
                if self.env_name == "HalfCheetah-v2":
                    env = HalfCheetahTaskWrapper(gym.make("HalfCheetah-v2"), task=self.task)
                    env = DummyVecEnv([lambda: SkillWrapperFinetune(Monitor(env,  f"{self.directory}/finetune_train_results"), self.params['n_skills'], max_steps=gym.make(self.env_name)._max_episode_steps, skill=best_skill_index)])
            else:
                env = DummyVecEnv([lambda: SkillWrapperFinetune(Monitor(gym.make(
        self.env_name),  f"{self.directory}/finetune_train_results"), self.params['n_skills'], max_steps=gym.make(self.env_name)._max_episode_steps, skill=best_skill_index)])
                
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
        elif self.alg == "ppo":
            if self.task:
                if self.env_name == "HalfCheetah-v2":
                    env = HalfCheetahTaskWrapper(gym.make("HalfCheetah-v2"), task=self.task)
                    env = DummyVecEnv([lambda: SkillWrapperFinetune(Monitor(env,  f"{self.directory}/finetune_train_results"), self.params['n_skills'], max_steps=gym.make(self.env_name)._max_episode_steps, skill=best_skill_index), 
                                      lambda: SkillWrapperFinetune(Monitor(env,  f"{self.directory}/finetune_train_results"), self.params['n_skills'], max_steps=gym.make(self.env_name)._max_episode_steps, skill=best_skill_index), 
                                      lambda: SkillWrapperFinetune(Monitor(env,  f"{self.directory}/finetune_train_results"), self.params['n_skills'], max_steps=gym.make(self.env_name)._max_episode_steps, skill=best_skill_index), 
                                       lambda: SkillWrapperFinetune(Monitor(env,  f"{self.directory}/finetune_train_results"), self.params['n_skills'], max_steps=gym.make(self.env_name)._max_episode_steps, skill=best_skill_index)])
            else:
                env = DummyVecEnv([lambda: Monitor(RewardWrapper(SkillWrapper(gym.make(
                    self.env_name), self.params['n_skills'], max_steps=self.conf.max_steps), self.d, self.params['n_skills']),  self.directory)]*self.alg_params['n_actors'])
            
            model = PPO('MlpPolicy', env, verbose=1,
                        learning_rate=self.alg_params['learning_rate'],
                        n_steps=self.alg_params['n_steps'],
                        batch_size=self.alg_params['batch_size'],
                        n_epochs=self.alg_params['n_epochs'],
                        gamma=self.alg_params['gamma'],
                        gae_lambda=self.alg_params['gae_lambda'],
                        clip_range=self.alg_params['clip_range'],
                        policy_kwargs=dict(activation_fn=nn.Tanh,
                                           net_arch=[self.conf.layer_size_shared, self.conf.layer_size_shared,dict(pi=[self.conf.layer_size_policy], vf=[self.conf.layer_size_value])]),
                        tensorboard_log=self.directory,
                        seed=self.seed
                        )
        
        # env = DummyVecEnv([lambda: SkillWrapperFinetune(Monitor(gym.make(
        # self.env_name),  f"{self.directory}/finetune_train_results"), self.params['n_skills'], max_steps=gym.make(self.env_name)._max_episode_steps, skill=best_skill_index)])

        eval_env = SkillWrapperFinetune(gym.make(
            self.env_name), self.params['n_skills'], max_steps=gym.make(self.env_name)._max_episode_steps, skill=best_skill_index)
        eval_env = Monitor(eval_env, f"{self.directory}/finetune_eval_results")
        
        eval_callback = EvalCallback(eval_env, best_model_save_path=self.directory + f"/best_finetuned_model_skillIndex:{best_skill_index}",
                                    log_path=f"{self.directory}/finetune_eval_results", eval_freq=1000,
                                    deterministic=True, render=False)
        if self.alg == "sac":
            model = SAC.load(model_dir, env=env, tensorboard_log=self.directory)
            model.learn(total_timesteps=self.params['finetune_steps'],
                        callback=eval_callback, tb_log_name="SAC_FineTune")
        elif self.alg == "ppo":
            model = PPO.load(model_dir, env=env, tensorboard_log=self.directory,
                            clip_range=get_schedule_fn(self.alg_params['clip_range']))

            model.learn(total_timesteps=self.params['finetune_steps'],
                        callback=eval_callback, tb_log_name="PPO_FineTune")

        # record a video of the agent
        # record_video_finetune(self.env_name, best_skill_index, model,
        #                     self.params['n_skills'], video_length=1000, video_folder='videos_finetune/', alg=self.alg)