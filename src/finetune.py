import argparse
from stable_baselines3.common import callbacks

from stable_baselines3.common.vec_env.dummy_vec_env import DummyVecEnv
from stable_baselines3 import PPO, SAC
from stable_baselines3.common.callbacks import  EvalCallback

from config import conf
from env_wrappers import SkillWrapperFinetune, RewardWrapper, SkillWrapper
from utils import record_video_finetune
from stable_baselines3.common.utils import get_schedule_fn

import numpy as np
import torch
import torch.nn as nn
import gym

# ppo hyperparams
ppo_hyperparams = dict(
    learning_rate = 3e-4,
    gamma = 0.99,
    batch_size = 128,
    n_epochs = 10,
    gae_lambda = 0.95,
    n_steps = 2048,
    clip_range = 0.2,
    ent_coef=0.1,
)

# SAC Hyperparameters 
sac_hyperparams = dict(
    learning_rate = 3e-4,
    gamma = 0.99,
    batch_size = 128,
    buffer_size = int(1e6),
    tau = 0.01,
    gradient_steps = 1,
    ent_coef=0.1,
    learning_starts = 10000
)

def cmd_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--env', type=str, default="MountainCarContinuous-v0")
    parser.add_argument("--alg", type=str, default="ppo")
    args = parser.parse_args()
    return args


def augment_obs(obs, skill, n_skills):
    onehot = np.zeros(n_skills)
    onehot[skill] = 1
    aug_obs = np.array(list(obs) + list(onehot)).astype(np.float32)
    return torch.FloatTensor(aug_obs).unsqueeze(dim=0)


def best_skill(model, env_name):
    env = gym.make(env_name)
    env._max_episode_steps = conf.max_steps
    total_rewards = []
    for skill in range(conf.n_z):
        obs = env.reset()
        aug_obs = augment_obs(obs, skill, conf.n_z)
        reward = 0
        done = False
        while not done:
            action, _ = model.predict(aug_obs, deterministic=True)
            obs, reward, done, _ = env.step(action)
            aug_obs = augment_obs(obs, skill, conf.n_z)
            reward += 1
        total_rewards.append(reward)
    # print(f"Total rewards {total_rewards}")
    # print(f"Best skill: {np.argmax(total_rewards)}")
    return np.argmax(total_rewards)


def finetune(args):
    env = DummyVecEnv([lambda: SkillWrapper(gym.make(args.env), conf.n_z, max_steps=conf.max_steps)])
    model_dir = conf.log_dir + f"{args.alg}_{args.env}" + "/best_model"
    if args.alg == "sac":
        model = SAC.load(model_dir, env=env, )
    elif args.alg == "ppo":
        model = PPO.load(model_dir, env=env,  clip_range= get_schedule_fn(ppo_hyperparams['clip_range']))
    
    best_skill_index = best_skill(model, args.env)
    env_finetune = DummyVecEnv([lambda: SkillWrapperFinetune(gym.make(args.env), conf.n_z, max_steps=conf.max_steps, skill=best_skill_index)])
    
    del model
    if args.alg == "sac":
        model = SAC('MlpPolicy', env, verbose=1, 
                    learning_rate = sac_hyperparams['learning_rate'], 
                    batch_size = sac_hyperparams['batch_size'],
                    gamma = sac_hyperparams['gamma'],
                    buffer_size = sac_hyperparams['buffer_size'],
                    tau = sac_hyperparams['tau'],
                    gradient_steps = sac_hyperparams['gradient_steps'],
                    learning_starts = sac_hyperparams['learning_starts'],
                    policy_kwargs = dict(net_arch=dict(pi=[conf.layer_size_policy, conf.layer_size_policy], qf=[conf.layer_size_value, conf.layer_size_value])),
                    tensorboard_log = conf.log_dir + f"/sac_{args.env}",
                    )
    elif args.alg == "ppo":
        model = PPO('MlpPolicy', env, verbose=1, 
                learning_rate=ppo_hyperparams['learning_rate'], 
                n_steps=ppo_hyperparams['n_steps'], 
                batch_size=ppo_hyperparams['batch_size'],
                n_epochs=ppo_hyperparams['n_epochs'],
                gamma=ppo_hyperparams['gamma'],
                gae_lambda=ppo_hyperparams['gae_lambda'],
                clip_range=ppo_hyperparams['clip_range'],
                policy_kwargs = dict(activation_fn=nn.Tanh,
                                net_arch=[dict(pi=[conf.layer_size_policy, conf.layer_size_policy], vf=[conf.layer_size_value, conf.layer_size_value])]),
                tensorboard_log = conf.log_dir + f"/ppo_{args.env}",

                )

    eval_env = SkillWrapperFinetune(gym.make(args.env), conf.n_z, max_steps=conf.max_steps, skill=best_skill_index)
    eval_callback = EvalCallback(eval_env, best_model_save_path=conf.log_dir_finetune + f"/{args.alg}_{args.env}",
                                log_path=conf.log_dir_finetune + f"{args.alg}_{args.env}", eval_freq=1000,
                                deterministic=True, render=False)
    if args.alg == "sac":
        model = SAC.load(model_dir, env = env_finetune, tensorboard_log = conf.log_dir_finetune + f"sac_{args.env}")
        model.learn(total_timesteps=conf.total_timesteps, callback=eval_callback)
    elif args.alg == "ppo":
        model = PPO.load(model_dir, env = env_finetune, tensorboard_log = conf.log_dir_finetune + f"ppo_{args.env}", clip_range= get_schedule_fn(ppo_hyperparams['clip_range']))
        model.learn(total_timesteps=conf.total_timesteps, callback=eval_callback)
    

    # record a video of the agent
    record_video_finetune(args.env, best_skill_index, model, conf.n_z ,video_length=1000, video_folder='videos_finetune/', alg=args.alg)




if __name__ == "__main__":
    args = cmd_args()
    finetune(args)



    

