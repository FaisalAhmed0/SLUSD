import torch
import torch.nn as nn
import torch.multiprocessing as mp
torch.set_num_threads(1)
torch.multiprocessing.set_sharing_strategy('file_system')


from stable_baselines3.common.vec_env.dummy_vec_env import DummyVecEnv
from stable_baselines3.common.monitor import Monitor
from stable_baselines3 import PPO, SAC

from src.config import conf


import time
import os
import gym

import numpy as np

# list for multip-processing
envs_mp = [
    # 2
    {
     # 'ppo':{
     #    'Reacher-v2': dict( 
     #       steps = int(25e6),
     #        clip_range = 0.01
     #         ), # 2 dof
     #    },
    'sac':{
        'Reacher-v2': dict( 
           steps = int(5e6),
             ), # 2 dof
        }
    },
    # 3
    {
     # 'ppo':{
     #    'Swimmer-v2': dict( 
     #       steps = int(30e6)
     #         ), # 2 dof
     #    },
    'sac':{
        'Swimmer-v2': dict( 
           steps = int(5e6),
             ), # 2 dof
        }
    },
    # 4
    {
     # 'ppo':{
     #    'Hopper-v2': dict( 
     #       steps = int(100e6),
     #         ), # 2 dof
     #    },
    'sac':{
        'Hopper-v2': dict( 
           steps = int(5e6),
             ), # 2 dof
        }
    },
    # 5
    {
     # 'ppo':{
     #    'HalfCheetah-v2': dict( 
     #       steps = int(500e6),
     #         ), # 2 dof
     #    },
    'sac':{
        'HalfCheetah-v2': dict( 
           steps = int(5e6),
             ), # 2 dof
        }
    },
    # 6
    {
     # 'ppo':{
     #    'Walker2d-v2': dict( 
     #       steps = int(550e6),
     #         ), # 2 dof
     #    },
    'sac':{
        'Walker2d-v2': dict( 
           steps = int(5e6),
             ), # 2 dof
        }
    },
    # 7
    {
     # 'ppo':{
     #    'Ant-v2': dict( 
     #       steps = int(700e6),
     #         ), # 2 dof
     #    },
    'sac':{
        'Ant-v2': dict( 
           steps = int(5e6),
             ), # 2 dof
        }
    },
    # 8
    {
     # 'ppo':{
     #    'Humanoid-v2': dict( 
     #       steps = int(800e6),
     #         ), # 2 dof
     #    },
    'sac':{
        'Humanoid-v2': dict( 
           steps = int(5e6),
             ), # 2 dof
        }
    },   
]

# ppo hyperparams
ppo_hyperparams = dict(
    n_steps = 2048,
    learning_rate = 3e-4,
    n_epochs = 10,
    batch_size = 64,
    gamma = 0.99,
    gae_lambda = 0.95,
    clip_range = 0.1,
    ent_coef=0.5,
    n_actors = 5,
    algorithm = "ppo",
    
)

# SAC Hyperparameters 
sac_hyperparams = dict(
    learning_rate = 3e-4,
    gamma = 0.99,
    buffer_size = int(1e7),
    batch_size = 128,
    tau = 0.005,
    gradient_steps = 1,
    ent_coef=0,
    learning_starts = 10000,
    algorithm = "sac"
)

hyperparams = {
    'ppo': ppo_hyperparams,
    'sac': sac_hyperparams,
}

@torch.no_grad()
def evaluate_policy(env_name, model):
    env = gym.make(env_name)
    done = False
    total_reward = 0
    obs = env.reset()
    while not done:
        action, _ = model.predict(obs, deterministic=True)
        obs, reward, done, info = env.step(action)
        total_reward += reward
    return total_reward



def train_model(env_name, alg, directory, alg_params, steps, seed):
    if alg == "ppo":
        env = DummyVecEnv([lambda: Monitor(gym.make(env_name), directory)] * alg_params['n_actors'])
        # create the model with the speicifed hyperparameters
        model = PPO('MlpPolicy', env, verbose=1,
                    learning_rate=alg_params['learning_rate'],
                    n_steps=alg_params['n_steps'],
                    batch_size=alg_params['batch_size'],
                    n_epochs=alg_params['n_epochs'],
                    gamma=alg_params['gamma'],
                    gae_lambda=alg_params['gae_lambda'],
                    clip_range=alg_params['clip_range'],
                    policy_kwargs=dict(activation_fn=nn.Tanh,
                                       net_arch=[conf.layer_size_shared, conf.layer_size_shared,dict(pi=[conf.layer_size_policy], vf=[conf.layer_size_value])]),
                    tensorboard_log=directory,
                    seed=seed
                    )
        model.learn(total_timesteps=steps, tb_log_name="PPO Asymptotic")
        return model
    elif alg == "sac":
        env = DummyVecEnv([lambda: Monitor(gym.make(env_name), directory)])
        
        # create the model with the speicifed hyperparameters
        model = SAC('MlpPolicy', env, verbose=1,
                    learning_rate=alg_params['learning_rate'],
                    batch_size=alg_params['batch_size'],
                    gamma=alg_params['gamma'],
                    buffer_size=alg_params['buffer_size'],
                    tau=alg_params['tau'],
                    ent_coef=alg_params['ent_coef'],
                    gradient_steps=alg_params['gradient_steps'],
                    learning_starts=alg_params['learning_starts'],
                    policy_kwargs=dict(net_arch=dict(pi=[2*conf.layer_size_policy, 2*conf.layer_size_policy], qf=[
                                       conf.layer_size_q, conf.layer_size_q])),
                    tensorboard_log=directory,
                    seed=seed
                    )
        model.learn(total_timesteps=steps, tb_log_name="SAC Asymptotic")
        return model
    elif alg == "pets":
        pass
        
        

def train_all(env_params, results_list):
    main_exper_dir = conf.log_dir_finetune + f"asymptotics/"
    for alg in env_params:
        for env_name in env_params[alg]:
            # train_model(env_name, alg, directory, alg_params, steps, seed)
            # save a timestamp
            timestamp = time.time()
            directory = main_exper_dir + f"env: {env_name}, alg:{alg}, stamp:{timestamp}/"
            os.makedirs(directory, exist_ok=True)
            alg_params = hyperparams[alg]
            if (env_params[alg][env_name]).get('clip_range'):
                # clip_range
                alg_params['clip_range'] = (env_params[alg][env_name]).get('clip_range')
            seed_results = []
            steps = env_params[alg][env_name]['steps']
            for seed in conf.seeds:
                trained_policy = train_model(env_name, alg, directory, alg_params, steps, seed)
                seed_results.append( np.mean([evaluate_policy(env_name, trained_policy) for _ in range(10)]) )
            final_reward_mean = np.mean(seed_results, axis=0)
            final_reward_std = np.std(seed_results, axis=0)
        results_list.append([alg, env_name, final_reward_mean, final_reward_std])
        
        
if __name__ == "__main__":
    # for mp
    manager = mp.Manager()
    results_df_list = manager.list()
    n_processes = len(envs_mp)
    processes_list = []
    for i in range(n_processes):
        p = mp.Process(target=train_all, args=(envs_mp[i], results_df_list))
        p.start()
        processes_list.append(p)
    for p in processes_list:
        p.join()
    with open(f"{conf.log_dir_finetune}asymptotics/asymptotic_results.txt", "w") as results:
        results.write("Asymptotic Performance\n")
        results.write("(Algorithm, Environment, Mean Reward, Std)\n")
        for row in results_df_list:
            results.write(f"{row}\n")
        
            
            
            
        
    
    
    
    