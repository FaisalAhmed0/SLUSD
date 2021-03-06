# import argparse
# import ast
# import math

# from stable_baselines3.common.vec_env.dummy_vec_env import DummyVecEnv
# from stable_baselines3 import PPO, SAC
# from stable_baselines3.common.utils import get_schedule_fn

# import os
# from src.config import conf
# from src.utils import seed_everything, evaluate_pretrained_policy_ext, evaluate_adapted_policy, evaluate_state_coverage, evaluate_pretrained_policy_intr, evaluate
# from src.environment_wrappers.env_wrappers import SkillWrapperFinetune, RewardWrapper, SkillWrapper, TimestepsWrapper
# from src.models.models import Discriminator

# import numpy as np
# import matplotlib.pyplot as plt
# import seaborn as sns
# import pandas as pd
# sns.set()
# sns.set_style('whitegrid')
# sns.set_context("paper", font_scale = conf.font_scale)

# import torch
# import torch.nn as nn

# import gym


# # Asymptotic performance of the expert
# asymp_perofrmance = {
#     'HalfCheetah-v2': 8070.205213363435,
#     'Walker2d-v2': 5966.481684037183,
#     'Ant-v2': 6898.620003217143,
#     "MountainCarContinuous-v0": 95.0
# <<<<<<< scalability

#     }

# random_performance = {
#     'HalfCheetah-v2': -0.37117783,
#     'Walker2d-v2': -0.48223244,
#     'Ant-v2': 0.39107815,
#     "MountainCarContinuous-v0": -0.04685473, 
# }

# random_initlized_performance = {
#     'HalfCheetah-v2': np.mean([3.977155901244182132e+03, 3.661330217032705150e+03, 3.777719310452177979e+03]),
#     'Walker2d-v2': np.mean([4.950961607268399121e+02, 5.050660942998074461e+02, 2.915505078088318669e+02]),
#     'Ant-v2': np.mean([4.935468207680875139e+02, 2.522865492915514665e+02, 7.162829405419695377e+02]),
#     "MountainCarContinuous-v0": np.mean([-1.697574145149261080e-04, -3.889756080078754464e-04, -3.093423134238547888e-04]),
# }



# seeds = [0, 10, 42] # 10, 42 ]#, 1234, 5]#, 42]
# # cmd arguments
# # cmd arguments
# def cmd_args():
#     parser = argparse.ArgumentParser()
#     parser.add_argument('--env', type=str, default="MountainCarContinuous-v0")
#     parser.add_argument("--algs", nargs="*", type=str, default=["ppo"]) # pass the algorithms that correspond to each timestamp
#     parser.add_argument("--skills", nargs="*", type=int, default=[6])
#     parser.add_argument("--stamps", nargs="*", type=str, required=False) # pass the timestamps as a list
#     parser.add_argument("--colors", nargs="*", type=str, required=False, default=['b', 'r']) # pass the timestamps as a list
#     parser.add_argument("--labels", nargs="*", type=str, required=False, default=None) # add custom labels 
#     parser.add_argument("--cls", nargs="*", type=str, required=False) # pass the timestamps as a list
#     parser.add_argument("--lbs", nargs="*", type=str, required=False) # pass the timestamps as a list
#     parser.add_argument("--suffix", type=str, required=False, default=None) # add custom labels 
#     args = parser.parse_args()
#     return args

# def scalability_plots(env_name, alg, pm, lb, stamp, color, label, ax1, ax2):
#     '''
#     axes.plot(range(len(data_mean)), data_mean, label=legend, color=plot_color)
# 	axes.fill_between(range(len(data_mean)), (data_mean-data_std), (data_mean+data_std), color=plot_color, alpha=0.3)
# 	axes.set_xlabel('Iteration')
# 	axes.set_ylabel(ylabel)
# 	axes.grid(True)
# 	axes.set_title(f'{env_name}')
# 	axes.legend()
#     '''
#     main_exper_dir = conf.scalability_exper_dir + f"cls:{pm}, lb:{lb}/"
#     env_dir = main_exper_dir + f"env: {env_name}, alg:{alg}, stamp:{stamp}/"
#     seeds_list = []
#     seed_list_intr_reward = []
#     for seed in seeds:
#         seed_dir = env_dir + f"seed:{seed}/"
#         files = np.load(seed_dir + "scalability_evaluations.npz")
#         # timesteps=timesteps, intr_rewards=intr_rewards, extr_rewards=extr_rewards
#         print(files.files)
#         intr_rewards = files['intr_rewards']
#         extr_rewards = files['extr_rewards']
#         seeds_list.append(extr_rewards)
#         seed_list_intr_reward.append(intr_rewards)
        
#     timesteps = files['timesteps']
#     data_mean = np.mean(seeds_list, axis=0)/asym * 100
#     data_std = np.std(seeds_list, axis=0)/asym * 100
    
    
#     ax1.set_xscale('log')
#     # ax1.set_yscale('log')
#     ax1.set_xlabel("Pre-training Timesteps")
#     ax1.set_ylabel("Normalized Extrinsic Return (Best Skill) (%)")
    
#     ax1.plot(timesteps, data_mean, label=label, color=color)
#     ax1.fill_between(timesteps, (data_mean-data_std), (data_mean+data_std), color=color, alpha=0.3)
    
#     ax2.set_xscale('log')
#     ax2.set_xlabel("Intrinsic Reward")
#     ax2.set_ylabel("Normalized Extrinsic Return (Best Skill) (%)")
    
#     data_mean_x = np.mean(seed_list_intr_reward, axis=0)
#     data_std_x = np.std(seed_list_intr_reward, axis=0)
#     ax2.scatter(data_mean_x, data_mean, label=label, color=color)
#     # sns.regplot(x=data_mean_x, y=data_mean, ci=68, truncate=False)
#     return ax1, ax2
    
    
#     # ax2.errorbar(data_mean_x, data_mean, yerr=data_std, xerr=data_std_x ,ecolor=color, fmt=' ', zorder=-1)
    
# if __name__ == "__main__":
#     # args = cmd_args()
#     # algs = args.algs
#     # stamps = args.stamps
#     # colors = args.colors
#     xlabel = "Timesteps"
#     ylabel = "Reward"
#     # legends = args.labels if args.labels else algs
#     # env_name = args.env
#     env_names = ['MountainCarContinuous-v0', 'HalfCheetah-v2', 'Walker2d-v2', 'Ant-v2' ]
#     algs = ['sac', 'ppo']
#     colors = ['b', 'r']
#     stamps = [ [1645717501.9848473, 1645717501.971655], [1645717502.0205646, 1645717501.9979885], [1645717502.0579038, 1645717502.0401757], [1645717502.0999389, 1645717502.0758302]  ]
#     legends = algs
#     pm = "MLP"
#     lb = "ba"
#     # This is for testing comment it when the test pass
#     # algs = ['ppo']
#     # colors = ['r']
#     # stamps = ["1645207392.8617382"]
#     # legends = algs
#     for i in range(len(env_names)):
#         env_name = env_names[i]
#         asym = asymp_perofrmance[env_name]
#         # set_title
#         fig1 = plt.figure(figsize=conf.learning_curve_figsize)
#         ax1 = fig1.add_subplot(111)
#         ax1.set_title(f"{env_name[: env_name.index('-')]}")
#         r = random_performance[env_name]
#         ri = random_initlized_performance[env_name]

#         ax1.axhline(r/asym * 100, label="Random Policy",  c='black', linestyle='dashed')
#         ax1.axhline(ri/asym * 100, label="RL from Scratch (100k steps)",  c='black', linestyle='solid')
#         ax1.axhline(100, label="Expert Policy",  c='black', linestyle='dotted')
#         fig2 = plt.figure(figsize=conf.learning_curve_figsize)
#         ax2 = fig2.add_subplot(111)
#         ax2.set_title(f"{env_name[: env_name.index('-')]}")
#         ax2.axhline(r/asym * 100, label="Random Policy",  c='black', linestyle='dashed')
#         ax2.axhline(100, label="Expert Policy",  c='black', linestyle='dotted')
#         ax2.axhline(ri/asym * 100, label="RL from Scratch (100k steps)",  c='black', linestyle='solid')
    
#         for j in range(len(algs)):
#             alg = algs[j]
#             stamp = stamps[i][j]
#             # print(env_name, alg, stamp)
#             color = colors[j]
#             ax1, ax2 = scalability_plots(env_name, alg, pm, lb, stamp, color, alg.upper(), ax1, ax2)
#             ax1.legend()
#             ax2.legend()
#             ax1.legend_.remove()
#             ax2.legend_.remove()
#             files_dir = f"Vis/{env_name}_cls:{pm}_lb:{lb}"
#             os.makedirs(files_dir, exist_ok=True)
#             filename = f"{env_name}_scalability_experiment_pretraining_steps.png"
#             fig1.savefig(f'{files_dir}/{filename}')  
#             filename = f"{env_name}_scalability_experiment_intrinsic_reward.png"
#             fig2.savefig(f'{files_dir}/{filename}')  
#     # ax1.legend_.remove()
#     # ax2.legend_.remove()
#     figLegend = plt.figure(figsize = (9.2,0.5))
#     plt.figlegend(*ax1.get_legend_handles_labels(), loc = 'upper left', frameon=False, ncol=5, bbox_to_anchor=(0, 1), fontsize='small')
#     plt.savefig(f"{files_dir}/scalability_experiment_learning_curve_legend.png")
            
            
#     print("Done")

            
            
            
    
#     # file_dir = conf.log_dir_finetune + f"{args.alg}_{args.env}_{args.stamp}/" + "eval_results/" + "evaluations.npz"
#     # file_dir = conf.log_dir_finetune + f"{args.alg}_{args.env}_{args.stamp}/" + "evaluations.npz"
# #     print("######## Skills evaluation ########")
# #     plt.figure()
# #     plt.title(f"{env_name[: env_name.index('-')]}")
# #     for i in range(len(algs)):
# #         skills = args.skills[i]
# #         pm = args.cls[i]
# #         lb = args.lbs[i]
# #         alg = algs[i]
# #         print(f"Algorithm: {alg}")
# #         plot_color = colors[i]
# #         stamp = stamps[i]
# #         # legend = alg
# #         main_exper_dir = scalability_exper_dir + f"cls:{pm}, lb:{lb}/"
# #         env_dir = main_exper_dir + f"env: {env_name}, alg:{alg}, stamp:{stamp}/"
# #         seeds_list = []
# #         for seed in seeds:
# #             seed_everything(seed)
# #             seed_dir = env_dir + f"seed:{seed}/"
# #             file_dir_skills = seed_dir + "eval_results/" + "evaluations.npz"
# #             files = np.load(file_dir_skills)
# #             steps = files['timesteps']
# #             results = files['results']
# #             print(f"results.shape: {results.shape}")
# #             seeds_list.append(results.mean(axis=1).reshape(-1))
# #             # data_mean = results.mean(axis=1)
# #             # data_std = results.std(axis=1)
# #             # print(f"mean: {data_mean[-1]}")
# #             # print(f"std: {data_std[-1]}")
# #         data_mean = np.mean(seeds_list, axis=0)
# #         print(f"seeds list shape {np.array(seeds_list).shape}")
# #         # input()
# #         data_std = np.std(seeds_list, axis=0)
# #         # data_mean = np.convolve(data_mean, np.ones(10)/10, mode='valid') 
# #         # data_std = np.convolve(data_std, np.ones(10)/10, mode='valid') 
# #         # steps = np.convolve(steps, np.ones(10)/10, mode='valid') 
# #         print(f"data shape: {data_mean.shape}")
# #         print(f"steps shape: {steps.shape}")
# #         # input()
# #         if len(algs) > 1:
# #             plt.plot(steps, data_mean, label=legends[i], color=plot_color)
# #             plt.legend()
# #         else:
# #             plt.plot(steps, data_mean, color=plot_color)
# #         plt.fill_between(steps, (data_mean-data_std), (data_mean+data_std), color=plot_color, alpha=0.3)
# #         plt.xlabel(xlabel)
# #         plt.ylabel(ylabel)
# #         # plt.grid(False)
# #         if len(algs) == 1:
# #             filename = f"{args.env}_pretraining_{alg}"
# #             files_dir = f"Vis/{args.env}"
# #             os.makedirs(files_dir, exist_ok=True)
# #             plt.savefig(f'{files_dir}/{filename}', dpi=150)    
# #     if len(algs) > 1:
# #         filename = f"{args.env}_pretraining_all_algs {args.suffix}"
# #         files_dir = f"Vis/{args.env}_cls:{pm}_lb:{lb}"
# #         os.makedirs(files_dir, exist_ok=True)
# #         plt.savefig(f'{files_dir}/{filename}', dpi=150)  
# =======

#     }

# random_performance = {
#     'HalfCheetah-v2': -0.37117783,
#     'Walker2d-v2': -0.48223244,
#     'Ant-v2': 0.39107815,
#     "MountainCarContinuous-v0": -0.04685473, 
# }
# # across all seeds
# random_init = [
#     {"Environment": 'HalfCheetah', "Normalized Extrinsic Return After Adaptation (%)": 3.977155901244182132e+03/asymp_perofrmance['HalfCheetah-v2']*100, "Algorithm": "RL from Scratch (100k steps)"},
#     {"Environment": 'HalfCheetah', "Normalized Extrinsic Return After Adaptation (%)": 3.661330217032705150e+03/asymp_perofrmance['HalfCheetah-v2']*100, "Algorithm": "RL from Scratch (100k steps)"},
#     {"Environment": 'HalfCheetah', "Normalized Extrinsic Return After Adaptation (%)": 3.777719310452177979e+03/asymp_perofrmance['HalfCheetah-v2']*100, "Algorithm": "RL from Scratch (100k steps)"},
    
#     {"Environment": 'Walker2d', "Normalized Extrinsic Return After Adaptation (%)": 4.950961607268399121e+02/asymp_perofrmance['Walker2d-v2']*100, "Algorithm": "RL from Scratch (100k steps)"},
#     {"Environment": 'Walker2d', "Normalized Extrinsic Return After Adaptation (%)": 5.050660942998074461e+02/asymp_perofrmance['Walker2d-v2']*100, "Algorithm": "RL from Scratch (100k steps)"},
#     {"Environment": 'Walker2d', "Normalized Extrinsic Return After Adaptation (%)": 2.915505078088318669e+02/asymp_perofrmance['Walker2d-v2']*100, "Algorithm": "RL from Scratch (100k steps)"},
    
#     {"Environment": 'Ant', "Normalized Extrinsic Return After Adaptation (%)": 4.935468207680875139e+02/asymp_perofrmance['Ant-v2']*100, "Algorithm": "RL from Scratch (100k steps)"},
#     {"Environment": 'Ant', "Normalized Extrinsic Return After Adaptation (%)": 2.522865492915514665e+02/asymp_perofrmance['Ant-v2']*100, "Algorithm": "RL from Scratch (100k steps)"},
#     {"Environment": 'Ant', "Normalized Extrinsic Return After Adaptation (%)": 7.162829405419695377e+02/asymp_perofrmance['Ant-v2']*100, "Algorithm": "RL from Scratch (100k steps)"},
    
#     {"Environment": 'MountainCarContinuous', "Normalized Extrinsic Return After Adaptation (%)": -1.697574145149261080e-04/asymp_perofrmance['MountainCarContinuous-v0']*100, "Algorithm": "RL from Scratch (100k steps)"},
#     {"Environment": 'MountainCarContinuous', "Normalized Extrinsic Return After Adaptation (%)": -3.889756080078754464e-04/asymp_perofrmance['MountainCarContinuous-v0']*100, "Algorithm": "RL from Scratch (100k steps)"},
#     {"Environment": 'MountainCarContinuous', "Normalized Extrinsic Return After Adaptation (%)": -3.093423134238547888e-04/asymp_perofrmance['MountainCarContinuous-v0']*100, "Algorithm": "RL from Scratch (100k steps)"},
    
# ]

# # random seeds
# seeds = conf.seeds
# # Columns of the result dataframe
# columns = ['Algorithm', "Environment", "Normalized Extrinsic Return After Adaptation (%)", "State Entropy", "Intrinsic Return" , "Normalized Extrinsic Return Before Adaptation (Best Skill) (%)"]

# def extract_results(env_name, n_skills, pm, alg, stamp, lb):
#     '''
#     Extract the results saved during the training process.
#     '''
#     seeds_list = []
#     asym = asymp_perofrmance[env_name]
    
#     # experiment directory
#     main_exper_dir = conf.log_dir_finetune + f"cls:{pm}, lb:{lb}/"
#     # environment directory
#     env_dir = main_exper_dir + f"env: {env_name}, alg:{alg}, stamp:{stamp}/"
    
#     params = pd.read_csv(env_dir + "params.csv").to_dict()
#     discriminator_hyperparams_df = pd.read_csv(env_dir + "discriminator_hyperparams.csv")
#     # print(discriminator_hyperparams_df.replace(math.nan, 12))
#     discriminator_hyperparams = discriminator_hyperparams_df.to_dict('records')[0]
#     # print(discriminator_hyperparams)
#     state_dim = gym.make(env_name).observation_space.shape[0]
#     hidden_dims = conf.num_layers_discriminator*[conf.layer_size_discriminator]
#     temperature = discriminator_hyperparams['temperature']
#     dropout = discriminator_hyperparams['dropout'] if 0 < discriminator_hyperparams['dropout'] < 1 else None
#     # print(f"dropout type: {type(dropout)}")
#     # print(dropout)
#     # input()
#     results_list = []
#     # (self, env, discriminator, n_skills, parametrization="MLP", batch_size=128)
#     for seed in seeds:
#         seed_dir = env_dir + f"seed:{seed}/"
#         seed_everything(seed)
#         discriminator = Discriminator(state_dim, hidden_dims, n_skills, dropout=dropout).to(conf.device)
#         # print(discriminator)
#         # print(torch.load(seed_dir + "disc.pth"))
#         discriminator.load_state_dict(torch.load(seed_dir + "disc.pth"))
        
#         env = DummyVecEnv([lambda: RewardWrapper(SkillWrapper(gym.make(env_name), n_skills, max_steps=1000), discriminator, n_skills, pm)])
#         if alg == "sac":
#             model_dir = seed_dir + "/best_model"
#             pretrained_policy = SAC.load(model_dir, env=env, seed=seed)
#             best_skill = int([ d[d.index(":")+1:] for d in  os.listdir(seed_dir) if "best_finetuned_model_skillIndex:" in d][0])
#             print(best_skill)
#             finetuned_model_dir =  seed_dir + f"best_finetuned_model_skillIndex:{best_skill}/" + "/best_model"
#             adapted_policy = SAC.load(finetuned_model_dir, env=env, seed=seed)
#         elif alg == "ppo":
#             model_dir = seed_dir + "/best_model"
#             hyper_params = pd.read_csv(env_dir + "ppo_hyperparams.csv").to_dict('records')[0]
#             pretrained_policy = PPO.load(model_dir, env=env, seed=seed, clip_range= get_schedule_fn(hyper_params['clip_range']))
            
#             best_skill = int([ d[d.index(":")+1:] for d in  os.listdir(seed_dir) if "best_finetuned_model_skillIndex:" in d][0])
#             # print(best_skill)
#             finetuned_model_dir =  seed_dir + f"best_finetuned_model_skillIndex:{best_skill}/" + "/best_model"
#             adapted_policy = SAC.load(finetuned_model_dir, env=env, seed=seed)
        
#         intrinsic_reward_mean, reward_beforeFinetune_mean, reward_mean, entropy_mean = evaluate(env_name, n_skills, pretrained_policy, adapted_policy, discriminator, pm, best_skill, alg)
#         # intrinsic_reward_mean= evaluate(env_name, n_skills, pretrained_policy, adapted_policy, discriminator, pm, best_skill, alg)
#         # continue
#         # for testing 
#         # intrinsic_reward_mean, reward_beforeFinetune_mean, reward_mean, entropy_mean = (100 + np.random.randn()*0.25, 100 + np.random.randn()*0.25, 100 + np.random.randn()*0.25, 100 + np.random.randn()*0.25)
#         d = {'Algorithm': alg.upper(), 
#                      "Normalized Extrinsic Return After Adaptation (%)": reward_mean/asym*100, 
#                      "Intrinsic Return": intrinsic_reward_mean, 
#                      "State Entropy": entropy_mean, 
#                      "Normalized Extrinsic Return Before Adaptation (Best Skill) (%)": reward_beforeFinetune_mean/asym*100,
#                      "Environment": env_name[: env_name.index('-')] if env != "MountainCarContinuous-v0" else "MountainCar"
#                     }
#         results_list.append(d)
        
#     return results_list
    
# def barplot(env_names, n_skills_list, lbs, pms, algs, stamps, colors):
#     results = []
#     for i in range(len(env_names)):
#         env_name = env_names[i]
#         n_skills = n_skills_list[i]
#         pm = pms[i]
#         lb = lbs[i]
#         stamp = stamps[i]
#         alg = algs[i]
#         main_exper_dir = conf.log_dir_finetune + f"cls:{pm}, lb:{lb}/"
#         env_dir = main_exper_dir + f"env: {env_name}, alg:{alg}, stamp:{stamp}/"
#         discriminator_hyperparams = pd.read_csv(env_dir + "discriminator_hyperparams.csv").to_dict('records')[0]
#         # (env_name, n_skills, pm, best_skill, alg, stamp, lb)
#         r = extract_results(env_name, n_skills, pm, alg, stamp, lb)
#         results.extend(r)
#     results_df = pd.DataFrame(results)
#     files_dir = f"Vis/adaptation_experiment_cls:{discriminator_hyperparams['parametrization']}, lb:{discriminator_hyperparams['lower_bound']}/"
#     results_df.to_csv(files_dir + "result.csv")
#     for col in columns[2:]:
#         plt.figure(figsize=conf.barplot_figsize)
#         y_axis = col
#         # plt.title(y_axis)
#         # ax = sns.barplot(x="Algorithm", y=y_axis, data=results_df, hue="Environment")
#         if col == "Normalized Extrinsic Return After Adaptation (%)":
#             print(algs)
#             results_copy = results_df.copy()
#             envs = set(results_df["Environment"].to_list())
#             for e in envs:
#                 print(e)
#                 d = [ r for r in random_init if r['Environment'] == e  ]
#                 print(d)
#                 results_copy = results_copy.append(d)
#             print(results_copy)
#             results_copy.to_csv("Adaptation_experiment_final_results.csv")
#             # input()
#             ax = sns.barplot(x="Environment", y=y_axis, data=results_copy, hue="Algorithm", hue_order=[a.upper() for a in ["sac", "ppo"]] + ["RL from Scratch (100k steps)"], palette=colors)
#             ax.legend_.remove()
#         else:
#             ax = sns.barplot(x="Environment", y=y_axis, data=results_df, hue="Algorithm", hue_order=[a.upper() for a in ["sac", "ppo"]] + ["RL from Scratch (100k steps)"], palette=colors)
#             ax.legend_.remove()
#         filename = f"adaptation_experiment_final_{y_axis}.png"
#         files_dir = f"Vis/adaptation_experiment_cls:{discriminator_hyperparams['parametrization']}, lb:{discriminator_hyperparams['lower_bound']}/"
#         os.makedirs(files_dir, exist_ok=True)
#         plt.savefig(files_dir + filename)
#         figLegend = plt.figure(figsize = (8.2,0.5))
#         plt.figlegend(*ax.get_legend_handles_labels(), loc = 'upper left', frameon=False, ncol=3, bbox_to_anchor=(0, 1), fontsize='small')
#         plt.savefig(f"{files_dir}/adaptation_experiment_barplot_legend.png")
        
    
    
    


# if __name__ == "__main__":
#     # args = cmd_args()
#     # file_dir = conf.log_dir_finetune + f"{args.alg}_{args.env}_{args.stamp}/" + "eval_results/" + "evaluations.npz"
#     # file_dir = conf.log_dir_finetune + f"{args.alg}_{args.env}_{args.stamp}/" + "evaluations.npz"
#     # print("######## BarPlot ########")
#     env_names = ["MountainCarContinuous-v0", "MountainCarContinuous-v0", "HalfCheetah-v2", "HalfCheetah-v2", "Walker2d-v2", "Walker2d-v2", "Ant-v2", "Ant-v2"]
#     # env_names = ["Walker2d-v2", "Walker2d-v2"]
#     skills_l = [10, 10, 30, 30, 30, 30, 30, 30]
#     # skills_l = [30, 30]
#     algs = ["sac", "ppo", "sac", "ppo", "sac", "ppo", "sac", "ppo"]
#     # MC MC HC HC W-SAC, W-PPO, Ant-SAC, Ant-PPO
#     stamps = [1645692889.4627287, 1645692889.4497814, 1645692889.5071442, 1645692889.484781, 1646209337.3552468, 1645692889.5303376, 1645692889.6045046, 1645692889.5788283]
#     pms = ["MLP"]*8
#     lbs = ["ba"]*8
#     colors = ["b", "r", 'g']
#     # barplot(env_names, skills_l, lbs, pms, algs, stamps, colors)
#     print("######## Skills evaluation ########")
#     env_names = ["MountainCarContinuous-v0", "HalfCheetah-v2", "Walker2d-v2", "Ant-v2"]
#     stamps_l = [[1645692889.4627287, 1645692889.4497814], [1645692889.5071442, 1645692889.484781], [1646209337.3552468, 1645692889.5303376], [1645692889.6045046, 1645692889.5788283]]
#     skill_l = [[10, 10], [30, 30], [30, 30], [30, 30]]
#     for j in range(len(env_names)):
#         env_name = env_names[j]
#         skills_l = skill_l[j]
#         algs = ["sac", "ppo"]
#         stamps = stamps_l[j]
#         cls = ["MLP", "MLP"]
#         lbs = ["ba", "ba"]
#         colors = ["b", "r"]
#         legends = [a.upper() for a in algs]
#         plt.figure()
#         title = "MountainCar" if env_name == "MountainCarContinuous-v0" else env_name[: env_name.index('-')]
#         plt.title(title)
#         xlabel = "Environment Timesteps"
#         ylabel = "Intrinsic Return"
#         asym = asymp_perofrmance[env_name]
#         for i in range(len(algs)):
#             skills = skills_l[i]
#             pm = cls[i]
#             lb = lbs[i]
#             alg = algs[i]
#             print(f"Algorithm: {alg}")
#             plot_color = colors[i]
#             stamp = stamps[i]
#             # legend = alg
#             main_exper_dir = conf.log_dir_finetune + f"cls:{pm}, lb:{lb}/"
#             env_dir = main_exper_dir + f"env: {env_name}, alg:{alg}, stamp:{stamp}/"
#             seeds_list = []
#             for seed in seeds:
#                 seed_everything(seed)
#                 seed_dir = env_dir + f"seed:{seed}/"
#                 file_dir_skills = seed_dir + "eval_results/" + "evaluations.npz"
#                 files = np.load(file_dir_skills)
#                 steps = files['timesteps']
#                 results = files['results']
#                 print(f"results.shape: {results.shape}")
#                 seeds_list.append(results.mean(axis=1).reshape(-1))
#                 # data_mean = results.mean(axis=1)
#                 # data_std = results.std(axis=1)
#                 # print(f"mean: {data_mean[-1]}")
#                 # print(f"std: {data_std[-1]}")
#             data_mean = np.mean(seeds_list, axis=0)
#             print(f"seeds list shape {np.array(seeds_list).shape}")
#             # input()
#             data_std = np.std(seeds_list, axis=0)
#             # data_mean = np.convolve(data_mean, np.ones(10)/10, mode='valid') 
#             # data_std = np.convolve(data_std, np.ones(10)/10, mode='valid') 
#             # steps = np.convolve(steps, np.ones(10)/10, mode='valid') 
#             print(f"data shape: {data_mean.shape}")
#             print(f"steps shape: {steps.shape}")
#             # input()
#             if len(algs) > 1:
#                 sns.lineplot(x=steps, y=data_mean, label=legends[i], color=plot_color)
#                 plt.legend()
#             else:
#                 sns.lineplot(x=steps, y=data_mean, color=plot_color)
#             plt.fill_between(steps, (data_mean-data_std), (data_mean+data_std), color=plot_color, alpha=0.3)
#             plt.xlabel(xlabel)
#             plt.ylabel(ylabel)
#             # plt.grid(False)
#             if len(algs) == 1:
#                 filename = f"{env_name}_pretraining_{alg}"
#                 files_dir = f"Vis/{env_name}"
#                 os.makedirs(files_dir, exist_ok=True)
#                 plt.savefig(f'{files_dir}/{filename}')    
#         if len(algs) > 1:
#             filename = f"{env_name}_pretraining_all_algs"
#             files_dir = f"Vis/{env_name}_cls:{pm}_lb:{lb}"
#             os.makedirs(files_dir, exist_ok=True)
#             plt.savefig(f'{files_dir}/{filename}')    

#         # loop throgh the finetuning results
#         # loop throgh the finetuning results
#     print("######## Finetine evaluation ########")
#     ylabel = "Normalized Extrinsic Return (%)"
#     for j in range(len(env_names)):
#         env_name = env_names[j]
#         skills_l = skill_l[j]
#         stamps = stamps_l[j]
#         plt.figure(figsize = conf.learning_curve_figsize)
#         title = "MountainCar" if env_name == "MountainCarContinuous-v0" else env_name[: env_name.index('-')]
#         plt.title(title)
#         baseline = False
#         asym = asymp_perofrmance[env_name]
#         for i in range(len(algs)):
#             skills = skills_l[i]
#             pm = cls[i]
#             lb = lbs[i]
#             alg = algs[i]
#             print(f"Algorithm: {alg}")
#             plot_color = colors[i]
#             stamp = stamps[i]
#             legend = alg
#             main_exper_dir = conf.log_dir_finetune + f"cls:{pm}, lb:{lb}/"
#             main_baseline_dir = "random_init_baselines/"
#             env_dir = main_exper_dir + f"env: {env_name}, alg:{alg}, stamp:{stamp}/"
#             discriminator_hyperparams = pd.read_csv(env_dir + "discriminator_hyperparams.csv").to_dict('records')[0]
#             seeds_list = []
#             seed_list_b = []
#             for seed in seeds:
#                 seed_everything(seed)
#                 seed_dir = env_dir + f"seed:{seed}/"
#                 file_dir_finetune = seed_dir + "finetune_eval_results/" + "evaluations.npz"
#                 if not baseline:
#                     baseline_values = pd.read_csv(f"run-{env_name}_seed_{seed}_SAC baseline_1-tag-eval_mean_reward.csv")['Value'].to_numpy()
#                     steps_b = pd.read_csv(f"run-{env_name}_seed_{seed}_SAC baseline_1-tag-eval_mean_reward.csv")['Step'].to_numpy()
#                     seed_list_b.append(baseline_values)
#                 files = np.load(file_dir_finetune)
#                 steps = files['timesteps']
#                 results = files['results']
#                 # if env_name == "Walker2d-v2" and alg=="sac":
#                 #     print(results.shape)
#                 #     input()
#                 #     print(stamp, alg)
#                 #     print(results[:5], results[-5:])
#                 #     sns.lineplot(x=steps, y=results.mean(axis=1).reshape(-1), label=legends[i], color=plot_color)
#                 #     plt.fill_between(steps, (data_mean-data_std), (data_mean+data_std), color=plot_color, alpha=0.2)
#                     # if env_name == "Walker2d-v2" and alg=="sac":
#                     #     plt.savefig("sac.png")
#                     #     input()
#                 seeds_list.append(results.mean(axis=1).reshape(-1))
                
#             data_mean = np.mean(seeds_list, axis=0)/asym*100
#             data_std = np.std(seeds_list, axis=0)/asym*100
            
#             if not baseline:
#                 data_mean_b = np.mean(seed_list_b, axis=0)/asym * 100
#                 data_std_b = np.std(seed_list_b, axis=0)/asym * 100
#                 sns.lineplot(x=steps, y=data_mean, label=legends[i], color=plot_color)
#                 plt.fill_between(steps, (data_mean-data_std), (data_mean+data_std), color=plot_color, alpha=0.2)
#                 # if env_name == "Walker2d-v2" and alg=="sac":
#                 #     plt.savefig("sac.png")
#                 ax = sns.lineplot(x=steps_b, y=data_mean_b, label="RL from Scratch (100k steps)", color='g')
#                 plt.fill_between(steps_b, (data_mean_b-data_std_b), (data_mean_b+data_std_b), color='g', alpha=0.2)
#                 baseline = True
#             else:
#                 ax = sns.lineplot(x=steps, y=data_mean, label=legends[i], color=plot_color)
#                 plt.fill_between(steps, (data_mean-data_std), (data_mean+data_std), color=plot_color, alpha=0.2)
#             plt.xlabel(xlabel)
#             plt.ylabel(ylabel)
#             # plt.grid(False)
#             r = random_performance[env_name]
#             # if len(algs) == 1:
#             #     ax.axhline(r/asym * 100, label="Random Policy",  c='black', linestyle='dashed')
#             #     ax.axhline(100, label="Expert Policy",  c='black', linestyle='dotted')
#             #     plt.legend()
#             #     filename = f"{env_name}_finetune_{alg}"
#             #     files_dir = f"Vis/{env_name}_cls:{pm}_lb:{lb}"
#             #     os.makedirs(files_dir, exist_ok=True)
#             #     plt.savefig(f'{files_dir}/{filename}')    
#         if len(algs) > 1:
#             ax.axhline(r/asym * 100, label="Random Policy",  c='black', linestyle='dashed')
#             ax.axhline(100, label="Expert Policy",  c='black', linestyle='dotted')
#             plt.legend(loc = "upper left")
#             # if env_name != "MountainCarContinuous-v0":
#             ax.legend_.remove()
#             filename = f"{env_name}_finetune_all_algs"
#             files_dir =  f"Vis/adaptation_experiment_cls:{discriminator_hyperparams['parametrization']}, lb:{discriminator_hyperparams['lower_bound']}/"
#             os.makedirs(files_dir, exist_ok=True)
#             plt.savefig(f'{files_dir}/{filename}')  
#             figLegend = plt.figure(figsize = (9.2,0.5))
#             plt.figlegend(*ax.get_legend_handles_labels(), loc = 'upper left', frameon=False, ncol=5, bbox_to_anchor=(0, 1), fontsize='small')
#             plt.savefig(f"{files_dir}/adaptation_experiment_learning_curve_legend.png")

# >>>>>>> main
        
        
        
