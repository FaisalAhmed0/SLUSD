import argparse
import numpy as np
import matplotlib.pyplot as plt

import os
from src.config import conf




def cmd_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--env', type=str, default="MountainCarContinuous-v0")
    parser.add_argument("--alg", type=str, default="ppo")
    parser.add_argument("--stamp", type=str, required=True)
    args = parser.parse_args()
    return args


def plot(x, y, xlabel, ylabel, legend, color, filename):
    fig = plt.figure()
    axes  = fig.add_subplot(111)
    plot_color = color
    data_mean = y.mean(axis=1)
    data_std = y.std(axis=1)
    axes.plot(x, data_mean, label=legend, color=plot_color)
    axes.fill_between(x, (data_mean-data_std), (data_mean+data_std), color=plot_color, alpha=0.3)
    axes.set_xlabel(xlabel)
    axes.set_ylabel(ylabel)
    axes.grid(True)
    axes.legend()
    files_dir = f"Vis/{args.alg}_{args.env}_{args.stamp}"
    os.makedirs(files_dir, exist_ok=True)
    fig.savefig(f'{files_dir}/{filename}', dpi=150)



if __name__ == "__main__":
    args = cmd_args()
    file_dir = conf.log_dir_finetune + f"{args.alg}_{args.env}_{args.stamp}/" + "eval_results/" + "evaluations.npz"
    files = np.load(file_dir)
    steps = files['timesteps']
    results = files['results']
    plot(steps, results, "Timesteps", "Reward", f"{args.alg}", 'b', f"{args.alg}_{args.env}")
