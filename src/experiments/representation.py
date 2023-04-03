import argparse
import pickle
import json

from copy import deepcopy

from src.config import conf
# diayn with model-free RL
from src.diayn import DIAYN 

# Utils
from src.utils import seed_everything
from src.utils import evaluate_pretrained_policy_ext
from src.utils import evaluate_adapted_policy
from src.utils import evaluate_state_coverage
from src.utils import evaluate
from src.utils import save_final_results
from src.utils import plot_learning_curves

import torch
import torch.multiprocessing as mp
torch.set_num_threads(1)
torch.multiprocessing.set_sharing_strategy('file_system') # For running multiple experiments in parallel 


import numpy as np
import pandas as pd
import gym
import d4rl

import time
import os
import random
from collections import defaultdict

import matplotlib.pyplot as plt
import seaborn as sns
sns.set()
sns.set_style('whitegrid')
sns.set_context("paper", font_scale = conf.font_scale)

# Asymptotic performance of the expert
asymp_perofrmance = {
    'HalfCheetah-v2': 8070.205213363435,
    'Walker2d-v2': 5966.481684037183,
    'Ant-v2': 6898.620003217143,
    "MountainCarContinuous-v0": 95.0

    }

# shared parameters
params = dict( n_skills = 30,
           pretrain_steps = int(20e3),
           finetune_steps = int(1e5),
           buffer_size = int(1e6),
           min_train_size = int(1e4),
             )