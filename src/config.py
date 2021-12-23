import torch
from argparse import Namespace
# configiration variables
conf = Namespace(
# location of the log folder (testing)
testing_log_dir = "testing_logs/",

# locationn for the actual training logs
log_dir = "logs/",

# locationn for the videos for the agent interactions
videos_dir = "videos/",

# log directory for the fine tunning
log_dir_finetune  = "logs_finetune/",

# pretrain_steps_expriment_directory
pretrain_steps_exper_dir = "pretrain_steps_exper/",

# generalization experiment directory
generalization_exper_dir = "generalization_exper/",
    
# number of the runs for the agent evaluation
eval_runs = 5,

# total time steps
total_timesteps = int(1e8),

# layer size of the shared network
layer_size_shared = 512,
    
# layer size of the actor
layer_size_policy = 128,

# layer size of the value function critic
layer_size_value = 128,
    
# layer size of the Q-function critic
layer_size_q = 500,

# layer size of the skills discriminator
layer_size_discriminator = 128,

# size of the latent vector in CPC style discriminator
latent_size = 50,

# number of skills
n_z = 12,

# buffer size for the discriminator 
buffer_size = int(1e6),

# buffer size before training
min_train_size = int(1e4),

# max steps in the environment
max_steps = 1000,

# device
device = "cuda" if torch.cuda.is_available() else "cpu",
)