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
scalability_exper_dir = "logs_scalability_exper/",

# generalization experiment directory
generalization_exper_dir = "generalization_exper/",
    
regularization_exper_dir = "regularization_exper/",
    
# number of the runs for the agent evaluation
eval_runs = 5,
    
# evaluation frequency
eval_freq = 1000,

# total time steps
total_timesteps = int(1e8),

# layer size of the shared network
layer_size_shared = 300,
    
# layer size of the actor
layer_size_policy = 300,

# layer size of the value function critic
layer_size_value = 300,
    
# layer size of the Q-function critic
layer_size_q = 300,

# layer size of the skills discriminator
layer_size_discriminator = 300,
    
# layer size of the skills discriminator
num_layers_discriminator = 2,

# size of the latent vector in CPC style discriminator
latent_size = 512,

# buffer size for the discriminator 
buffer_size = int(1e6),

# buffer size before training
min_train_size = int(1e4),

# max steps in the environment
max_steps = 1000,
    
# bar plot figure size
barplot_figsize = (8, 6),

font_scale = 2.5,
    
# learning curve fig size
learning_curve_figsize = (10, 8.5),
    
# random seeds
seeds = (0, 10, 42), #(0, 10, 1234),

# device
device = "cuda" if torch.cuda.is_available() else "cpu",
)