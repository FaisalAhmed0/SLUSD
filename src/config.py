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

# number of the runs for the agent evaluation
eval_runs = 5,

# total time steps
total_timesteps = int(1e5),

# layer size of the actor
layer_size_policy = 300,

# layer size of the critic
layer_size_value = 300,

# layer size of the skills discriminator
layer_size_discriminator = 300,

# number of skills
n_z = 8,

# buffer size for the discriminator 
buffer_size = int(1e7),

# buffer size before training
min_train_size = int(1e4),

# max steps in the environment
max_steps = 1000,

# device
device = "cpu",
)