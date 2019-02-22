"""Parameters for ARDA."""

# params for dataset and data loader
data_root = "data"
dataset_mean_value = 0.5
dataset_std_value = 0.5
dataset_mean = (dataset_mean_value, dataset_mean_value, dataset_mean_value)
dataset_std = (dataset_std_value, dataset_std_value, dataset_std_value)
batch_size = 30
#image_size = 64
src_dataset = "caltech"



# params for training network
num_gpu = 1
num_epochs = 20000
log_step = 20
save_step = 5000
manual_seed = None
model_root = "snapshots"
eval_only = False

# params for optimizing models
learning_rate = 5e-5
#beta1 = 0.5
#beta2 = 0.9

# params for WGAN and WGAN-GP
#use_gradient_penalty = True  # quickly switch WGAN and WGAN-GP
#penalty_lambda = 10

# params for interaction of discriminative and transferable feature learning
#dc_lambda = 10
