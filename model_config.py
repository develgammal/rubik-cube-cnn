import os

# set the document in train mode or evaluation mode
eval_mode: bool = True

# Define paths to the dataset
train_dir_path: str = 'source/training/training/images'
labels_file_path: str = 'source/training/training/labels.csv'

# set the path to the dataset using a raw string to avoid escape characters
load_state_dict_path: str = r'models/EPOCHS_400_BATCH_SIZES_32_IMG_PIXEL_VALS_128_LEARN_RATES_0.001_ACTIVATIONS_relu_T2024-07-24_08-33-45.pth'

# check if the file exists
if not os.path.isfile(load_state_dict_path) and eval_mode:
    raise FileNotFoundError(f"The file {load_state_dict_path} does not exist.")

# define hyperparameters grid
Hparams = dict[str, list]

Hparams_grid: Hparams = {
    "epochs": [400],
    "batch_sizes": [32],
    "img_pixel_vals": [128],  # when loading a state dict make sure to use the same value here
    "learn_rates": [0.001],
    "activations": ['relu'],
}

# quick train for debugging

# Hparams_grid: Hparams = {
#     "epochs": [2],
#     "batch_sizes": [32],
#     "img_pixel_vals": [128],  # when loading a state dict make sure to use the same value here
#     "learn_rates": [0.1],
#     "activations": ['relu'],
# }
