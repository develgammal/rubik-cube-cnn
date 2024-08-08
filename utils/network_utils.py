import itertools
import os
import torch
from model_config import Hparams

# save the model
def save_model(trained_model, save_state_sict_path='model.pth'):
    """
    Save the trained model's state dictionary to a file.

    Parameters:
    - trained_model: The trained model object.
    - save_state_sict_path: The path to save the model's state dictionary. Default is 'model.pth'.
    """
    print(f'Saving model to {save_state_sict_path}')
    torch.save(trained_model.state_dict(), save_state_sict_path)

# load the model
def load_model(model_to_update, load_state_dict_path):
    """
    Loads a model from a given state dictionary file.

    Args:
        model_to_update (torch.nn.Module): The model to update with the loaded state dictionary.
        load_state_dict_path (str): The path to the state dictionary file.

    Returns:
        torch.nn.Module: The loaded model.

    Raises:
        FileNotFoundError: If the model file is not found at the specified path.
    """
    print(f'Loading model from {load_state_dict_path}')
    # Check if model file exists
    if not os.path.exists(load_state_dict_path):
        raise FileNotFoundError(f"Model file not found at path {load_state_dict_path}")

    # Load the state dictionary from the file
    state_dict = torch.load(load_state_dict_path)

    # Get the number of output features in the 'fc1' layer of the state dictionary
    fc1_weight_shape = state_dict['fc1.weight'].shape

    num_output_features = fc1_weight_shape[0]
    img_pixel_val = model_to_update.image_size[1]

    if num_output_features != img_pixel_val:
        raise ValueError(f"Number of output features in the 'fc1' layer of the state dictionary ({num_output_features}) and img_pixel_val of the untrained model ({img_pixel_val}) do not match. Please update the img_pixel_val of the untrained model in the model_config file to {num_output_features}.")

    # Load the state dictionary into the model and capture the combination process in a variable for mismatch checking
    combining_keys = model_to_update.load_state_dict(state_dict, strict=False)

    # check if there are any missing or unexpected keys when loading the model
    if combining_keys.missing_keys:
        print(f'Missing keys when loading the model: {combining_keys.missing_keys}')
    if combining_keys.unexpected_keys:
        print(f'Unexpected keys when loading the model: {combining_keys.unexpected_keys}')

    loaded_model = model_to_update # now the model is updated from state_dict and checked for missing or unexpected keys it can be used for inference

    return loaded_model

# generate all possible combinations of hyperparameters
def generate_combinations(grid: Hparams):
    """
    Generate all possible combinations of hyperparameters from a given grid.

    Parameters:
    grid (Hparams): A dictionary containing the hyperparameter grid.

    Returns:
    list: A list of dictionaries representing all possible combinations of hyperparameters.
    """
    keys, values = zip(*grid.items()) # Unpacks the dictionary into two tuples: keys containing the hyperparameter names and values containing lists of possible values for each hyperparameter

    combinations = [dict(zip(keys, v)) for v in itertools.product(*values)] # generates the Cartesian product of the hyperparameter values, resulting in all possible combinations. For each combination, zip(keys, v) pairs the hyperparameter names with the values, and dict(zip(keys, v)) creates a dictionary representing that combination
    return combinations
