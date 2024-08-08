from typing import List
from PIL import Image
import torchvision
import os
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import csv
from tabulate import tabulate

from model_config import Hparams

# Augment the python print function for better readability
def pretty_print(table_data: List[List[str]]) -> None:
    """
    Prints a formatted table with bold headers and values.

    Args:
        table_data (List[List[str]]): A 2D list of strings representing the table data.

    Returns:
        None
    """
    # ANSI escape codes for bold
    bold = "\033[1m"
    reset = "\033[0m"

    # Apply bold formatting to the first column and header row
    headers = [f"{bold}Description{reset}", f"{bold}Value{reset}"]
    formatted_table = [[f"{bold}{cell}{reset}" if col == 0 else cell for col, cell in enumerate(row)] for row in table_data]

    # Insert headers as the first row in the table data
    formatted_table.insert(0, headers)

    # Print the table with tabulate
    print(tabulate(formatted_table, headers="firstrow", tablefmt="grid", maxcolwidths=[None, 80]))


# get the mean and standard deviation of the widths and heights of images in a given directory
def get_image_stats(directory):
    """
    Calculate the mean and standard deviation of the widths and heights of images in a given directory.

    Parameters:
    directory (str): The path to the directory containing the images.

    Returns:
    tuple: A tuple containing the mean width, standard deviation of width, mean height, and standard deviation of height.
    """
    # Initialize lists to store widths and heights
    widths = []
    heights = []

    # Loop through all images and store their widths and heights
    for img_name_path in os.listdir(directory):
        img_path = os.path.join(directory, img_name_path)
        with Image.open(img_path) as img:
            widths.append(img.width)
            heights.append(img.height)

    # Convert lists to numpy arrays
    widths = np.array(widths)
    heights = np.array(heights)

    # Calculate mean and standard deviation
    mean_width = np.mean(widths)
    std_width = np.std(widths)
    mean_height = np.mean(heights)
    std_height = np.std(heights)

    return mean_width, std_width, mean_height, std_height

# Function to update and return metrics during training or validation
def update_metrics(outputs, targets, mae, mse, r2, loss, total_loss, total_mae, total_mse, total_r2):
    """
    Update the metrics and loss values for the given outputs and targets.

    Parameters:
    - outputs (Tensor): The predicted outputs.
    - targets (Tensor): The target values.
    - mae (function): The mean absolute error function.
    - mse (function): The mean squared error function.
    - r2 (function): The R-squared function.
    - loss (Tensor): The loss value.
    - total_loss (float): The accumulated loss value.
    - total_mae (float): The accumulated mean absolute error value.
    - total_mse (float): The accumulated mean squared error value.
    - total_r2 (float): The accumulated R-squared value.

    Returns:
    - total_loss (float): The updated accumulated loss value.
    - total_mae (float): The updated accumulated mean absolute error value.
    - total_mse (float): The updated accumulated mean squared error value.
    - total_r2 (float): The updated accumulated R-squared value.
    - loss_value (float): The individual loss value.
    """
    # Calculate individual loss value
    loss_value = loss.item()
    # Accumulate the loss and metrics
    total_loss += loss_value
    total_mae += mae(outputs, targets.unsqueeze(1)).item() # unsqueeze to match the shape of the outputs
    total_mse += mse(outputs, targets.unsqueeze(1)).item() # gives more weight to larger errors, thus penalizing them more heavily compared to MAE
    total_r2 += r2(outputs, targets.unsqueeze(1)).item()

    return total_loss, total_mae, total_mse, total_r2

# Log training metrics to TensorBoard
def log_metrics(writer, global_step, loss, train_mae, train_mse, train_r2, inputs, model):
    """
    Logs metrics, images, and weights to TensorBoard.

    Args:
        writer (SummaryWriter): The TensorBoard SummaryWriter object used to log the data.
        global_step (int): The global step of the training process "processing a single batch of data"
        loss (float): The training loss value.
        train_mae (float): The Mean Absolute Error (MAE) value.
        train_mse (float): The Mean Squared Error (MSE) "Penalizes large errors more".
        train_r2 (float): The R-squared (R2) value.
        inputs (torch.Tensor): The input images to the model.
        model (torch.nn.Module): The model being trained.

    Returns:
        None
    """
    # Create a grid of images
    img_grid = torchvision.utils.make_grid(inputs)
    # Log training images, weights, and metrics
    writer.add_image('Training images', img_grid, global_step=global_step)
    writer.add_histogram('fc1', model.fc1.weight, global_step=global_step)
    writer.add_scalar('Training loss', loss, global_step=global_step)
    writer.add_scalar('Training MAE', train_mae, global_step=global_step)
    writer.add_scalar('Training MSE', train_mse, global_step=global_step)
    writer.add_scalar('Training R2', train_r2, global_step=global_step)

# Compute and log average metrics for an epoch
def compute_and_log_avg_metrics(loader, total_loss, total_mae, total_mse, total_r2, writer, epoch, phase):
    """
    Computes the average metrics (loss, MAE, MSE, R2) and logs them to TensorBoard.

    Parameters:
    - loader (iterable): The data loader.
    - total_loss (float): The total loss accumulated during the epoch.
    - total_mae (float): The total mean absolute error accumulated during the epoch.
    - total_mse (float): The total mean squared error accumulated during the epoch.
    - total_r2 (float): The total R-squared value accumulated during the epoch.
    - writer (SummaryWriter): The TensorBoard writer.
    - epoch (int): The current epoch number.
    - phase (str): The phase of the training (e.g., 'train', 'val').

    Returns:
    - avg_loss (float): The average loss.
    - avg_mae (float): The average mean absolute error.
    - avg_mse (float): The average mean squared error.
    - avg_r2 (float): The average R-squared value.
    """
    # Calculate average values
    avg_loss = total_loss / len(loader)
    avg_mae = total_mae / len(loader)
    avg_mse = total_mse / len(loader)
    avg_r2 = total_r2 / len(loader)

    # Log average metrics to TensorBoard
    if writer:
        writer.add_scalar(f'{phase}/Loss', avg_loss, epoch)
        writer.add_scalar(f'{phase}/MAE', avg_mae, epoch)
        writer.add_scalar(f'{phase}/MSE', avg_mse, epoch)
        writer.add_scalar(f'{phase}/R2', avg_r2, epoch)

    return avg_loss, avg_mae, avg_mse, avg_r2

# Generate a log directory path based on the given base directory and hyperparameters.
def generate_log_dir(base_dir, hparams : Hparams):
    """
    Generate a log directory path based on the given base directory and hyperparameters.

    Parameters:
    base_dir (str): The base directory where the log directory will be created.
    hparams (Hparams): The hyperparameters used to generate the dynamic part of the log directory.

    Returns:
    str: The path of the generated log directory.

    """
    # Generate the dynamic part of the log directory
    dynamic_dir = "_".join([f"{key.upper()}_{val}" for key, val in hparams.items()])

    # timestamp added to prevent overwriting
    dynamic_part = dynamic_dir +'_T'+ datetime.now().strftime("%Y-%m-%d_%H-%M-%S")

    # Combine with base directory and current timestamp
    log_dir_path = os.path.join(base_dir, dynamic_part)

    # Ensure the base directory exists
    os.makedirs(base_dir, exist_ok=True)

    return log_dir_path

# Track the execution time of a function and save the results to a CSV file.
def track_execution_time(start_time : float, end_time : float, save_state_sict_path : str, hparams_grid : Hparams, base_dir : str='execution-times'):
    """
    Track the execution time of a function and save the results to a CSV file.

    Parameters:
    - start_time (float): The start time of the execution.
    - end_time (float): The end time of the execution.
    - save_state_sict_path (str): The file path to save the state of the execution.
    - hparams_grid (Hparams): The hyperparameters grid used for the execution.
    - base_dir (str): The base directory to save the execution times CSV file. Default is 'execution-times'.

    Returns:
    None
    """
    # Create the base directory if it does not exist
    os.makedirs(base_dir, exist_ok=True)

    # Function to format hyperparameter values
    def format_hparam_value(value):
        if isinstance(value, list):
            if len(value) > 1:
                return ','.join(map(str, value))
            else:
                return str(value[0])
        return str(value)

    # Generate a file name based on the hyperparameters grid
    hparams_str = '-'.join([f"{key}-{format_hparam_value(value).replace(' ', '')}" for key, value in hparams_grid.items()])
    csv_file = os.path.join(base_dir, f"{hparams_str}.csv")

    # Calculate execution time and format it
    execution_time = end_time - start_time
    execution_time_str = str(timedelta(seconds=execution_time)).split('.')[0]  # Remove milliseconds for readability

    start_time_str = datetime.fromtimestamp(start_time).strftime('%Y-%m-%d %H:%M:%S')
    end_time_str = datetime.fromtimestamp(end_time).strftime('%Y-%m-%d %H:%M:%S')

    cumulative_time = timedelta(seconds=0)
    if os.path.isfile(csv_file):
        df = pd.read_csv(csv_file)
        cumulative_time = pd.to_timedelta(df['Execution Time (HH:MM:SS)']).sum()

    cumulative_time += timedelta(seconds=execution_time)
    cumulative_time_str = str(cumulative_time).split('.')[0]

    file_exists = os.path.isfile(csv_file)
    with open(csv_file, mode='a', newline='') as file:
        writer = csv.writer(file)
        if not file_exists:
            writer.writerow(['Model File Path', 'Execution Time (HH:MM:SS)', 'Start Time', 'End Time', 'Cumulative Duration'])
        writer.writerow([save_state_sict_path, execution_time_str, start_time_str, end_time_str, cumulative_time_str])

    print("Execution time :")
    pretty_print([["Model File Path", save_state_sict_path], ["Execution Time", execution_time_str], ["Start Time", start_time_str], ["End Time", end_time_str], ["Cumulative Duration", cumulative_time_str]])

