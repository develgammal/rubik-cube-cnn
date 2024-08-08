import torch
import torch.nn as nn
from torchmetrics import MeanAbsoluteError, MeanSquaredError, R2Score
from tqdm.auto import tqdm
from model_config import Hparams
from utils.logging import update_metrics, log_metrics, compute_and_log_avg_metrics, pretty_print

# validate one epoch
def validate_epoch(model, val_loader, criterion, mae, mse, r2, device):
    """
    Perform validation for one epoch.

    Args:
        model (torch.nn.Module): The model to be evaluated.
        val_loader (torch.utils.data.DataLoader): The validation data loader.
        criterion: The loss function.
        mae: The mean absolute error metric.
        mse: The mean squared error metric.
        r2: The R-squared metric.
        device: The device to run the evaluation on.

    Returns:
        Tuple[float, float, float, float]: The validation loss, mean absolute error,
        mean squared error, and R-squared value.
    """
    model.eval()
    val_loss, val_mae, val_mse, val_r2 = 0.0, 0.0, 0.0, 0.0

    with torch.no_grad():
        with tqdm(total=len(val_loader), desc='Validation', unit='batch') as pbar:
            for inputs, targets in val_loader:
                inputs, targets = inputs.to(device), targets.to(device)
                outputs = model(inputs)
                loss = criterion(outputs, targets.unsqueeze(1))
                val_loss, val_mae, val_mse, val_r2 = update_metrics(
                    outputs, targets, mae, mse, r2, loss, val_loss, val_mae, val_mse, val_r2)
                pbar.set_postfix({'val_loss': val_loss})
                pbar.update(1)

    return val_loss, val_mae, val_mse, val_r2

# train one epoch
def train_epoch(model, train_loader, criterion, optimizer, mae, mse, r2, device, writer, global_step):
    """
    Trains the model for one epoch.

    Args:
        model (nn.Module): The model to be trained.
        train_loader (DataLoader): The data loader for the training dataset.
        criterion (loss function): The loss function used for training.
        optimizer (optimizer): The optimizer used for updating the model's weights.
        mae (Metric): The mean absolute error metric.
        mse (Metric): The mean squared error metric.
        r2 (Metric): The R-squared metric.
        device (torch.device): The device on which the model and data are located.
        writer (SummaryWriter): The summary writer for logging metrics.
        global_step (int): The global step counter.

    Returns:
        tuple: A tuple containing the training loss, mean absolute error, mean squared error,
               R-squared value, and the updated global step counter.
    """
    model.train()
    train_loss, train_mae, train_mse, train_r2 = 0.0, 0.0, 0.0, 0.0

    with tqdm(total=len(train_loader), desc='Training', unit='batch') as pbar:
        for inputs, targets in train_loader:
            inputs, targets = inputs.to(device), targets.to(device)
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, targets.unsqueeze(1))
            loss.backward()
            optimizer.step()

            train_loss, train_mae, train_mse, train_r2 = update_metrics(
                outputs, targets, mae, mse, r2, loss, train_loss, train_mae, train_mse, train_r2)

            if writer and global_step % 10 == 0:
                log_metrics(writer, global_step, train_loss, train_mae, train_mse, train_r2, inputs, model)

            pbar.set_postfix({'loss': train_loss})
            pbar.update(1)
            global_step += 1

    return train_loss, train_mae, train_mse, train_r2, global_step

# train the neural network
def train_network(model_params : Hparams, model, train_loader, val_loader, device='cpu', writer=None):
    """
    Train the CNN regression model.

    Args:
        model_params (Hparams): Hyperparameters for the model.
        model (nn.Module): The CNN regression model.
        train_loader (DataLoader): DataLoader for the training data.
        val_loader (DataLoader): DataLoader for the validation data.
        device (str, optional): Device to train the model on. Defaults to 'cpu'.
        writer (SummaryWriter, optional): TensorBoard writer for logging. Defaults to None.

    Returns:
        nn.Module: Trained model.
    """

    # Unpack hyperparameters from the dictionary
    epochs = model_params['epochs']
    learn_rate = model_params['learn_rates']
    batch_size = model_params['batch_sizes']
    img_pixel_val = model_params['img_pixel_vals']
    activation = model_params['activations']

    # Use pretty print to log the hyperparameters
    print("Training Model with those Specific hyperparameters: ")
    pretty_print([
        ["Epochs", epochs],
        ["Learning Rate", learn_rate],
        ["Batch Size", batch_size],
        ["Image Size", img_pixel_val],
        ["Activation", activation]
    ])

    # Move the model to the specified device
    model.to(device)

    # Define the loss function and optimizer
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=learn_rate)

    # Initialize metrics
    mae = MeanAbsoluteError().to(device)
    mse = MeanSquaredError().to(device)
    r2 = R2Score().to(device)

    global_step = 0  # Initialize global step for TensorBoard logging, an epoch is one complete pass through the entire training dataset. During each epoch, the model processes every training example exactly once. In contrast, the global_step usually refers to a single update to the model's parameters, which occurs after processing a single batch of data.

    lowest_val_mse = float('inf')  # Initialize the lowest validation MSE to infinity to ensure that any valid MSE value encountered during the first validation pass will be smaller
    best_epoch = -1  # Initialize the best epoch to an invalid value

    for epoch in range(epochs):
        # Train for one epoch
        train_loss, train_mae, train_mse, train_r2, global_step = train_epoch(
            model, train_loader, criterion, optimizer, mae, mse, r2, device, writer, global_step)

        # Compute and log average training metrics
        avg_train_loss, avg_train_mae, avg_train_mse, avg_train_r2 = compute_and_log_avg_metrics(
            train_loader, train_loss, train_mae, train_mse, train_r2, writer, epoch, 'Train')

        # Validate for one epoch
        val_loss, val_mae, val_mse, val_r2 = validate_epoch(
            model, val_loader, criterion, mae, mse, r2, device)

        # Compute and log average validation metrics
        avg_val_loss, avg_val_mae, avg_val_mse, avg_val_r2 = compute_and_log_avg_metrics(
            val_loader, val_loss, val_mae, val_mse, val_r2, writer, epoch, 'Validation')

        # Check and update the lowest validation MSE since it is the most important metric in our use case
        if avg_val_mse < lowest_val_mse:
            lowest_val_mse = avg_val_mse
            best_epoch = epoch + 1 # Epochs start at 0, so add 1 to get the actual epoch number for human readability

        # Print epoch results
        print(f'Model name : {model.__class__.__name__}, Epoch [ {epoch + 1}/{epochs}], Train Loss: {avg_train_loss:.4f}, Validation Loss: {avg_val_loss:.4f}, '
              f'Train MAE: {avg_train_mae:.4f}, Validation MAE: {avg_val_mae:.4f}, '
              f'Train MSE: {avg_train_mse:.4f}, Validation MSE: {avg_val_mse:.4f}, '
              f'Train R2: {avg_train_r2:.4f}, Validation R2: {avg_val_r2:.4f}')

        # Log embeddings periodically for projector tab in tensorboard currently disabled due to performance issues
        # log_embeddings_periodically(model, val_loader, writer, device, epoch, global_step)

        # Print the lowest validation MSE after training
        print(f'Lowest Validation MSE: {lowest_val_mse:.4f} at epoch {best_epoch}')

        # Log hyperparameters and metrics to TensorBoard
        if writer:
            writer.add_hparams(
                {"learn_rate": learn_rate, "batch_size": batch_size, "epochs": epochs, "img_pixel_val": img_pixel_val, "activation": activation},
                {
                    "train_loss": avg_train_loss,
                    "val_loss": avg_val_loss,
                    "train_mae": avg_train_mae,
                    "val_mae": avg_val_mae,
                    "train_mse": avg_train_mse,
                    "val_mse": avg_val_mse,
                    "train_r2": avg_train_r2,
                    "val_r2": avg_val_r2,
                },
            )

    # Close the TensorBoard writer if it exists
    if writer:
        writer.close()

    return model