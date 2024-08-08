import matplotlib.pyplot as plt
import random
import torch
from tensorboard.backend.event_processing import event_accumulator

# Visualize predictions
def visualize_predictions(model, val_loader, num_samples=8):
    """
    Visualizes the predictions made by a given model on a validation dataset.

    Args:
        model (torch.nn.Module): The trained model.
        val_loader (torch.utils.data.DataLoader): The validation data loader.
        device (torch.device): The device to perform computations on.
        num_samples (int, optional): The number of samples to visualize. Defaults to 8.
    """
    model.eval()
    samples = random.sample(range(len(val_loader.dataset)), num_samples)

    fig, axes = plt.subplots(num_samples // 2, 2, figsize=(10, num_samples * 2))
    axes = axes.flatten()

    for i, idx in enumerate(samples):
        image, true_label = val_loader.dataset[idx]

        print(f"Image shape: {image.shape}, label type: {true_label.dtype}")

        with torch.no_grad():
            prediction = model(image.unsqueeze(0).to(next(model.parameters()).device)).cpu().item()  # Ensure image is in a batch format

        image = image.permute(1, 2, 0)  # Move channels to the end for plotting compatibility

        # Calculate percentage error
        percentage_error = abs((true_label - prediction) / true_label) * 100

        axes[i].imshow(image, cmap='gray')
        axes[i].set_title(f"True: {true_label:.2f}, Pred: {prediction:.2f}, Error: {percentage_error:.2f}%")
        axes[i].axis('off')

    plt.tight_layout()
    plt.show()

# Plot and save metrics
def plot_and_save_metrics(tensorboard_log_dir_path : str, metrics_file_path : str):
    """
    Plot and save metrics from TensorBoard logs.

    Parameters:
    tensorboard_log_dir_path (str): The directory path where the TensorBoard logs are stored.
    metrics_file_path (str): The full path and file name with extension where the metrics plots will be saved.

    Returns:
    None
    """
    ea = event_accumulator.EventAccumulator(tensorboard_log_dir_path)
    ea.Reload()

    metrics = ['Train/Loss', 'Validation/Loss', 'Train/MAE', 'Validation/MAE', 'Train/MSE', 'Validation/MSE', 'Train/R2', 'Validation/R2']
    fig, axes = plt.subplots(len(metrics) // 2, 2, figsize=(15, 20))
    axes = axes.flatten()

    for ax, metric in zip(axes, metrics):
        events = ea.Scalars(metric)
        steps = [e.step for e in events]
        values = [e.value for e in events]
        ax.plot(steps, values, label=metric)
        ax.set_title(metric)
        ax.legend()
        ax.grid(True)

    plt.tight_layout()
    print(f"Saving metrics plot to {metrics_file_path}")
    fig.savefig(metrics_file_path)  # Save the figure to a file
    plt.show()
