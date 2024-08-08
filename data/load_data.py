import pandas as pd
from torch.utils.data import Dataset
from PIL import Image
import os
import torch

# Custom dataset class for Rubik's Cube images dataset with PyTorch
class RubikCustomDataset(Dataset):
    def __init__(self, img_dir_path: str, labels_file_path: str, transform=None):
        """
        Initializes a custom dataset for Rubik's Cube images.

        Args:
            img_dir_path (str): The directory path where the images are stored.
            labels_file_path (str): The file path of the CSV file containing the image labels.
            transform (callable, optional): A function/transform to be applied on the images. Default is None.
        """
        self.img_dir_path = img_dir_path
        self.labels = pd.read_csv(labels_file_path)
        self.transform = transform

    # magic methods
    def __len__(self):
        """
        Returns the total number of samples in the dataset.

        Returns:
            int: The number of samples in the dataset.
        """
        return len(self.labels)

    def __iter__(self):
        """
        Returns an iterator over the dataset.

        Returns:
            iterator: An iterator over the dataset.
        """
        return iter(self.labels)

    def __getitem__(self, idx):
        """
        Retrieves the image and label at the given index.

        Args:
            idx (int): The index of the sample to retrieve.

        Returns:
            tuple: A tuple containing the image and label.
        """
        img_name_path = os.path.join(self.img_dir_path, self.labels.iloc[idx, 0])
        image = Image.open(img_name_path)
        label = self.labels.iloc[idx, 1]
        if self.transform:
            image = self.transform(image)
        label = torch.tensor(label, dtype=torch.float32)
        return image, label

def load_data(img_dir_path, labels_file_path, transform):
    dataset = RubikCustomDataset(img_dir_path, labels_file_path, transform)
    return dataset
