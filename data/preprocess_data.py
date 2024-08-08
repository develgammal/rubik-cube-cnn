import torchvision.transforms as transforms
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, Subset
from .load_data import RubikCustomDataset
from utils.logging import pretty_print

# helper function to calculate crop value
def calculate_crop_value(pixel_value):
    """
    Calculates the crop value based on the given pixel value.

    Parameters:
    pixel_value (float): The pixel value to calculate the crop value for.

    Returns:
    float: The calculated crop value.
    """
    return (64 / pixel_value) * 5.0

# Transforms the dataset so that it can be used for training
class TransformedDataset:
    """
    A class representing a transformed dataset for a machine learning project.

    Args:
        model_specific_hparams (dict): The model-specific hyperparameters.

    Attributes:
        img_dir_path (str): The file path to the directory containing the images.
        labels_file_path (str): The file path to the labels CSV file.
        img_pixel_val (int): The numbers of width = length of the square images pixels.
        batch_size (int): The batch size for the data loaders.
        test_split_size (float): The size of the test split as a fraction of the dataset.

    Methods:
        __init__(self, model_specific_hparams)
        __calculate_crop_dimensions(self)
        __create_transform(self)
        __create_dataset(self)
        __create_data_loaders(self)
        __len__(self)
        __getitem__(self, idx)
        __iter__(self)
    """

    def __init__(self, model_specific_hparams: dict):
        """
        Initializes a TransformedDataset object.

        Args:
            model_specific_hparams (dict): The model-specific hyperparameters.
        """

        # file paths
        self.img_dir_path: str = model_specific_hparams.get("img_dir_path", 'source/training/training/images')
        self.labels_file_path: str = model_specific_hparams.get("labels_file_path", 'source/training/training/labels.csv')

        # model parameters
        self.img_pixel_val = model_specific_hparams.get("img_pixel_vals")
        self.batch_size = model_specific_hparams.get("batch_sizes")
        self.test_split_size = model_specific_hparams.get("test_split_size", 0.2)

        print("Creating a dataset with the following parameters:")
        pretty_print([
            ["img_pixel_val", self.img_pixel_val],
            ["batch_size", self.batch_size],
            ["test_split_size", self.test_split_size]
        ])

        # calculated values
        self.crop_dimensions = self.__calculate_crop_dimensions()
        self.transform = self.__create_transform()
        self.dataset = self.__create_dataset()
        self.train_loader, self.val_loader = self.__create_data_loaders()

    # private methods
    def __calculate_crop_dimensions(self):
        crop_height = int(self.img_pixel_val * calculate_crop_value(self.img_pixel_val))
        crop_width = int(self.img_pixel_val * calculate_crop_value(self.img_pixel_val))
        return (crop_height, crop_width)

    def __create_transform(self):
        resize_dimensions = (self.img_pixel_val, self.img_pixel_val)
        return transforms.Compose([
            transforms.CenterCrop(self.crop_dimensions),
            transforms.Grayscale(),
            transforms.Resize(resize_dimensions),
            transforms.ToTensor()
        ])

    def __create_dataset(self):
        return RubikCustomDataset(img_dir_path=self.img_dir_path, labels_file_path=self.labels_file_path, transform=self.transform)

    def __create_data_loaders(self):
        train_indices, val_indices = train_test_split(
            list(range(len(self.dataset))), test_size=self.test_split_size, random_state=42
        )
        train_dataset = Subset(self.dataset, train_indices)
        val_dataset = Subset(self.dataset, val_indices)

        train_loader = DataLoader(train_dataset, batch_size=self.batch_size, shuffle=True)
        val_loader = DataLoader(val_dataset, batch_size=self.batch_size, shuffle=False)

        return train_loader, val_loader

    # magic methods
    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        return self.dataset[idx]

    def __iter__(self):
        return iter(self.dataset)
