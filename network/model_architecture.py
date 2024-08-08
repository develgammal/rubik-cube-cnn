from typing import Tuple
import torch.nn as nn

ImageSize = Tuple[int, int, int]
class CNNRegression(nn.Module):
    def __init__(self, image_size : ImageSize , activation_func_name):
        """
        Initializes a CNNRegression object.

        Args:
        - image_size (tuple): A tuple representing the size of the input image in the format (channels, height, width).
        """
        super(CNNRegression, self).__init__()

        self.image_size = image_size

        # activation function
        self.activation_func = getattr(nn.functional, activation_func_name)


        # First Convolutional Layer and Pooling Layer
        self.conv1 = nn.Conv2d(in_channels=self.image_size[0], out_channels=4, kernel_size=3, stride=1, padding=1)
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2)

        # Second Convolutional Layer and Pooling Layer
        self.conv2 = nn.Conv2d(in_channels=4, out_channels=16, kernel_size=3, stride=1, padding=1)
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2)

        # Fully Connected Layers
        self.fc_input_size = int(16 * (self.image_size[1] // 4) * (self.image_size[2] // 4))
        self.hidden_layer_size = 128
        self.output_size = 1 # Regression output, so only 1 value if it was classification it would be the number of classes (e.g.len(class_names) )

        self.fc1 = nn.Linear(in_features=self.fc_input_size, out_features=self.hidden_layer_size)
        self.fc2 = nn.Linear(in_features=self.hidden_layer_size, out_features=self.output_size)  # Output 1 value for regression

    def forward(self, x):
        """
        Passes the data through the network.
        There are commented out print statements that can be used to
        check the size of the tensor at each layer. These are very useful when
        the image size changes and you want to check that the network layers are
        still the correct shape.
        """
        # layers = []

        # The batch size does not need to be explicitly referenced within the model definition. It is implicitly managed by the DataLoader and the tensor operations in the model. The DataLoader ensures that tensors are batched, and these batches are processed by the model. The shapes of the tensors during the forward pass reflect the batch size as the first dimension. In this specific example, the batch size of 64 comes from how the DataLoader was set up to batch the data.

        # Forward pass through the first convolutional layer and pooling layer
        x = self.conv1(x)
        # layers.append(["conv1", str(x.size())])
        x = self.activation_func(x)
        # layers.append(["relu1", str(x.size())])
        x = self.pool1(x)
        # layers.append(["pool1", str(x.size())])

        # Forward pass through the second convolutional layer and pooling layer
        x = self.conv2(x)
        # layers.append(["conv2", str(x.size())])
        x = self.activation_func(x)
        # layers.append(["relu2", str(x.size())])
        x = self.pool2(x)
        # layers.append(["pool2", str(x.size())])

        # Flattening the tensor before passing it to fully connected layers
        x = x.view(-1, self.fc_input_size)
        # layers.append(["view1", str(x.size())])

        # Forward pass through the first fully connected layer
        x = self.fc1(x)
        # layers.append(["fc1", str(x.size())])
        x = self.activation_func(x)
        # layers.append(["relu2", str(x.size())])

        # Forward pass through the second fully connected layer
        x = self.fc2(x)
        # layers.append(["fc2", str(x.size())])

        # Use pretty_print to log the layer sizes
        # pretty_print(layers)

        return x