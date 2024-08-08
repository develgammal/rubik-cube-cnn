# Rubik-Cube-CNN

A Convolutional Neural Network (CNN) leveraging Computer Vision to predict the axis of rotation of a Rubik's Cube.

![Image Alt Text](https://github.com/develgammal/rubik-cube-cnn/blob/master/screenshots/background-img.jpg)
![Image Alt Text](https://github.com/develgammal/rubik-cube-cnn/blob/master/screenshots/mse.png)

## Overview

This project utilizes a Convolutional Neural Network to analyze and predict the orientation of a Rubik's Cube based on input images. The network is designed to handle both training and evaluation modes, and incorporates various utilities for data preprocessing, logging, and visualization.

## Features

- **Image Analysis**: Processes images to extract meaningful features for orientation prediction.
![Image Alt Text](https://github.com/develgammal/rubik-cube-cnn/blob/master/screenshots/error.png)
- **Training and Evaluation**: Supports both modes to facilitate model development and testing.
![Image Alt Text](https://github.com/develgammal/rubik-cube-cnn/blob/master/screenshots/comparison.png)
- **Visualization**: Includes tools to visualize training metrics and predictions.
![Image Alt Text](https://github.com/develgammal/rubik-cube-cnn/blob/master/screenshots/distribution.png)
- **Logging**: Detailed logging of training progress and metrics for performance tracking.
![Image Alt Text](https://github.com/develgammal/rubik-cube-cnn/blob/master/screenshots/validation.png)
![Image Alt Text](https://github.com/develgammal/rubik-cube-cnn/blob/master/screenshots/weights.png)

## Getting Started

### Installation

Clone the repository and navigate to the project directory:

```sh
git clone <repo path>
cd rubik-cube-cnn
```

Install the required packages:

```sh
pip install -r requirements.txt
```

### Configuration

#### Training Mode

To run the program in training mode, configure the hyperparameters then set the `eval_mode` variable to `False` in `model_config.py`:

```python
eval_mode = False
```

This will train the model using the specified hyperparameters and save the trained model.

#### Evaluation Mode

*To run the program in evaluation mode, set the `eval_mode` variable to `True` in `model_config.py` (make sure img_pixel_vals variable matches your loaded file):*

```python
eval_mode = True
```
#### Paths

This will load the trained model and visualize its predictions on the validation dataset.

Configure the necessary paths and parameters in the `model_config.py` file. Set the following variables to match your dataset and model requirements:

```python
# Define paths to the dataset ( you can leave them by default)
train_dir_path = 'source/training/training/images'
labels_file_path = 'source/training/training/labels.csv'

# Set the path to the saved state dictionary of the trained model (after you generated one if needed)
load_state_dict_path = r'<path_to_your_trained_model'
```

### Running the Program

To run the program, use the following command:

```sh
python main.ipynb
```

Alternatively, if you prefer the Jupyter notebook format, you will find two fully documented notebooks in the project root: `university_documented-notebook-en.ipynb` (in English) and `university_documented-notebook-de.ipynb` (in German). These notebooks were used for my university assignment. Please note, however, that I will not be maintaining them, so they may not include all future features.

## Project Structure

```
rubik-cube-cnn/
│
├── data/
├── network/
├── utils/
├── config.py
├── main.py
├── README.md
```

- **data**: Contains scripts for data preprocessing and analysis.
- **network**: Includes the model architecture and training loop scripts.
- **utils**: Utility scripts for logging, visualization, and cleanup.
- **main.py**: The main script to run the program.
- **config.py**: Configuration file for setting paths and hyperparameters.
