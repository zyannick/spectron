# Spectron Classification Project

This project aims to classify vibrational molecular spectra using machine learning techniques. The goal is to develop a model that can accurately predict the type of molecule based on its vibrational spectrum.


## Dataset

The dataset used for this project consists of a collection of vibrational spectra from various molecules. Each spectrum is represented as a set of numerical values corresponding to the intensity of different vibrational modes.

## Installation

To run this project, you will need to have the following dependencies installed:

```
conda create -n env python=3.10
```

- Python 3.10 or higher

You can install the required packages by running the following command:

```
pip install -r requirements.txt
```

if needed (if you got this error "AssertionError: Torch not compiled with CUDA enabled" ) you can update the torch installation:

```
conda install pytorch==2.1.0 torchvision==0.16.0 torchaudio==2.1.0 pytorch-cuda=12.1 -c pytorch -c nvidia
```

## Mlflow

For the training, you can use MLflow to track the result and compare different models.


## Usage

You can train the model using the file : train_model.ipynb. For SVM grid_search, you can use the file svm.ipynb, and for SVM with or without features extraction (PCA, PLS), you can use the file svm_features.ipynb.

You can also infer from trained weights using the file : infer_from_weights.ipynb.

You can also train the models with the config you want. For now the supported methods are:

- Fully connected model
- SVM
- XGboost
- KNN
- Random Forest

You can modify and set your own config as describes in method_config.yaml.

## Config File

Regarding the config file, there are some parameters about the datatset and the model. You can set the data augmentation you want to apply.