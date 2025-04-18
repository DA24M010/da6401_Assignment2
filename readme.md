## Deep Learning(DA6401) Assignment 2
Repo for assignment2 submission in DA6401

#### Roll no: DA24M010
#### Name: Mohit Singh

## Wandb Report link : 
https://wandb.ai/da24m010-indian-institute-of-technology-madras/DA6401%20Assignments/reports/DA6401-Assignment-2--VmlldzoxMjExMDQwNA

## Github repo link :
https://github.com/DA24M010/da6401_Assignment2.git

# Assignment Overview
The goal of this assignment is twofold: 
1. train a CNN model from scratch and learn how to tune the hyperparameters and visualize filters
2. finetune a pre-trained model just as you would do in many real-world applications

This project implements a **CNN network** from scratch using **PyTorch** and finetune VGG-16 model using **Torchvision** and other libraries allowed. The network is trained and tested on the **iNaturalist12K** for classifying images into **10 different categories**.

# Structure
```
├── partA/
│   ├── extract_dataset.py
│   ├── train.csv
│   ├── val.csv
│   ├── data.py
│   ├── model.py
│   ├── train.py
│   ├── train_sweeps.py
│   └── train_best_model.py
├── partB/
│   ├── train_vgg.py
│   ├── vgg_model.py
│   └── vgg_sweeps.py
├── .gitignore
├── readme.md
└── requirements.txt

```

### Part A
- **`extract_data.py`**: Generates CSV files for train and validation splits from the raw dataset, used for loading data via custom dataloader.
- **`data.py`**: Defines the N12KDATA dataset class and a utility function to return data loaders for train, validation, and test sets.
- **`model.py`**: Implements the CNN architecture used for training on the iNaturalist 12K dataset.
- **`train.py`**: Contains the training loop logic for the CNN model, using specified hyperparameters.
- **`train_sweeps.py`**: Executes WandB hyperparameter sweeps to tune the CNN model and logs metrics.
- **`train_best_model.py`**: Trains the CNN using the best hyperparameters from WandB sweeps, evaluates on the test set, and logs final test accuracy and predictions.

### Part B
- **`vgg_model.py`**: Implements the VGG model with support for 3 fine-tuning strategies mentioned in the report.
- **`train_vgg.py`**: Contains the training loop logic for fine-tuning the VGG model.
- **`vgg_sweeps.py`**: Runs WandB sweeps for VGG model fine-tuning and logs validation metrics.

*Part B uses the same scripts for loading the data as Part A.*

# Hyperparameter Tuning

WandB sweeps are used to explore various configurations for training the CNN in partA. The following hyperparameters are considered:

- **Number of filters** (Number of filters in the convolution layers): [32, 64, 128]
- **Filter Multiplier** (Multiplier for organising the filters, 1 means all conv layers have same filter, 0.5 means halving the filter value in subsequent layers, 2 means doubling the filter value): [0.5, 1, 2]
- **Kernel Size** (Size of k*k kernel used in conv layers): [3, 5]
- **Activation** (Activation function used in conv and dense layers): [relu, leaky_relu, gelu, silu, mish]
- **Rate of Dropout** (Dropout percentage applied in conv and dense layers): [0.2, 0.5, 0.7]
- **Batch normalization** (If True applies batchnorm in conv layers else no batch normalization): [True, False]
- **Dense units** (Number of nodes in the dense layer): [512, 1024]
- **Data Augmentation** (Applies data augmentation(random rotation, flip, crop, jitter) in training data if True): [True, False]
- **Learning Rate** (learning rate): [0.01, 0.001, 0.0001]

WandB will automatically generate plots for the sweeps. Each run is given a meaningful name (e.g., 
`nf_128_ks_5_fm_2_act_silu_do_0.2_bn_True_da_True_lr_0.0001_du_1024`)

# Installation and Setup
### 1. Clone the repository:
```sh
git clone https://github.com/DA24M010/da6401_Assignment2.git
cd da6401_Assignment2
```

### 2. Install dependencies:
```sh
pip install -r requirements.txt
```

### 3. Setup Weights & Biases (W&B)
Create an account on [W&B](https://wandb.ai/) and log in:
```sh
wandb login
```

# Running Scripts
## Part A
### Running hypereparameter tuning (CNN)
Running hyperparameter tuning on CNN model and logging to wandb
```bash
python ./partA/scripts/train_sweeps.py --project your_project_name --entity your_wandb_username
```
*Change the hyperparameters for tuning inside the script.*

### Evaluating the Model on the Test Set
If you need to evaluate model for a specific set of hyperparameters on the test set:
```bash
python ./partA/scripts/train_best_model.py --project your_project_name --entity your_wandb_username 
```
*Change the hyperparameters for running inference inside the script. Generates test accuracy and prediction on test dataset logs in WandB*

## Part B
### Running Fine tuning on VGG 16
If you need to run fine tuning on VGG 16 and log the metrics in WandB:
```bash
python ./partB/vgg_sweeps.py --project your_project_name --entity your_wandb_username 
```
