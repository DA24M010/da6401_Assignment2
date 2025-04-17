from data import get_data_loaders
from model import CNNModel
from train import train_model
import torch
import wandb

sweep_config = {
    "method": "bayes",
    "metric": {
        "name": "val_accuracy",
        "goal": "maximize"
    },
    "parameters": {
        "num_filters": {"values": [32, 64, 128]},
        "filter_multiplier": {"values": [0.5, 1, 2]},
        "kernel_size": {"values": [3, 5]},
        "activation": {"values": ["relu", "leaky_relu", "gelu", "silu", "mish"]},
        "dropout_rate": {"values": [0.2, 0.5, 0.7]},
        "use_batchnorm": {"values": [True, False]},
        "dense_units": {"values": [512, 1024]},
        "data_augmentation": {"values": [True, False]},
        "lr": {"values": [0.01, 0.001, 0.0001]},
    }
}

def make_run_name(config):
    return (
        f"nf_{config['num_filters']}_"
        f"ks_{config['kernel_size']}_"
        f"fm_{config['filter_multiplier']}_"
        f"act_{config['activation']}_"
        f"do_{config['dropout_rate']}_"
        f"bn_{config['use_batchnorm']}_"
        f"da_{config['data_augmentation']}_"
        f"lr_{config['lr']}_"
        f"du_{config['dense_units']}"
    )

def wandb_train(config=None):
    with wandb.init(config=config):
        config = wandb.config
        wandb.run.name = make_run_name(config)
        print(wandb.run.name)
        # Setup data
        train_loader, val_loader, test_loader = get_data_loaders('./partA/train.csv', './partA/val.csv', data_augmentation = config.data_augmentation, seed = 7)
        print("Data loading done")
        # Create model
        model = CNNModel(
            input_shape = (3, 224, 224),
            num_filters=config.num_filters,
            kernel_size=config.kernel_size,
            filter_multiplier=config.filter_multiplier,
            activation=config.activation,
            dropout_rate=config.dropout_rate,
            dense_units=config.dense_units,
            use_batchnorm=config.use_batchnorm,
        )
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model.to(device)
        print(f"Training model on {device}")
        train_model(model, train_loader, val_loader, epochs=10, lr=config.lr, device=device, wandb_logging = True)
