import argparse
import torch
import wandb
from vgg_model import FineTunedVGG_Strategy2
from train_vgg import train
from partA.scripts.data import get_data_loaders

def main(project, entity):
    # Initialize model
    model = FineTunedVGG_Strategy2(num_classes=10)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)

    # Load data
    train_loader, val_loader, test_loader, _ = get_data_loaders(
        '/kaggle/working/train.csv',
        '/kaggle/working/val.csv',
        seed=7
    )

    # Log to wandb
    with wandb.init(project=project, entity=entity, name="VGG16 Fine-tuned"):
        train(model, train_loader, val_loader, test_loader, device, num_epochs=10, wandb_logging=True)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--project", default="DA6401_Assignments", help="WandB project name")
    parser.add_argument("--entity", default="da24m010-indian-institute-of-technology-madras", help="WandB entity name")
    args = parser.parse_args()

    main(args.project, args.entity)
