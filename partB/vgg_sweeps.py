from vgg_model import FineTunedVGG
import torch
from train_vgg import train
from partA.scripts.data import get_data_loaders
import wandb

model = FineTunedVGG(num_classes = 10)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = model.to(device)

train_loader, val_loader, test_loader, _ = get_data_loaders('/kaggle/working/train.csv', '/kaggle/working/val.csv', seed= 7)train(model, train_loader, val_loader, test_loader, device, 10, True)

with wandb.init(project="DA6401 Assignments", name="VGG 16 Fine tuned"):
    train(model, train_loader, val_loader, test_loader, device, 10, True)