import torch
import pandas as pd
from torch.utils.data import Dataset
from torchvision import transforms
from PIL import Image

class N12KDATA(Dataset):
    def __init__(self, csv_path, image_size=(224, 224), normalize=True, data_augmentation=False):
        self.data = pd.read_csv(csv_path)
        self.image_size = image_size

        # Label Encoding (Mapping labels to indices)
        self.label_map = {label: idx for idx, label in enumerate(sorted(self.data["label"].unique()))}
        self.data["encoded_label"] = self.data["label"].map(self.label_map)

        # Define transformations
        self.transform = self._get_transforms(normalize, data_augmentation)

    def _get_transforms(self, normalize, data_augmentation):
        transform_list = []

        if data_augmentation:
            transform_list.extend([
                transforms.RandomHorizontalFlip(),    # Randomly flip images
                transforms.RandomRotation(10),       # Rotate within ±10 degrees
                transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1), # Adjust color properties
                transforms.RandomResizedCrop(self.image_size, scale=(0.8, 1.0)) # Random crop and resize
            ])
        else:
            transform_list.append(transforms.Resize(self.image_size))  # Just resize for non-augmented images
        
        transform_list.append(transforms.ToTensor())

        if normalize:
            transform_list.append(transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]))

        return transforms.Compose(transform_list)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        row = self.data.iloc[idx]
        image_path = row["path"]
        label = row["encoded_label"]

        # Load image
        image = Image.open(image_path).convert("RGB")
        image = self.transform(image)

        return image, torch.tensor(label, dtype=torch.long)