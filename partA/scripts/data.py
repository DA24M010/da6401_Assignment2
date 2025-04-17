import torch
import pandas as pd
from torch.utils.data import Dataset
from torchvision import transforms
from PIL import Image

from torch.utils.data import DataLoader, random_split
from sklearn.model_selection import StratifiedShuffleSplit
import torch
import pandas as pd
from torch.utils.data import Dataset
from torchvision import transforms
from PIL import Image

class N12KDATA(Dataset):
    def __init__(self, csv_data, image_size=(224, 224), normalize=True, data_augmentation=False):
        self.data = csv_data
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

def get_data_loaders(train_csv_path, test_csv_path, val_split_ratio = 0.2, normalize = True, data_augmentation = False, seed = 10):
    # Define batch size
    BATCH_SIZE = 64

    # Load the full train CSV
    train_csv = pd.read_csv(train_csv_path)
    labels = train_csv['label'].values

    # Stratified Split such that each class is equally represented in the validation data
    splitter = StratifiedShuffleSplit(n_splits=1, test_size=val_split_ratio, random_state=seed)
    train_idx, val_idx = next(splitter.split(train_csv, labels))
    
    train_df = train_csv.iloc[train_idx].reset_index(drop=True)
    val_df = train_csv.iloc[val_idx].reset_index(drop=True)

    print(val_df['label'].value_counts())

    # Prepare the Datasets
    train_dataset = N12KDATA(train_df, normalize = normalize, data_augmentation=data_augmentation)
    val_dataset = N12KDATA(val_df, normalize = normalize, data_augmentation=False)
    test_data = pd.read_csv(test_csv_path)
    test_dataset = N12KDATA(test_data, normalize = normalize, data_augmentation=False) 

    # Prepare train, val, test loaders
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=4)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=4)
    test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=4)

    print(f"Train dataset size: {len(train_dataset)}")
    print(f"Validation dataset size: {len(val_dataset)}")
    print(f"Test dataset size: {len(test_dataset)}")

    return train_loader, val_loader, test_loader, train_dataset.label_map
