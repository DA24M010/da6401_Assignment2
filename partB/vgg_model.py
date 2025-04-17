import torch.nn as nn
from torchvision.models import vgg16

class FineTunedVGG(nn.Module):
    def __init__(self, num_classes = 10):
        super().__init__()
        # Load pre-trained VGG model
        base_model = vgg16(pretrained=True)
        # Freeze all layers of VGG
        for param in base_model.parameters():
            param.requires_grad = False
        self.vgg = base_model
        self.fc = nn.Sequential(
            nn.Linear(1000, 128),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(128, num_classes)
        )

    def forward(self, x):
        x = self.vgg(x)         # Output: [batch_size, 1000]
        x = self.fc(x)          # Output: [batch_size, 10]
        return x
