import torch.nn as nn
from torchvision.models import vgg16

# Applying strategy 1
# Freeze all convolutional layers, add new FC layers after conv output
class FineTunedVGG_Strategy1(nn.Module):
    def __init__(self, num_classes=10):
        super().__init__()
        # Load pre-trained VGG model
        base_model = vgg16(pretrained=True)
        
        # Freeze all convolutional layers
        for param in base_model.features.parameters():
            param.requires_grad = False
            
        # Extract the feature extractor and pooling
        self.features = base_model.features  
        self.avgpool = base_model.avgpool
        
        # Create new classifier that connects to conv output
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(512 * 7 * 7, 4096),
            nn.ReLU(True),
            nn.Dropout(0.5),
            nn.Linear(4096, 1024),
            nn.ReLU(True),
            nn.Dropout(0.3),
            nn.Linear(1024, num_classes)
        )
        
    def forward(self, x):
        x = self.features(x)   
        x = self.avgpool(x)       
        x = self.classifier(x)  
        return x

# Applying strategy 2
# Freeze all convolutional layers and dense layers, add new FC layers after dense 1000 output
class FineTunedVGG_Strategy2(nn.Module):
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

# Applying strategy 3
# Freeze first 3 conv blocks, fine-tune last 2 blocks and FC layers
class FineTunedVGG_Strategy3(nn.Module):
    def __init__(self, num_classes=10):
        super().__init__()
        # Load pre-trained VGG model
        model = vgg16(pretrained=True)
        
        # VGG16 features structure - freeze first 3 blocks (layers 0-16)
        # Block 1: layers 0-4 (Conv-ReLU-Conv-ReLU-MaxPool)
        # Block 2: layers 5-9 (Conv-ReLU-Conv-ReLU-MaxPool)
        # Block 3: layers 10-16 (Conv-ReLU-Conv-ReLU-Conv-ReLU-MaxPool)
        for i in range(17):
            for param in model.features[i].parameters():
                param.requires_grad = False
                
        # Replace the final classifier layer
        num_features = model.classifier[6].in_features
        model.classifier[6] = nn.Linear(num_features, num_classes)
        
        self.model = model
        
    def forward(self, x):
        return self.model(x)
