import torch
import torch.nn as nn
import torch.nn.functional as F
from torchinfo import summary

class CNNModel(nn.Module):
    def __init__(self, input_shape, num_filters, kernel_size, filter_multiplier, activation, dropout_rate, use_batchnorm, dense_units):
        super(CNNModel, self).__init__()
        
        self.input_shape = input_shape
        self.layers = nn.ModuleList()
        self.num_classes = 10
        self.activation = activation
        self.multiplier = filter_multiplier # takes values like 0.5, 2, 1
        in_channels = input_shape[0]  
        
        for i in range(5):
            self.layers.append(nn.Conv2d(in_channels, num_filters, kernel_size = kernel_size, padding = kernel_size//2))
            if use_batchnorm:
                self.layers.append(nn.BatchNorm2d(num_filters))  # Apply batch normalization if enabled
            self.layers.append(self._get_activation())
            self.layers.append(nn.MaxPool2d(kernel_size = 2, stride = 2))
            self.layers.append(nn.Dropout(dropout_rate))
            in_channels = num_filters
            num_filters= int(num_filters*self.multiplier)
            
        # Flatten layer
        self.flatten = nn.Flatten()

        # Compute the output size after convolutions and pooling
        with torch.no_grad():
            dummy_input = torch.zeros(1, *input_shape)
            for layer in self.layers:
                dummy_input = layer(dummy_input)
            self.flattened_size = dummy_input.numel()

        # Fully connected layers
        self.fc1 = nn.Linear(self.flattened_size, dense_units)
        self.fc_activation = self._get_activation() 
        self.dropout_fc = nn.Dropout(dropout_rate)
        self.fc2 = nn.Linear(dense_units, self.num_classes)

    def _get_activation(self):
        if self.activation.lower() == 'relu':
            return nn.ReLU()
        elif self.activation.lower() == 'leaky_relu':
            return nn.LeakyReLU()
        elif self.activation.lower() == 'silu':
            return nn.SiLU()
        elif self.activation.lower() == 'mish':
            return nn.Mish()
        elif self.activation.lower() == 'gelu':
            return nn.GELU()
        else:
            return nn.ReLU()

    def forward(self, x):
        for layer in self.layers:
            x = layer(x)
        x = self.flatten(x)
        x = self.fc1(x)
        x = self.fc_activation(x)
        x = self.dropout_fc(x)
        x = self.fc2(x)
        return x

    def summary(self, batch_size=1):
        return summary(self, input_size=(batch_size, *self.input_shape), col_names=["input_size", "output_size", "num_params", "mult_adds"])
