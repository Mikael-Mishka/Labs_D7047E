import torch
import torch.nn as nn

# Imports ToTensor
from torchvision.transforms import ToTensor

# Imports random split
from torch.utils.data import random_split, DataLoader

# Import transforms
import torchvision.transforms as transforms

class CNN_model(nn.Module):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        # If cuda is available, use it
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'

        self.conv_layers = nn.Sequential(
            nn.Conv2d(1, 32, 5, (1, 1), padding=1, device=self.device),
            nn.LeakyReLU, #nn.Tanh(),
            nn.MaxPool2d(5, 1),
            nn.BatchNorm2d(32),
            nn.Conv2d(32, 64, 5, (1, 1), padding=1, device=self.device),
            nn.LeakyReLU, #nn.Tanh(),
            nn.MaxPool2d(5, 1),
            nn.BatchNorm2d(64),
            nn.Conv2d(64, 64, 3, (1, 1), padding=1, device=self.device),
            nn.LeakyReLU, #nn.Tanh(),
            nn.MaxPool2d(5, 1),
            nn.BatchNorm2d(64),
            nn.Conv2d(64, 32, 3, (1, 1), padding=1, device=self.device),
            nn.LeakyReLU, #nn.Tanh(),
            nn.MaxPool2d(5, 1),
            nn.BatchNorm2d(32),
            nn.Conv2d(32, 16, 3, (1, 1), padding=1, device=self.device),
            nn.LeakyReLU, #nn.Tanh(),
            nn.MaxPool2d(5, 1),
            nn.BatchNorm2d(16),
            nn.Conv2d(16, 1, 3, (1, 1), padding=1, device=self.device),
            nn.LeakyReLU, #nn.Tanh(),
        )
        
        self.conv_layers.to(self.device)

        image_width, image_height = kwargs["image_shape"]

        # Dummy pass to get the ouput tensor size of the conv layers, adjusted with image size
        dummy = torch.randn(1, 1, image_width, image_height).to(self.device)

        # Get the output size of the conv layers
        conv_out = self.conv_layers(dummy)

        num_features = Flatten().forward(conv_out).shape[1]
        
        self.internal_model = nn.Sequential(
            self.conv_layers,
            Flatten(),
            nn.Linear(num_features, 200),
            nn.LeakyReLU, #nn.Tanh(),
            nn.Dropout(0.2),
            nn.Linear(200, 200),
            nn.LeakyReLU, #nn.Tanh(),
            nn.Dropout(0.2),
            nn.Linear(200, 3),
        )

        self.internal_model.to(self.device)
        
    def forward(self, x):
        return self.internal_model(x)

class Flatten(nn.Module):
    def forward(self, x):
        return x.view(x.size(0), -1)

