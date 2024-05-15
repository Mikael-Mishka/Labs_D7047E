import torch
from torch import nn
from torchvision.models import resnet18
from torchvision.models.resnet import ResNet18_Weights

class OUR_Resnet18(nn.Module):
    def __init__(self, *args, **kwargs):
        # Initialization code
        super().__init__(*args, **kwargs)

        self.resnet18_model = resnet18(weights = ResNet18_Weights.DEFAULT)
        self.TF = ResNet18_Weights.DEFAULT.transforms()

        feature_out = self.resnet18_model.fc.in_features
        self.resnet18_model.fc = nn.Linear(feature_out, 2)
    def forward(self, x, *args, **kwargs):
        return self.resnet18_model.forward(x)

    def get_transforms(self):
        return self.TF
        # I added this here, grab resnet transforms here and return them :)



torch.save(OUR_Resnet18(), 'ammonia_resnet18.pth')
#gvghh