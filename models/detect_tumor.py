import torch
import torch.nn as nn
import torchvision.models as models
from torchvision.models import ResNet18_Weights


class TumorDetector(nn.Module):

    def __init__(self):
        super().__init__()

        self.model = models.resnet18(weights=ResNet18_Weights.DEFAULT)

        # freeze feature extractor
        for param in self.model.parameters():
            param.requires_grad = False

        # replace classifier
        self.model.fc = nn.Sequential(
            nn.Linear(512,128),
            nn.ReLU(),
            nn.Dropout(0.4),
            nn.Linear(128,2)
        )

    def forward(self,x):
        return self.model(x)