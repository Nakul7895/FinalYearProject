import torch
import torch.nn as nn
import torchvision.models as models

class TumorClassifier(nn.Module):

    def __init__(self):
        super().__init__()

        self.model = models.resnet18(pretrained=True)

        self.model.fc = nn.Linear(512, 3)

    def forward(self, x):
        return self.model(x)