import torch
import torch.nn as nn

class DoubleConv(nn.Module):

    def __init__(self, in_channels, out_channels):
        super().__init__()

        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, 3, padding=1),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.conv(x)

class UNet(nn.Module):

    def __init__(self):
        super().__init__()

        # encoder
        self.down1 = DoubleConv(4, 64)
        self.down2 = DoubleConv(64, 128)

        self.pool = nn.MaxPool2d(2)

        # decoder
        self.up = nn.ConvTranspose2d(128, 64, kernel_size=2, stride=2)

        # output layer (3 classes)
        self.out = nn.Conv2d(64, 3, kernel_size=1)

    def forward(self, x):

        d1 = self.down1(x)
        d2 = self.down2(self.pool(d1))

        up = self.up(d2)

        return torch.sigmoid(self.out(up))
