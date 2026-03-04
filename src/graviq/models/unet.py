import torch
import torch.nn as nn
import torch.nn.functional as F


class DoubleConv(nn.Module):
    """(Conv2d -> BatchNorm -> ReLU) x 2 + optional spatial dropout"""

    def __init__(self, in_channels, out_channels, dropout: float = 0.0):
        super().__init__()
        layers = [
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        ]
        if dropout > 0.0:
            # Spatial dropout to regularize feature maps and reduce overfitting.
            layers.append(nn.Dropout2d(p=dropout))
        self.double_conv = nn.Sequential(*layers)

    def forward(self, x):
        return self.double_conv(x)


class Down(nn.Module):
    """Downscaling with maxpool then double conv"""

    def __init__(self, in_channels, out_channels, dropout: float = 0.0):
        super().__init__()
        self.maxpool_conv = nn.Sequential(
            nn.MaxPool2d(2),
            DoubleConv(in_channels, out_channels, dropout=dropout),
        )

    def forward(self, x):
        return self.maxpool_conv(x)


class Up(nn.Module):
    """Upscaling then double conv"""

    def __init__(self, in_channels, out_channels, dropout: float = 0.0):
        super().__init__()
        self.up = nn.ConvTranspose2d(
            in_channels, in_channels // 2, kernel_size=2, stride=2
        )
        self.conv = DoubleConv(in_channels, out_channels, dropout=dropout)

    def forward(self, x1, x2):
        x1 = self.up(x1)

        # Pad x1 to match x2 size if needed
        diffY = x2.size()[2] - x1.size()[2]
        diffX = x2.size()[3] - x1.size()[3]
        x1 = F.pad(
            x1,
            [diffX // 2, diffX - diffX // 2, diffY // 2, diffY - diffY // 2],
        )

        # Concatenate along channel axis
        x = torch.cat([x2, x1], dim=1)
        return self.conv(x)


class UNet(nn.Module):
    """
    U-Net architecture for tunnel segmentation.

    Input: (B, 1, H, W) - density or Gzz-like grid
    Output: (B, 1, H, W) - tunnel probability mask
    """

    def __init__(
        self,
        in_channels: int = 1,
        out_channels: int = 1,
        features=None,
        dropout: float = 0.1,
    ):
        super().__init__()
        if features is None:
            features = [32, 64, 128, 256]

        self.in_channels = in_channels
        self.out_channels = out_channels

        # Encoder (downsampling)
        self.inc = DoubleConv(in_channels, features[0], dropout=dropout)
        self.down1 = Down(features[0], features[1], dropout=dropout)
        self.down2 = Down(features[1], features[2], dropout=dropout)
        self.down3 = Down(features[2], features[3], dropout=dropout)

        # Decoder (upsampling)
        self.up1 = Up(features[3], features[2], dropout=dropout)
        self.up2 = Up(features[2], features[1], dropout=dropout)
        self.up3 = Up(features[1], features[0], dropout=dropout)

        # Output layer
        self.outc = nn.Conv2d(features[0], out_channels, kernel_size=1)

    def forward(self, x):
        # Encoder
        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)

        # Decoder with skip connections
        x = self.up1(x4, x3)
        x = self.up2(x, x2)
        x = self.up3(x, x1)

        # Output
        logits = self.outc(x)
        return logits


def get_model(device='cuda' if torch.cuda.is_available() else 'cpu'):
    """Initialize model and move to device"""
    model = UNet(in_channels=1, out_channels=1, dropout=0.1)
    model = model.to(device)
    return model
