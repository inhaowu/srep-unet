import torch
import torch.nn as nn
import torch.nn.functional as F


class DoubleConv3D(nn.Module):
    def __init__(self, in_channels, out_channels, mid_channels=None):
        super().__init__()
        if not mid_channels:
            mid_channels = out_channels
        self.double_conv3d = nn.Sequential(
            nn.Conv3d(in_channels, mid_channels, kernel_size=3, padding=1),
            nn.BatchNorm3d(mid_channels),
            nn.ReLU(inplace=True),
            nn.Conv3d(mid_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm3d(out_channels),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        return self.double_conv3d(x)


class Down3D(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.maxpool_conv = nn.Sequential(nn.MaxPool3d(2), DoubleConv3D(in_channels, out_channels))

    def forward(self, x):
        return self.maxpool_conv(x)


class Up3D(nn.Module):
    def __init__(self, in_channels, out_channels, bilinear=True):
        super().__init__()
        if bilinear:
            self.up = nn.Upsample(scale_factor=2, mode="trilinear", align_corners=True)
            self.conv = DoubleConv3D(in_channels, out_channels, in_channels // 2)
        else:
            self.up = nn.ConvTranspose3d(in_channels, in_channels // 2, kernel_size=2, stride=2)
            self.conv = DoubleConv3D(in_channels, out_channels)

    def forward(self, x1, x2):
        x1 = self.up(x1)
        dz = x2.size(2) - x1.size(2)
        dy = x2.size(3) - x1.size(3)
        dx = x2.size(4) - x1.size(4)
        x1 = F.pad(x1, [dx // 2, dx - dx // 2, dy // 2, dy - dy // 2, dz // 2, dz - dz // 2])
        return self.conv(torch.cat([x2, x1], dim=1))


class OutConv3D(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.conv = nn.Conv3d(in_channels, out_channels, kernel_size=1)

    def forward(self, x):
        return self.conv(x)


class UNet3D(nn.Module):
    def __init__(self, in_channels=1, out_channels=2, fChannel=32, bilinear=True):
        super().__init__()
        factor = 2 if bilinear else 1
        self.inc = DoubleConv3D(in_channels, fChannel)
        self.down1 = Down3D(fChannel, fChannel * 2)
        self.down2 = Down3D(fChannel * 2, fChannel * 4)
        self.down3 = Down3D(fChannel * 4, fChannel * 8)
        self.down4 = Down3D(fChannel * 8, fChannel * 16 // factor)
        self.up1 = Up3D(fChannel * 16, fChannel * 8 // factor, bilinear)
        self.up2 = Up3D(fChannel * 8, fChannel * 4 // factor, bilinear)
        self.up3 = Up3D(fChannel * 4, fChannel * 2 // factor, bilinear)
        self.up4 = Up3D(fChannel * 2, fChannel, bilinear)
        self.outc = OutConv3D(fChannel, out_channels)

    def forward(self, x):
        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x5 = self.down4(x4)
        d = self.up1(x5, x4)
        d = self.up2(d, x3)
        d = self.up3(d, x2)
        d = self.up4(d, x1)
        return self.outc(d)
