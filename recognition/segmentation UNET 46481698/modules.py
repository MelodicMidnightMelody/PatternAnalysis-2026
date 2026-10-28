import torch
import torch.nn as nn

class DoubleConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        x = self.block(x)
        return x

class DownBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.d_block = nn.Sequential(
            nn.MaxPool2d(2),
            DoubleConvBlock(in_channels, out_channels)
        )

    def forward(self, x):
        x = self.d_block(x)
        return x

class UpBlock(nn.Module):
    def __init__(self, in_channels, skip_channels, out_channels):
        super().__init__()
        self.up = nn.ConvTranspose2d(in_channels, in_channels//2, kernel_size=2, stride=2)
        self.conv = DoubleConvBlock(in_channels//2 + skip_channels, out_channels)

    def forward(self, x, skip):
        x = self.up(x)
        diff_y = skip.size(2) - x.size(2)
        diff_x = skip.size(3) - x.size(3)
        x = nn.functional.pad(x, [diff_x // 2, diff_x - diff_x // 2,
                      diff_y // 2, diff_y - diff_y // 2])

        x = self.conv(torch.cat([skip, x], dim=1))
        return x

class UNet2D(nn.Module):
    def __init__(self, in_channels = 1, num_classes = 2, base_channels = 32):
        super().__init__()

        #Encoder
        self.double_conv = DoubleConvBlock(in_channels, base_channels)
        self.down1 = DownBlock(base_channels, base_channels*2)
        self.down2 = DownBlock(base_channels*2, base_channels*4)
        self.down3 = DownBlock(base_channels*4, base_channels*8)
        self.down4 = DownBlock(base_channels*8, base_channels*16)

        #Decoder
        self.up1 = UpBlock(base_channels*16, base_channels*8, base_channels*8)
        self.up2 = UpBlock(base_channels*8, base_channels*4, base_channels*4)
        self.up3 = UpBlock(base_channels*4, base_channels*2, base_channels*2)
        self.up4 = UpBlock(base_channels*2, base_channels, base_channels)

        self.outc = nn.Conv2d(base_channels, num_classes, kernel_size=1)

    def forward(self, x):
        x1 = self.double_conv(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x5 = self.down4(x4)

        x = self.up1(x5, x4)
        x = self.up2(x, x3)
        x = self.up3(x, x2)
        x = self.up4(x, x1)

        logits = self.outc(x)
        return logits

def dice_loss(prediction, target, epsilon=1e-8):

    prediction_prob = torch.sigmoid(prediction)
    dimensions = (0, 2, 3)
    intersection = torch.sum(prediction_prob * target, dimensions)
    union = torch.sum(prediction_prob + target, dimensions)
    dice = (2.0 * intersection + epsilon) / (union + epsilon)
    dice_mean = dice.mean()

    return 1.0 - dice_mean

def dice_coefficient(prediction, target, epsilon=1e-8):

    prediction_prob = torch.sigmoid(prediction)
    binary_prediction = (prediction_prob > 0.5).float()

    dimensions = (0, 2, 3)
    intersection = torch.sum(binary_prediction * target, dimensions)
    union = torch.sum(binary_prediction + target, dimensions)
    dice = (2.0 * intersection + epsilon) / (union + epsilon)

    return dice
