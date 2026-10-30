import torch
import torch.nn as nn
import torch.nn.functional as F


class ImpConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1):
        super().__init__()

        self.block = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size, padding=kernel_size // 2, stride=stride, bias=False),
            nn.InstanceNorm2d(out_channels, affine=True),
            nn.LeakyReLU(negative_slope=1e-2, inplace=True),
        )

    def forward(self, x):
        x = self.block(x)
        return x

class ContextModule(nn.Module):
    def __init__(self, channels, p_drop=0.3):
        super().__init__()
        self.block = nn.Sequential(
            nn.InstanceNorm2d(channels, affine=True),
            nn.LeakyReLU(negative_slope=1e-2, inplace=True),
            nn.Conv2d(channels, channels, kernel_size=3, padding=1, bias=False),
            nn.Dropout(p=p_drop),
            nn.InstanceNorm2d(channels, affine=True),
            nn.LeakyReLU(negative_slope=1e-2, inplace=True),
            nn.Conv2d(channels, channels, kernel_size=3, padding=1, bias=False),
        )

    def forward(self, x):
        x = self.block(x)
        return x

class UpscaleBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.block = nn.Sequential(
            nn.Upsample(scale_factor=2, mode='bilinear'),
            ImpConvBlock(in_channels, out_channels),
        )

    def forward(self, x):
        x = self.block(x)
        return x

class LocalisationBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.block = nn.Sequential(
            ImpConvBlock(in_channels, out_channels, kernel_size=3),
            ImpConvBlock(out_channels, out_channels, kernel_size=1),
        )

    def forward(self, x):
        x = self.block(x)
        return x

class ImpDownBlock(nn.Module):
    def __init__(self, in_channels, out_channels, stride=2, p_drop=0.3):
        super().__init__()
        self.imp_conv = ImpConvBlock(in_channels, out_channels, stride=stride)
        self.context = ContextModule(out_channels, p_drop)

    def forward(self, x):
        x = self.imp_conv(x)
        x = x + self.context(x)
        return x

class ImpUpBlock(nn.Module):
    def __init__(self, in_channels, skip_channel, out_channels, final=False):
        super().__init__()

        self.up = UpscaleBlock(in_channels, out_channels)
        cat_channels = out_channels + skip_channel
        if final:
            self.loc = ImpConvBlock(cat_channels, out_channels)
        else:
            self.loc = LocalisationBlock(cat_channels, out_channels)

    def forward(self, x, skip):
        x = self.up(x)
        x = torch.cat([skip, x], dim=1)
        x = self.loc(x)
        return x


class ImprovedUNet(nn.Module):
    def __init__(self, in_channels = 1, num_classes = 6, base_channels = 32, softmax= False):
        super().__init__()
        self.softmax = softmax
        self.enc1 = ImpDownBlock(in_channels, base_channels, stride=1)
        self.enc2 = ImpDownBlock(base_channels, base_channels*2, stride=2)
        self.enc3 = ImpDownBlock(base_channels*2, base_channels*4, stride=2)
        self.enc4 = ImpDownBlock(base_channels*4, base_channels*8, stride=2)
        self.enc5 = ImpDownBlock(base_channels*8, base_channels*16, stride=2)

        self.dec1 = ImpUpBlock(base_channels*16, base_channels*8, base_channels*8)
        self.dec2 = ImpUpBlock(base_channels*8, base_channels*4, base_channels*4)
        self.dec3 = ImpUpBlock(base_channels*4, base_channels*2, base_channels*2)
        self.dec4 = ImpUpBlock(base_channels*2, base_channels, base_channels, final=True)

        self.seg3 = nn.Conv2d(base_channels*4, num_classes, 1)
        self.seg2 = nn.Conv2d(base_channels*2, num_classes, 1)
        self.seg1 = nn.Conv2d(base_channels, num_classes, 1)
        self.upscale = nn.Upsample(scale_factor=2, mode='bilinear')

    def forward(self, x):

        x1 = self.enc1(x)
        x2 = self.enc2(x1)
        x3 = self.enc3(x2)
        x4 = self.enc4(x3)
        x5 = self.enc5(x4)

        d1 = self.dec1(x5, x4)
        d2 = self.dec2(d1, x3)
        d3 = self.dec3(d2, x2)
        d4 = self.dec4(d3, x1)

        out = self.seg3(d2)
        out = self.upscale(out) + self.seg2(d3)
        out = self.upscale(out) + self.seg1(d4)

        if self.softmax:
            out = F.softmax(out, dim=1)

        return out

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
    def __init__(self, in_channels = 1, num_classes = 6, base_channels = 32):
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


def dice_loss(prediction, target, epsilon=1e-6):

    prediction_prob = torch.softmax(prediction, dim=1)
    dimensions = (0, 2, 3)
    intersection = torch.sum(prediction_prob * target, dimensions)
    union = torch.sum(prediction_prob, dimensions) + torch.sum(target, dimensions)
    dice = (2.0 * intersection + epsilon) / (union + epsilon)
    dice_mean = dice.mean()

    return 1.0 - dice_mean

def dice_counts(prediction, target):
    c = target.shape[1]
    index = prediction.argmax(dim=1)
    pred = (index.unsqueeze(1) == torch.arange(c, device=index.device).view(1, -1, 1, 1)).float()
    dims = (0, 2, 3)
    inter = (pred * target).sum(dims)
    union = pred.sum(dims) + target.sum(dims)
    return inter, union

def combined_loss(prediction, target):
    ce = F.cross_entropy(prediction, target.argmax(dim=1))
    dice = dice_loss(prediction, target)
    return ce + dice
