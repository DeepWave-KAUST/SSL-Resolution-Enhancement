import torch
import torch.nn as nn
import torch.nn.functional as F

class UNet(nn.Module):
    def __init__(self, in_channels=1, out_channels=1):
        super(UNet, self).__init__()

        self.en_block1 = nn.Sequential(
            Basicblock(in_channels, 48),
            Basicblock(48, 48),
            nn.MaxPool2d(2))

        self.en_block2 = nn.Sequential(
            Basicblock(48, 48),
            nn.MaxPool2d(2))

        self.en_block3 = nn.Sequential(
            Basicblock(48, 48),
            nn.MaxPool2d(2))

        self.en_block4 = nn.Sequential(
            Basicblock(48, 48),
            nn.MaxPool2d(2))

        self.en_block5 = nn.Sequential(
            Basicblock(48, 48),
            nn.MaxPool2d(2),
            Basicblock(48, 48),
            nn.Upsample(scale_factor=2, mode='nearest'))

        self.de_block1 = nn.Sequential(
            Basicblock(96, 96),
            Basicblock(96, 96),
            nn.Upsample(scale_factor=2, mode='nearest'))

        self.de_block2 = nn.Sequential(
            Basicblock(144, 96),
            Basicblock(96, 96),
            nn.Upsample(scale_factor=2, mode='nearest'))

        self.de_block3 = nn.Sequential(
            Basicblock(144, 96),
            Basicblock(96, 96),
            nn.Upsample(scale_factor=2, mode='nearest'))

        self.de_block4 = nn.Sequential(
            Basicblock(144, 96),
            Basicblock(96, 96),
            nn.Upsample(scale_factor=2, mode='nearest'))

        self.de_block5 = nn.Sequential(
            Basicblock(96 + in_channels, 64),
            Basicblock(64, 32))

        self.out = nn.Conv2d(32, out_channels, 3, padding=1, bias=True)

        self.padder_size = 2 ** 5

    def forward(self, x):
        B, C, H, W = x.shape
        x = self.check_image_size(x)

        pool1 = self.en_block1(x)
        pool2 = self.en_block2(pool1)
        pool3 = self.en_block3(pool2)
        pool4 = self.en_block4(pool3)
        upsample5 = self.en_block5(pool4)

        concat5 = torch.cat((upsample5, pool4), dim=1)
        upsample4 = self.de_block1(concat5)
        concat4 = torch.cat((upsample4, pool3), dim=1)
        upsample3 = self.de_block2(concat4)
        concat3 = torch.cat((upsample3, pool2), dim=1)
        upsample2 = self.de_block3(concat3)
        concat2 = torch.cat((upsample2, pool1), dim=1)
        upsample1 = self.de_block4(concat2)
        concat1 = torch.cat((upsample1, x), dim=1)
        out = self.de_block5(concat1)
        out = self.out(out)

        return out[:, :, :H, :W]

    def check_image_size(self, x):
        _, _, h, w = x.size()
        mod_pad_h = (self.padder_size - h % self.padder_size) % self.padder_size
        mod_pad_w = (self.padder_size - w % self.padder_size) % self.padder_size
        x = F.pad(x, (0, mod_pad_w, 0, mod_pad_h), mode='replicate')
        return x

class Basicblock(nn.Module):
    def __init__(self, in_channels, out_channels, ks=3, stride=1, padding=1):
        super().__init__()
        self.ks = ks
        self.stride = stride
        self.padding = padding

        self.block = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=self.ks, padding=self.padding),
            nn.LeakyReLU(inplace=True, negative_slope=0.1)
        )

    def forward(self, x):
        return self.block(x)
