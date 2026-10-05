#!/usr/bin/env python3
"""33_modelB_unet.py -- Scheme B U-Net, first version (v1), docs/PLAN.md v7 §6.

4 downsampling stages; encoder channels 32->64->128->256, bottleneck 512; each level 2 x (3x3 conv + BatchNorm + ReLU);
downsampling 2x2 max pooling; upsampling 2x2 transposed conv; skip connections concatenated along channels; output 1x1 conv + ReLU (h >= 0).
Input (B, 13, 208, 256), output cropped back to (B, 195, 255).
"""
import torch
import torch.nn as nn

NY, NX = 195, 255


def double_conv(cin, cout):
    return nn.Sequential(
        nn.Conv2d(cin, cout, 3, padding=1, bias=False), nn.BatchNorm2d(cout), nn.ReLU(inplace=True),
        nn.Conv2d(cout, cout, 3, padding=1, bias=False), nn.BatchNorm2d(cout), nn.ReLU(inplace=True))


class UNetV1(nn.Module):
    def __init__(self, in_ch=13, widths=(32, 64, 128, 256), bottom=512):
        super().__init__()
        self.enc = nn.ModuleList(); c = in_ch
        for w in widths:
            self.enc.append(double_conv(c, w)); c = w
        self.pool = nn.MaxPool2d(2)
        self.bottom = double_conv(c, bottom); c = bottom
        self.up = nn.ModuleList(); self.dec = nn.ModuleList()
        for w in reversed(widths):
            self.up.append(nn.ConvTranspose2d(c, w, 2, stride=2))
            self.dec.append(double_conv(2 * w, w)); c = w
        self.head = nn.Sequential(nn.Conv2d(c, 1, 1), nn.ReLU())

    def forward(self, x):
        skips = []
        for e in self.enc:
            x = e(x); skips.append(x); x = self.pool(x)
        x = self.bottom(x)
        for up, dec, s in zip(self.up, self.dec, reversed(skips)):
            x = dec(torch.cat([up(x), s], dim=1))
        return self.head(x)[:, 0, :NY, :NX]


def n_params(model):
    return sum(p.numel() for p in model.parameters())


if __name__ == '__main__':
    m = UNetV1()
    print('UNetV1 parameters: %d' % n_params(m))
    print(m(torch.zeros(1, 13, 208, 256)).shape)
