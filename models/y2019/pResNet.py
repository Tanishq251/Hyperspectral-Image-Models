"""pResNet: Deep Pyramidal Residual Networks for Spectral-Spatial HSI Classification

Paper: Deep Pyramidal Residual Networks for Spectral-Spatial Hyperspectral Image Classification
Authors: Paoletti et al.
Year: 2019
Venue: IEEE TGRS
Source: https://github.com/Candy-CY/Hyperspectral-Image-Classification-Models (2018/DPRN)

Adapted to the unified library API. Changes vs. the original reference code:
  * The shortcut AvgPool2d now uses ceil_mode=True. The original floors the
    spatial size (11 -> 5) while the stride-2 conv ceils it (11 -> 6), which
    crashes on odd patch sizes such as 11.
  * The zero-padding tensor for channel matching is created on the input's
    device instead of a hardcoded .cuda().
  * The final pooling is AdaptiveAvgPool2d(1) so any patch size works without
    hand-computing `avgpoosize`.
"""

import math

import torch
import torch.nn as nn

from models.registry import register_model


def conv3x3(in_planes, out_planes, stride=1):
    """3x3 convolution with padding."""
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                     padding=1, bias=False)


def _pad_shortcut(out, shortcut):
    """Zero-pad the shortcut's channels to match `out`, on the correct device."""
    batch_size = out.size(0)
    residual_channel = out.size(1)
    shortcut_channel = shortcut.size(1)
    if residual_channel != shortcut_channel:
        padding = torch.zeros(
            batch_size, residual_channel - shortcut_channel,
            shortcut.size(2), shortcut.size(3),
            device=out.device, dtype=out.dtype,
        )
        return out + torch.cat((shortcut, padding), 1)
    return out + shortcut


class BasicBlock(nn.Module):
    outchannel_ratio = 1

    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super().__init__()
        self.bn1 = nn.BatchNorm2d(inplanes)
        self.conv1 = conv3x3(inplanes, planes, stride)
        self.bn2 = nn.BatchNorm2d(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3(planes, planes)
        self.bn3 = nn.BatchNorm2d(planes)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        out = self.bn1(x)
        out = self.conv1(out)
        out = self.bn2(out)
        out = self.relu(out)
        out = self.conv2(out)
        out = self.bn3(out)
        shortcut = self.downsample(x) if self.downsample is not None else x
        return _pad_shortcut(out, shortcut)


class Bottleneck(nn.Module):
    outchannel_ratio = 4

    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super().__init__()
        self.bn1 = nn.BatchNorm2d(inplanes)
        self.conv1 = nn.Conv2d(inplanes, planes, kernel_size=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=stride,
                               padding=1, bias=False)
        self.bn3 = nn.BatchNorm2d(planes)
        self.conv3 = nn.Conv2d(planes, planes * Bottleneck.outchannel_ratio,
                               kernel_size=1, bias=False)
        self.bn4 = nn.BatchNorm2d(planes * Bottleneck.outchannel_ratio)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        out = self.bn1(x)
        out = self.conv1(out)
        out = self.bn2(out)
        out = self.relu(out)
        out = self.conv2(out)
        out = self.bn3(out)
        out = self.relu(out)
        out = self.conv3(out)
        out = self.bn4(out)
        shortcut = self.downsample(x) if self.downsample is not None else x
        return _pad_shortcut(out, shortcut)


class pResNet(nn.Module):
    """Pyramidal residual network operating on (B, bands, H, W) patches."""

    def __init__(self, depth, alpha, num_classes, n_bands, inplanes=16,
                 bottleneck=False):
        super().__init__()
        self.inplanes = inplanes
        if bottleneck:
            n = (depth - 2) // 9
            block = Bottleneck
        else:
            n = (depth - 2) // 6
            block = BasicBlock

        self.addrate = alpha / (3 * n * 1.0)
        self.input_featuremap_dim = self.inplanes
        self.conv1 = nn.Conv2d(n_bands, self.input_featuremap_dim,
                               kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(self.input_featuremap_dim)

        self.featuremap_dim = self.input_featuremap_dim
        self.layer1 = self.pyramidal_make_layer(block, n)
        self.layer2 = self.pyramidal_make_layer(block, n, stride=2)
        self.layer3 = self.pyramidal_make_layer(block, n, stride=2)

        self.final_featuremap_dim = self.input_featuremap_dim
        self.bn_final = nn.BatchNorm2d(self.final_featuremap_dim)
        self.relu_final = nn.ReLU(inplace=True)
        # Adaptive pooling -> independent of the input patch size.
        self.avgpool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Linear(self.final_featuremap_dim, num_classes)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def pyramidal_make_layer(self, block, block_depth, stride=1):
        downsample = None
        if stride != 1 or self.inplanes != int(round(self.featuremap_dim)) * block.outchannel_ratio:
            # ceil_mode=True keeps the shortcut aligned with the stride-2 conv
            # for odd spatial sizes (e.g. patch 11 -> 6, not 5).
            downsample = nn.AvgPool2d((2, 2), stride=(2, 2), ceil_mode=True)

        layers = []
        self.featuremap_dim = self.featuremap_dim + self.addrate
        layers.append(block(self.input_featuremap_dim,
                            int(round(self.featuremap_dim)), stride, downsample))
        for _ in range(1, block_depth):
            temp_featuremap_dim = self.featuremap_dim + self.addrate
            layers.append(block(int(round(self.featuremap_dim)) * block.outchannel_ratio,
                                int(round(temp_featuremap_dim)), 1))
            self.featuremap_dim = temp_featuremap_dim
        self.input_featuremap_dim = int(round(self.featuremap_dim)) * block.outchannel_ratio
        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.bn_final(x)
        x = self.relu_final(x)
        x = self.avgpool(x)
        x = x.view(x.size(0), -1)
        return self.fc(x)


@register_model('pResNet', expects_4d=True, depth=32, alpha=48, inplanes=16,
                bottleneck=False)
def presnet(num_classes, bands, patch_size=11, depth=32, alpha=48,
            inplanes=16, bottleneck=False, **kwargs):
    """Factory for pResNet.

    Args:
        num_classes: number of classes.
        bands: number of spectral bands (after PCA / band selection).
        patch_size: unused (adaptive pooling makes the head size-agnostic).
        depth: total network depth; must satisfy (depth - 2) % 6 == 0 for
            BasicBlock, or (depth - 2) % 9 == 0 for Bottleneck.
        alpha: pyramidal widening factor.
        inplanes: initial feature-map width.
        bottleneck: use Bottleneck blocks instead of BasicBlock.
    """
    return pResNet(depth=depth, alpha=alpha, num_classes=num_classes,
                   n_bands=bands, inplanes=inplanes, bottleneck=bottleneck)
