"""
HybridSN: Exploring 3-D–2-D CNN Feature Hierarchy for Hyperspectral Image Classification

Paper: https://ieeexplore.ieee.org/document/8736016
GitHub: https://github.com/Pancakerr/HybridSN
Venue: IEEE Geoscience and Remote Sensing Letters (GRSL)
Year: 2020

Architecture:
  1. Three 3D convolution layers (spectral-spatial feature extraction)
  2. One 2D convolution layer (spatial feature refinement)
  3. Fully connected classifier (256 → 128 → num_classes)

Adapted from Pancakerr/HybridSN — uses dummy forward pass at init to
dynamically compute intermediate feature dimensions instead of lazy init.
"""
from models.registry import register_model

import torch
import torch.nn as nn


class HybridSN(nn.Module):
    """HybridSN: 3D-2D CNN for Hyperspectral Image Classification.

    Input: [B, 1, C, H, W] (5D) where C = spectral bands
    Output: [B, num_classes]

    Args:
        num_classes (int): Number of output classes.
        bands (int): Number of input spectral bands.
        patch_size (int): Spatial patch size (H=W).
    """
    def __init__(self, num_classes, bands=30, patch_size=11):
        super().__init__()
        self.num_classes = num_classes
        self.bands = bands
        self.patch_size = patch_size

        # 3D convolution block
        self.conv1 = nn.Sequential(
            nn.Conv3d(1, 8, kernel_size=(7, 3, 3)),
            nn.ReLU(inplace=True))
        self.conv2 = nn.Sequential(
            nn.Conv3d(8, 16, kernel_size=(5, 3, 3)),
            nn.ReLU(inplace=True))
        self.conv3 = nn.Sequential(
            nn.Conv3d(16, 32, kernel_size=(3, 3, 3)),
            nn.ReLU(inplace=True))

        # Dynamically compute shape after 3D convs
        x1_shape = self._get_shape_after_3dconv()

        # 2D convolution block
        conv2d_in = x1_shape[1] * x1_shape[2]  # channels * spectral_remaining
        self.conv4 = nn.Sequential(
            nn.Conv2d(conv2d_in, 64, kernel_size=(3, 3)),
            nn.ReLU(inplace=True))

        # Dynamically compute shape after 2D conv
        fc_in = self._get_shape_after_2dconv(conv2d_in, x1_shape[3], x1_shape[4])

        # Classifier
        self.dense1 = nn.Sequential(
            nn.Linear(fc_in, 256),
            nn.ReLU(inplace=True),
            nn.Dropout(p=0.4))
        self.dense2 = nn.Sequential(
            nn.Linear(256, 128),
            nn.ReLU(inplace=True),
            nn.Dropout(p=0.4))
        self.dense3 = nn.Linear(128, num_classes)

    def _get_shape_after_3dconv(self):
        """Run dummy input through 3D convs to get output shape."""
        x = torch.zeros(1, 1, self.bands, self.patch_size, self.patch_size)
        with torch.no_grad():
            x = self.conv1(x)
            x = self.conv2(x)
            x = self.conv3(x)
        return x.shape  # [1, 32, spectral_rem, H_rem, W_rem]

    def _get_shape_after_2dconv(self, in_channels, h, w):
        """Run dummy input through 2D conv to get flattened size."""
        x = torch.zeros(1, in_channels, h, w)
        with torch.no_grad():
            x = self.conv4(x)
        return x.shape[1] * x.shape[2] * x.shape[3]

    def forward(self, x):
        # Handle both 4D [B, C, H, W] and 5D [B, 1, C, H, W] inputs
        if x.dim() == 4:
            x = x.unsqueeze(1)

        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        # Reshape: merge channels and spectral dims for 2D conv
        x = x.view(x.shape[0], x.shape[1] * x.shape[2], x.shape[3], x.shape[4])
        x = self.conv4(x)
        x = x.contiguous().view(x.shape[0], -1)
        x = self.dense1(x)
        x = self.dense2(x)
        x = self.dense3(x)
        return x


@register_model('HybridSN')
def hybridsn_model(pretrained: bool = False, **kwargs) -> HybridSN:
    """Constructs a HybridSN model."""
    return HybridSN(**kwargs)
