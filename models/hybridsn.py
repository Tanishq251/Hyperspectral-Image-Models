"""
HybridSN: Exploring 3-D–2-D CNN Feature Hierarchy for Hyperspectral Image Classification

Paper: https://ieeexplore.ieee.org/document/8736016
GitHub: https://github.com/gokriznastic/HybridSN
Venue: IEEE Geoscience and Remote Sensing Letters (GRSL)
Year: 2020
"""

import torch.nn as nn
import torch.nn.functional as F
import torch
from torch.nn import init
from thop import profile
from .registry import register_model

class HybridSN(nn.Module):
    def __init__(self, num_classes, bands=30, patch_size=25, self_attention=False):
        super(HybridSN, self).__init__()
        
        self.self_attention = self_attention
        self.num_classes = num_classes

        # 3D卷积块
        self.block_1_3D = nn.Sequential(
            nn.Conv3d(in_channels=1, out_channels=8, kernel_size=(7, 3, 3), stride=1, padding=0),
            nn.ReLU(inplace=True),
            nn.Conv3d(in_channels=8, out_channels=16, kernel_size=(5, 3, 3), stride=1, padding=0),
            nn.ReLU(inplace=True),
            nn.Conv3d(in_channels=16, out_channels=32, kernel_size=(3, 3, 3), stride=1, padding=0),
            nn.ReLU(inplace=True)
        )

        # 2D卷积块 and classifier will be initialized lazily
        self.block_2_2D = None
        self.classifier = None
        self._initialized = False

    def _init_layers(self, conv2d_in_channels, spatial_size, device):
        """Initialize 2D conv and classifier based on actual dimensions"""
        self.block_2_2D = nn.Sequential(
            nn.Conv2d(in_channels=conv2d_in_channels, out_channels=64, kernel_size=(3, 3)),
            nn.ReLU(inplace=True)
        ).to(device)
        
        # spatial after 2D conv
        spatial_after_2d = max(1, spatial_size - 2)
        fc_in_features = 64 * spatial_after_2d * spatial_after_2d

        self.classifier = nn.Sequential(
            nn.Linear(in_features=fc_in_features, out_features=256),
            nn.Dropout(p=0.4),
            nn.Linear(in_features=256, out_features=128),
            nn.Dropout(p=0.4),
            nn.Linear(in_features=128, out_features=self.num_classes)
        ).to(device)
        
        self._initialized = True

    def forward(self, x):
        # Handle both 4D [B, C, H, W] and 5D [B, 1, C, H, W] inputs
        if x.dim() == 4:
            x = x.unsqueeze(1)
        
        y = self.block_1_3D(x)
        # y shape: [B, 32, bands_reduced, H_reduced, W_reduced]
        y = y.view(-1, y.shape[1] * y.shape[2], y.shape[3], y.shape[4])
        # y shape: [B, 32*bands_reduced, H_reduced, W_reduced]
        
        # Initialize 2D layers on first forward pass
        if not self._initialized:
            self._init_layers(y.shape[1], y.shape[2], x.device)
        
        if self.self_attention:
            y = self.spatial_attention_1(y) * y
        y = self.block_2_2D(y)
        if self.self_attention:
            y = self.spatial_attention_2(y) * y
        y = y.view(y.size(0), -1)
        y = self.classifier(y)
        return y


@register_model('HybridSN', self_attention=False)
def hybridsn_model(pretrained: bool = False, **kwargs) -> HybridSN:
    """Constructs a HybridSN model."""
    # bands and patch_size are now used for dynamic sizing
    return HybridSN(**kwargs)


if __name__ == '__main__':
    t = torch.randn(size=(1, 1, 30, 25, 25))
    print("input shape:", t.shape)
    net = HybridSN(num_classes=16, bands=30, patch_size=25)
    print("output shape:", net(t).shape)
    flops, params = profile(net, inputs=(t,))
    print('params', params)
    print('flops', flops)