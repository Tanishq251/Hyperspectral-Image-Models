from models.registry import register_model
"""
MMFormer: Macro-Micro Transformer for Small-Sample Classification of Mars Hyperspectral Image

Paper: IEEE Transactions on Geoscience and Remote Sensing, 2026
GitHub: https://github.com/RUI-FT/TGRS_2026_MMFormer
Venue: IEEE TGRS
Year: 2026

Authors: Tingrui Feng, Yali Wang, Chuan Fu, Bo Du, and Fulin Luo
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class CenterBiasAvgPool(nn.Module):
    def __init__(self, kernel_size=13, center_weight=0.5):
        super().__init__()
        self.kernel_size = kernel_size
        self.center_idx = kernel_size // 2

        weight_matrix = torch.ones(kernel_size, kernel_size)
        surrounding_weight_total = 1.0 - center_weight
        num_surrounding_pixels = kernel_size**2 - 1
        weight_per_surrounding_pixel = surrounding_weight_total / num_surrounding_pixels

        weight_matrix[:] = weight_per_surrounding_pixel
        weight_matrix[self.center_idx, self.center_idx] = center_weight
        self.register_buffer("weight", weight_matrix.view(1, 1, kernel_size, kernel_size))

    def forward(self, x):
        _, channels, _, _ = x.shape
        weight = self.weight.expand(1, channels, self.kernel_size, self.kernel_size)
        return (x * weight).sum(dim=(2, 3), keepdim=True)


class CPA(nn.Module):
    """Center Prompt Attention module."""
    def __init__(self, in_channels, reduction=16, img_size=13):
        super().__init__()
        self.squeeze_center = CenterBiasAvgPool(kernel_size=img_size, center_weight=0.5)
        self.excitation = nn.Sequential(
            nn.Linear(in_channels, in_channels // reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(in_channels // reduction, in_channels, bias=False),
            nn.Sigmoid(),
        )
        self.img_size = img_size
        self.register_buffer("spatial_mask", self._create_gaussian_mask())
        self.fusion_conv = nn.Conv2d(in_channels * 2, in_channels, kernel_size=1, bias=False)
        self.fusion_relu = nn.ReLU(inplace=True)

    def _create_gaussian_mask(self):
        height, width = self.img_size, self.img_size
        x = torch.arange(width, dtype=torch.float32)
        y = torch.arange(height, dtype=torch.float32)
        xx, yy = torch.meshgrid(x, y, indexing="xy")
        center_x, center_y = width // 2, height // 2
        sigma = 1
        gaussian = torch.exp(-((xx - center_x) ** 2 + (yy - center_y) ** 2) / (2 * sigma**2))
        return gaussian.unsqueeze(0).unsqueeze(0) / gaussian.max()

    def forward(self, x):
        batch_size, channels, _, _ = x.shape
        y = self.squeeze_center(x).view(batch_size, channels)
        channel_weight = self.excitation(y).view(batch_size, channels, 1, 1)
        se_feature = x * channel_weight
        mask_feature = x * self.spatial_mask
        fused_input = torch.cat([se_feature, mask_feature], dim=1)
        return self.fusion_relu(self.fusion_conv(fused_input))


class SpatialMacroMicroAttention(nn.Module):
    def __init__(self, in_channels, large_kernel_size=5, small_kernel_size=3):
        super().__init__()
        self.large_kernel_size = large_kernel_size
        self.small_kernel_size = small_kernel_size
        self.large_padding = (large_kernel_size - 1) // 2
        self.small_padding = (small_kernel_size - 1) // 2
        self.large_branch = nn.Sequential(
            nn.Conv2d(
                in_channels,
                small_kernel_size**2,
                kernel_size=large_kernel_size,
                padding=self.large_padding,
            ),
            nn.ReLU(inplace=True),
            nn.Sigmoid(),
        )

    def forward(self, x):
        batch_size, channels, height, width = x.shape
        weight_map = self.large_branch(x).unsqueeze(1)
        local_feat = F.unfold(
            x,
            kernel_size=self.small_kernel_size,
            padding=self.small_padding,
        ).view(batch_size, channels, self.small_kernel_size**2, height, width)
        modulated_feat = local_feat * weight_map
        return F.fold(
            modulated_feat.view(batch_size, channels * self.small_kernel_size**2, height * width),
            (height, width),
            self.small_kernel_size,
            1,
            self.small_padding,
        )


class SpectralMacroMicroAttention(nn.Module):
    def __init__(self, in_channels, large_kernel_size=5, small_kernel_size=5):
        super().__init__()
        self.large_kernel_size = large_kernel_size
        self.small_kernel_size = small_kernel_size
        self.channel_padding = (large_kernel_size - 1) // 2
        self.small_padding = (small_kernel_size - 1) // 2
        self.channel_weight_branch = nn.Sequential(
            nn.Conv3d(
                1,
                small_kernel_size,
                kernel_size=(large_kernel_size, 1, 1),
                padding=(self.channel_padding, 0, 0),
                bias=False,
            ),
            nn.ReLU(inplace=False),
            nn.Sigmoid(),
        )

    def forward(self, x):
        batch_size, channels, height, width = x.shape
        x_3d = x.unsqueeze(1)
        channel_weight = self.channel_weight_branch(x_3d).permute(0, 2, 1, 3, 4)
        x_permuted = x.permute(0, 2, 3, 1)
        x_padded = F.pad(
            x_permuted,
            (self.small_padding, self.small_padding, 0, 0, 0, 0),
            mode="constant",
            value=0,
        )
        x_reshaped = x_padded.reshape(batch_size * height * width, 1, x_padded.shape[-1])
        x_unfold = F.unfold(x_reshaped, kernel_size=(1, self.small_kernel_size), padding=0, stride=1)
        local_channel_feat = x_unfold.reshape(
            batch_size,
            height,
            width,
            channels,
            self.small_kernel_size,
        ).permute(0, 3, 4, 1, 2)
        return (local_channel_feat * channel_weight).sum(dim=2)


class MultiscaleSpatialMacroMicroBlock(nn.Module):
    def __init__(self, in_channels, large_kernel_sizes, small_kernel_size=3):
        super().__init__()
        self.num_branches = len(large_kernel_sizes)
        self.branches = nn.ModuleList(
            [
                SpatialMacroMicroAttention(
                    in_channels,
                    large_kernel_size=kernel_size,
                    small_kernel_size=small_kernel_size,
                )
                for kernel_size in large_kernel_sizes
            ]
        )
        self.fusion_block = nn.Sequential(
            nn.BatchNorm2d(in_channels * self.num_branches),
            nn.Conv2d(in_channels * self.num_branches, in_channels, kernel_size=3, padding=1),
            nn.ReLU(inplace=False),
        )

    def forward(self, x):
        branch_feats = [branch(x) for branch in self.branches]
        return self.fusion_block(torch.cat(branch_feats, dim=1))


class MultiscaleSpectralMacroMicroBlock(nn.Module):
    def __init__(self, in_channels, large_kernel_ratios, small_kernel_size):
        super().__init__()
        self.num_branches = len(large_kernel_ratios)
        channel_kernels = [int(in_channels * ratio) for ratio in large_kernel_ratios]
        channel_kernels = [kernel if kernel % 2 else kernel + 1 for kernel in channel_kernels]
        self.branches = nn.ModuleList(
            [
                SpectralMacroMicroAttention(
                    in_channels,
                    large_kernel_size=kernel_size,
                    small_kernel_size=small_kernel_size,
                )
                for kernel_size in channel_kernels
            ]
        )
        self.fusion_block = nn.Sequential(
            nn.BatchNorm2d(in_channels * self.num_branches),
            nn.Conv2d(in_channels * self.num_branches, in_channels, kernel_size=3, padding=1),
            nn.ReLU(inplace=False),
        )

    def forward(self, x):
        branch_feats = [branch(x) for branch in self.branches]
        return self.fusion_block(torch.cat(branch_feats, dim=1))


class Mlp(nn.Module):
    def __init__(self, in_features, hidden_features=None, out_features=None, drop=0.0):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Conv2d(in_features, hidden_features, 1)
        self.act = nn.ReLU(inplace=False)
        self.fc2 = nn.Conv2d(hidden_features, out_features, 1)
        self.drop = nn.Dropout(drop)
        self.apply(self._init_weights)

    def _init_weights(self, module):
        if isinstance(module, nn.Conv2d):
            nn.init.trunc_normal_(module.weight, std=0.02)
            if module.bias is not None:
                nn.init.constant_(module.bias, 0)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        return self.drop(x)


class SpatialMMFormerBlock(nn.Module):
    def __init__(self, in_channels, patch_size, large_kernel_sizes, small_kernel_size, dropout):
        super().__init__()
        self.norm1 = nn.BatchNorm2d(in_channels)
        self.spa_mixer = MultiscaleSpatialMacroMicroBlock(
            in_channels,
            large_kernel_sizes,
            small_kernel_size,
        )
        self.norm2 = nn.BatchNorm2d(in_channels)
        self.cpa = CPA(in_channels, img_size=patch_size, reduction=16)
        self.mlp = Mlp(in_channels, int(in_channels * 0.5), in_channels, drop=dropout)

    def forward(self, x):
        x1 = self.norm1(x)
        x1 = self.spa_mixer(x1)
        x1 = self.cpa(x1)
        x1 = x + x1
        x2 = self.norm2(x1)
        x2 = self.mlp(x2)
        return x1 + x2


class SpectralMMFormerBlock(nn.Module):
    def __init__(self, in_channels, patch_size, large_kernel_ratios, small_kernel_size, dropout):
        super().__init__()
        self.norm1 = nn.BatchNorm2d(in_channels)
        self.spe_mixer = MultiscaleSpectralMacroMicroBlock(
            in_channels,
            large_kernel_ratios,
            small_kernel_size,
        )
        self.norm2 = nn.BatchNorm2d(in_channels)
        self.cpa = CPA(in_channels, img_size=patch_size, reduction=16)
        self.mlp = Mlp(in_channels, int(in_channels * 0.5), in_channels, drop=dropout)

    def forward(self, x):
        x1 = self.norm1(x)
        x1 = self.spe_mixer(x1)
        x1 = self.cpa(x1)
        x1 = x + x1
        x2 = self.norm2(x1)
        x2 = self.mlp(x2)
        return x1 + x2


class PatchEmbedding(nn.Module):
    def __init__(self, in_channels, embedding_dim, dropout):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, embedding_dim, kernel_size=1, padding=0)
        self.bn1 = nn.BatchNorm2d(embedding_dim)
        self.activation = nn.LeakyReLU(negative_slope=0.01, inplace=True)
        self.conv2 = nn.Conv2d(embedding_dim, embedding_dim, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(embedding_dim)
        self.dropout = nn.Dropout2d(p=dropout)

    def forward(self, x):
        x = self.activation(self.bn1(self.conv1(x)))
        return self.dropout(self.bn2(self.conv2(x)))


class CWF(nn.Module):
    """Cross Weighted Fusion module."""
    def __init__(self, embedding_dim):
        super().__init__()
        self.pre_fuse_conv = nn.Sequential(
            nn.BatchNorm2d(embedding_dim * 2),
            nn.Conv2d(embedding_dim * 2, embedding_dim, kernel_size=3, padding=1),
            nn.ReLU(inplace=False),
        )
        self.spa_weight_gen = nn.Sequential(
            nn.Conv2d(embedding_dim, embedding_dim, kernel_size=3, padding=1, bias=False),
            nn.ReLU(inplace=False),
            nn.Conv2d(embedding_dim, embedding_dim, kernel_size=1, padding=0, bias=False),
            nn.Sigmoid(),
        )
        self.spe_weight_gen = nn.Sequential(
            nn.Conv2d(embedding_dim, embedding_dim // 4, kernel_size=1, padding=0, bias=False),
            nn.ReLU(inplace=False),
            nn.Conv2d(embedding_dim // 4, embedding_dim, kernel_size=1, padding=0, bias=False),
            nn.Sigmoid(),
        )
        self.final_fusion = nn.Sequential(
            nn.BatchNorm2d(embedding_dim * 2),
            nn.Conv2d(embedding_dim * 2, embedding_dim, kernel_size=3, padding=1),
            nn.ReLU(inplace=False),
        )

    def forward(self, spa_feat, spe_feat):
        fused_context = self.pre_fuse_conv(torch.cat([spa_feat, spe_feat], dim=1))
        spa_weight = self.spa_weight_gen(fused_context)
        spe_weight = self.spe_weight_gen(fused_context)
        spa_weighted = spa_feat * spe_weight + spa_feat
        spe_weighted = spe_feat * spa_weight + spe_feat
        return self.final_fusion(torch.cat([spa_weighted, spe_weighted], dim=1))


class MMFormer(nn.Module):
    """
    MMFormer: Macro-Micro Transformer for Hyperspectral Image Classification.
    
    Dual-branch architecture with spatial and spectral Macro-Micro Transformer
    branches, Center Prompt Attention (CPA), and Cross Weighted Fusion (CWF).
    
    Takes 4D input [B, C, H, W] (expects_4d=True, wrapped by InputShapeWrapper).
    """
    def __init__(
        self,
        num_classes=16,
        bands=30,
        patch_size=13,
        embedding_dim=128,
        spatial_macro_kernel_sizes=None,
        spectral_macro_kernel_ratios=None,
        spatial_micro_kernel_size=3,
        spectral_micro_kernel_size=9,
        dropout=0.1,
    ):
        super().__init__()
        
        if spatial_macro_kernel_sizes is None:
            spatial_macro_kernel_sizes = [5, 7, 9]
        if spectral_macro_kernel_ratios is None:
            spectral_macro_kernel_ratios = [0.3, 0.5, 0.7]
        
        self.shallow_extractor = PatchEmbedding(bands, embedding_dim, dropout)

        self.spa_block1 = SpatialMMFormerBlock(
            embedding_dim, patch_size, spatial_macro_kernel_sizes,
            spatial_micro_kernel_size, dropout,
        )
        self.spa_block2 = SpatialMMFormerBlock(
            embedding_dim, patch_size, spatial_macro_kernel_sizes,
            spatial_micro_kernel_size, dropout,
        )
        self.spa_block3 = SpatialMMFormerBlock(
            embedding_dim, patch_size, spatial_macro_kernel_sizes,
            spatial_micro_kernel_size, dropout,
        )

        self.spe_block1 = SpectralMMFormerBlock(
            embedding_dim, patch_size, spectral_macro_kernel_ratios,
            spectral_micro_kernel_size, dropout,
        )
        self.spe_block2 = SpectralMMFormerBlock(
            embedding_dim, patch_size, spectral_macro_kernel_ratios,
            spectral_micro_kernel_size, dropout,
        )
        self.spe_block3 = SpectralMMFormerBlock(
            embedding_dim, patch_size, spectral_macro_kernel_ratios,
            spectral_micro_kernel_size, dropout,
        )

        self.cross_fusion = CWF(embedding_dim)
        self.avg_pool = nn.AdaptiveAvgPool2d((1, 1))
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(embedding_dim, embedding_dim // 2),
            nn.ReLU(inplace=True),
            nn.Dropout(p=dropout),
            nn.Linear(embedding_dim // 2, num_classes),
        )

    def forward(self, x):
        """
        Args:
            x: [B, C, H, W] input (4D, after InputShapeWrapper squeezes 5D)
        Returns:
            logits: [B, num_classes]
        """
        x_embed = self.shallow_extractor(x)
        spa_x1 = self.spa_block1(x_embed)
        spe_x1 = self.spe_block1(x_embed)
        spa_x2 = self.spa_block2(spa_x1)
        spe_x2 = self.spe_block2(spe_x1)
        spa_x3 = self.spa_block3(spa_x2)
        spe_x3 = self.spe_block3(spe_x2)
        fusion_x = self.cross_fusion(spa_x3, spe_x3)
        return self.classifier(self.avg_pool(fusion_x))


@register_model('MMFormer', expects_4d=True, embedding_dim=128, dropout=0.1)
def mmformer_model(pretrained: bool = False, **kwargs) -> MMFormer:
    """Constructs an MMFormer model.
    
    Takes 4D input [B, C, H, W] via InputShapeWrapper (from 5D [B, 1, C, H, W]).
    """
    # Remove registry metadata keys
    for key in ['expects_4d']:
        kwargs.pop(key, None)
    
    return MMFormer(**kwargs)
