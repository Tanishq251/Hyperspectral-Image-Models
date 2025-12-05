"""
FAHM: Frequency-Aware Hierarchical Mamba for Hyperspectral Image Classification

Paper: https://ieeexplore.ieee.org/abstract/document/10877784
GitHub: https://github.com/zhangxc0105/FAHM
Venue: IEEE Transactions on Geoscience and Remote Sensing (TGRS)
Year: 2025
"""

import math
import torch
import torch.nn as nn
from einops.layers.torch import Rearrange
from .helpers.rope_spectral import RotaryEmbedding
from torch.nn import functional as F
from mamba_ssm import Mamba

class PA(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.pa_conv = nn.Conv2d(dim, dim, kernel_size=3, padding=1, groups=dim)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        return x * self.sigmoid(self.pa_conv(x))

class FGEM(nn.Module):
    def __init__(self, in_feature_map_size=7, in_chans=3, embed_dim=128, n_groups=2, sparsity_threshold=0.01, hard_thresholding_fraction=1):
        super().__init__()
        assert in_chans % n_groups == 0, f"hidden_size {in_chans} should be divisible by num_blocks {n_groups}"
        self.n_groups = n_groups

        self.hidden_size = in_chans
        self.sparsity_threshold = sparsity_threshold
        self.block_size = self.hidden_size // self.n_groups
        self.hard_thresholding_fraction = hard_thresholding_fraction
        self.scale = 0.02

        self.w1 = nn.Parameter(
            self.scale * torch.randn(2, self.n_groups, self.block_size, self.block_size))
        self.b1 = nn.Parameter(self.scale * torch.randn(2, self.n_groups, self.block_size))

        self.act = nn.GELU()
        self.proj = nn.Conv2d(in_chans, embed_dim, kernel_size=1)
        self.batch_norm = nn.BatchNorm2d(embed_dim)
        self.ifm_size = in_feature_map_size

    def forward(self, x):
        dtype = x.dtype
        x = x.float()
        B, C, H, W = x.shape
        x = torch.fft.rfft2(x, dim=(2, 3), norm="ortho")
        origin_ffted = x
        x = x.reshape(B, self.n_groups, self.block_size, x.shape[2], x.shape[3])

        o1_real_part1 = torch.einsum('bkihw,kio->bkohw', x.real, self.w1[0])
        o1_real_part2 = torch.einsum('bkihw,kio->bkohw', x.imag, self.w1[1])
        o1_real = self.act(o1_real_part1 - o1_real_part2 + self.b1[0, :, :, None, None])+x.real

        o1_imag_part1 = torch.einsum('bkihw,kio->bkohw', x.imag, self.w1[0])
        o1_imag_part2 = torch.einsum('bkihw,kio->bkohw', x.real, self.w1[1])
        o1_imag = self.act(o1_imag_part1 + o1_imag_part2 + self.b1[1, :, :, None, None])+x.imag

        x = torch.stack([o1_real, o1_imag], dim=-1)
        x = F.softshrink(x, lambd=self.sparsity_threshold)
        x = torch.view_as_complex(x)
        x = x.reshape(B, C, x.shape[3], x.shape[4])

        x = x + origin_ffted
        x = torch.fft.irfft2(x, s=(H, W), dim=(2, 3), norm="ortho")

        x = x.type(dtype)
        x = self.act(self.batch_norm(self.proj(x)))
        x = x.flatten(2).transpose(1, 2)

        after_feature_map_size = self.ifm_size
        return x, after_feature_map_size

class Channelaffinity(nn.Module):
    def __init__(self, dim, reduction=16):
        super(Channelaffinity, self).__init__()
        num_channels_reduced = dim // reduction

        self.fc1 = nn.Linear(dim, num_channels_reduced, bias=True)
        self.fc2 = nn.Linear(num_channels_reduced, dim, bias=True)
        self.relu = nn.ReLU()
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        z = x.permute(0, 2, 1).mean(dim=2)
        fc_out_1 = self.relu(self.fc1(z))
        fc_out_2 = self.sigmoid(self.fc2(fc_out_1))
        fc_out_2 = fc_out_2.unsqueeze(1)
        fc_out_2 = fc_out_2 * x
        return fc_out_2

class pre_conv(nn.Module):
    def __init__(self, in_chans):
        super().__init__()

        self.act = nn.GELU()
        self.conv1 = nn.Conv2d(in_chans, in_chans, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(in_chans)
        self.conv2 = nn.Conv2d(in_chans, in_chans, kernel_size=3, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(in_chans)

        self.proj = nn.Conv3d(1, 1, kernel_size=1, bias=False)
        self.bn4 = nn.BatchNorm3d(1)

    def forward(self, x):
        # x.shape = B, C, s, s
        x_1 = x

        x0 = self.conv1(x)
        x1 = self.conv2(self.act(self.bn1(x0)))
        x = self.act(self.bn2(x1))

        x_1 = x_1.unsqueeze(dim = 1)
        x_1 = self.act(self.proj(x_1))
        x_1 = x_1.squeeze(dim = 1)

        x = x + x_1
        return x

class Channel_Merge(nn.Module):
    def __init__(self, patchsize, in_chans, out_chans):
        super().__init__()
        self.patchsize = patchsize
        self.channel_conv = nn.Conv1d(in_chans, out_chans, kernel_size=1)

        self.batch_norm = nn.BatchNorm2d(out_chans)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        # B, N, C = x.shape
        x = x.transpose(1, 2)
        x = self.channel_conv(x)
        x = x.transpose(1, 2)

        return x

class FAHM(nn.Module):
    def __init__(self, img_size=224, in_chans=3, num_classes=1000, num_stages=3,
                 n_groups=[32, 32, 32], embed_dims=[256, 128, 64], depths=[1, 1, 1], **kwargs):
        super().__init__()

        self.num_stages = num_stages
        self.patchsize = img_size
        new_bands = math.ceil(in_chans / n_groups[0]) * n_groups[0]
        self.pad = nn.ReplicationPad3d((0, 0, 0, 0, 0, new_bands - in_chans))
        self.merge = nn.ModuleList()
        self.channel_fusion = Channelaffinity(embed_dims[-1])
        self.depths = depths
        self.patchmerge = nn.ModuleList()
        self.postion = PA(in_chans)
        self.w = nn.Parameter(torch.ones(3))

        self.act = nn.GELU()

        for i in range(num_stages):
            patch_embed = FGEM(
                in_feature_map_size=img_size,
                in_chans=new_bands if i == 0 else embed_dims[i - 1],
                embed_dim=embed_dims[i],
                n_groups=n_groups[i]
            )

            norm = nn.LayerNorm(embed_dims[i])
            rope = RotaryEmbedding(dim=embed_dims[i] // 2)

            self.merge.append(Channel_Merge(img_size, embed_dims[i], embed_dims[-1]))

            setattr(self, f"patch_embed{i + 1}", patch_embed)
            setattr(self, f"norm{i + 1}", norm)
            setattr(self, f"rope{i + 1}", rope)

        self.rope = RotaryEmbedding(dim=in_chans // 2)
        self.mamba = nn.ModuleList()
        self.mamba_back = nn.ModuleList()

        for stage_idx in range(num_stages):
            for _ in range(depths[stage_idx]):
                self.mamba.append(Mamba(embed_dims[stage_idx]))

        self.head = nn.Linear(embed_dims[-1], num_classes)
        self.patchify = Rearrange("b d c  -> b c d")

        self.pre_conv = pre_conv(in_chans)

    def forward_features(self, x):
        B, _, C, H, W = x.shape  # Adjusted to handle the input shape (B, 1, C, H, W)
        x = x.squeeze(1)  # Remove the channel dimension
        x = x.reshape(B, C, self.patchsize, self.patchsize)

        x = self.pre_conv(x)
        x = self.postion(x)

        x = self.pad(x).squeeze(dim=1)
        B = x.shape[0]

        z = []
        layer_idx = 0

        for i in range(self.num_stages):
            patch_embed = getattr(self, f"patch_embed{i + 1}")
            norm = getattr(self, f"norm{i + 1}")
            rope = getattr(self, f"rope{i + 1}")

            x, s = patch_embed(x)

            x = rope.rotate_queries_or_keys(x)
            # Mamba layer processing
            for _ in range(self.depths[i]):
                x = self.mamba[layer_idx](x)
                layer_idx += 1

            x = norm(x)

            y = self.channel_fusion(self.merge[i](x))
            z.append(y)
            if i == self.num_stages - 1:
                x = z[0]*self.w[0] + z[1]*self.w[1]+z[2]*self.w[2]

            if i != self.num_stages - 1:
                x = x.reshape(B, s, s, -1).permute(0, 3, 1, 2).contiguous()

        return x

    def forward(self, x):
        x = self.forward_features(x)
        x = x.mean(dim=1)
        x = self.head(x)
        return x


# Register model
from .registry import register_model

@register_model('FAHM', num_stages=3, n_groups=[32, 32, 32], embed_dims=[256, 128, 64], depths=[1, 1, 1])
def fahm(pretrained: bool = False, **kwargs) -> FAHM:
    """Constructs a FAHM model."""
    # Map standardized names to model-specific names
    if 'bands' in kwargs:
        kwargs['in_chans'] = kwargs.pop('bands')
    if 'patch_size' in kwargs:
        kwargs['img_size'] = kwargs.pop('patch_size')
    return FAHM(**kwargs)
