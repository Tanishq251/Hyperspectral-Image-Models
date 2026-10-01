from models.registry import register_model
"""
MambaHSI+: Enhanced Spatial-Spectral Mamba for Hyperspectral Image Classification

Paper: Pending publication
GitHub: https://github.com/RockAilab/MambaHSI_Plus
Venue: IEEE Transactions on Geoscience and Remote Sensing (TGRS)
Year: 2025
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from mamba_ssm import Mamba
from models.helpers.mambaHSI_plus_help import ECALayer, SpeMambaProcessor, SpaMambaProcessor
import os

class SpeMamba(nn.Module):
    def __init__(self, channels, token_num=8, use_residual=True, group_num=4, use_proj=True, use_att=True):
        super(SpeMamba, self).__init__()
        self.token_num = token_num
        self.use_residual = use_residual
        self.use_proj = use_proj
        self.use_att = use_att
        if self.use_att:
            self.weights = nn.Parameter(torch.ones(2) / 2)
            self.softmax = nn.Softmax(dim=0)
        self.group_channel_num = math.ceil(channels / token_num)
        self.channel_num = self.token_num * self.group_channel_num

        # Mamba 分别建模行优先和列优先
        self.mamba_col = SpeMambaProcessor(group_channel_num=self.group_channel_num)  # 列优先
        self.mamba_row = SpeMambaProcessor(group_channel_num=self.group_channel_num)
        # 加入 ECA 注意力
        self.eca = ECALayer(channel=self.channel_num)
        if self.use_proj:
            self.proj = nn.Sequential(
                nn.GroupNorm(group_num, self.channel_num),
                nn.SiLU(),
            )

    def padding_feature(self, x):
        B, C, H, W = x.shape
        if C < self.channel_num:
            pad_c = self.channel_num - C
            pad_features = torch.zeros((B, pad_c, H, W), device=x.device)
            return torch.cat([x, pad_features], dim=1)
        return x

    def forward(self, x):
        # Padding logic
        x_pad = self.padding_feature(x)
        # B, C, H, W -> B, H, W, C (row-major)

        # **行优先建模**
        x_row = x_pad.permute(0, 2, 3, 1).contiguous()  # [B, H, W, C]
        x_row = self.mamba_row(x_row.contiguous()).permute(0, 3, 1, 2) # [B, C, H, W]

        # **列优先建模**
        x_col = x_pad.permute(0, 3, 2, 1).contiguous()  # [B, W, H, C]
        x_col = self.mamba_col(x_col.contiguous()).permute(0, 3, 2, 1) # [B, C, H, W]

        # **融合行/列信息**
        if self.use_att:
            weights = self.softmax(self.weights)
            x_recon = x_row * weights[0] + x_col * weights[1]
        else:
            # 不开注意力就简单相加
            x_recon = x_row + x_col

        if self.use_proj:
            x_recon = self.proj(x_recon)
        # Apply ECA attention
        x_recon = self.eca(x_recon)
        if self.use_residual:
            return x_recon + x_pad
        else:
            return x_recon



class SpaMamba(nn.Module):
    def __init__(self, channels, use_residual=True, group_num=4, use_proj=True, use_att=True):
        super(SpaMamba, self).__init__()
        self.use_residual = use_residual
        self.use_proj = use_proj
        self.use_att = use_att
        if self.use_att:
            self.weights = nn.Parameter(torch.ones(2) / 2)
            self.softmax = nn.Softmax(dim=0)
        # Mamba 分别建模行优先和列优先
        self.mamba_col = SpaMambaProcessor(channels=channels)  # 列优先
        self.mamba_row = SpaMambaProcessor(channels=channels)
        # 同样加 ECA 注意力
        self.eca = ECALayer(channel=channels)

        if self.use_proj:
            self.proj = nn.Sequential(
                nn.GroupNorm(group_num, channels),
                nn.SiLU(),
            )

    def forward(self, x):

        # **行优先建模**
        x_row = x.permute(0, 2, 3, 1).contiguous()  # [B, H, W, C]
        x_row = self.mamba_row(x_row.contiguous()).permute(0, 3, 1, 2) # [B, H, W, C]

        # **列优先建模**
        x_col = x.permute(0, 3, 2, 1).contiguous()  # [B, W, H, C]
        x_col = self.mamba_col(x_col.contiguous()).permute(0, 3, 2, 1) # [B, H, W, C]

        # **融合行/列信息**
        if self.use_att:
            weights = self.softmax(self.weights)
            x_recon = x_row * weights[0] + x_col * weights[1]
            # print(weights[0], weights[1])
        else:
            # 不开注意力就简单相加
            x_recon = x_row + x_col
        
        if self.use_proj:
            x_recon = self.proj(x_recon)
        # ECA
        x_recon = self.eca(x_recon)
        if self.use_residual:
            return x_recon + x
        else:
            return x_recon


class BothMamba(nn.Module):
    def __init__(self, channels, token_num, use_residual=True, group_num=4, use_att=False):
        super(BothMamba, self).__init__()
        self.use_att = use_att
        self.use_residual = use_residual

        if self.use_att:
            self.weights = nn.Parameter(torch.ones(2) / 2)
            self.softmax = nn.Softmax(dim=0)

        self.spa_mamba = SpaMamba(channels, use_residual=use_residual, group_num=group_num)
        self.spe_mamba = SpeMamba(channels, token_num=token_num, use_residual=use_residual, group_num=group_num)

    def forward(self, x):
        spa_x = self.spa_mamba(x)
        spe_x = self.spe_mamba(x)

        fusion_x = spa_x + spe_x

        if self.use_residual:
            return fusion_x + x
        else:
            return fusion_x


class MambaHSI_Plus(nn.Module):
    """
    """
    def __init__(
        self,
        in_channels=128,
        hidden_dim=128,
        num_classes=10,
        use_residual=True,
        mamba_type='both',
        token_num=4,
        group_num=2,
        use_att=True
    ):
        super(MambaHSI_Plus, self).__init__()
        self.mamba_type = mamba_type

        # 1. 光谱压缩/增强
        self.patch_embedding = nn.Sequential(nn.Conv2d(in_channels=in_channels,out_channels=hidden_dim,kernel_size=1,stride=1,padding=0),
                                             nn.GroupNorm(group_num,hidden_dim),
                                             nn.SiLU())


        # 3. Mamba 堆叠
        self.mamba = nn.Sequential(
            BothMamba(channels=hidden_dim, token_num=token_num, use_residual=use_residual, group_num=group_num, use_att=use_att),
            nn.AvgPool2d(kernel_size=2, stride=2),
            BothMamba(channels=hidden_dim, token_num=token_num, use_residual=use_residual, group_num=group_num, use_att=use_att),
            nn.AvgPool2d(kernel_size=2, stride=2),
            BothMamba(channels=hidden_dim, token_num=token_num, use_residual=use_residual, group_num=group_num, use_att=use_att),
        )

        # 4. 分类头（保持不变）
        self.cls_head = nn.Sequential(
            nn.Conv2d(in_channels=hidden_dim, out_channels=hidden_dim, kernel_size=3, stride=1, padding=1),
            nn.GroupNorm(group_num, hidden_dim),
            nn.SiLU(),
            nn.Conv2d(in_channels=hidden_dim, out_channels=num_classes, kernel_size=3, stride=1, padding=1),
        )

    def forward(self, x):
        # step1: 压缩光谱
        x = self.patch_embedding(x)
        # step2: 多尺度卷积
        x = self.mamba(x)
        # step4: 分类头
        logits = self.cls_head(x)
        # Pool spatial dimensions to get [B, num_classes]
        logits = logits.mean(dim=[-2, -1])  # Global average pooling over H, W
        return logits


@register_model('MambaHSI_Plus', expects_4d=True, hidden_dim=64, use_residual=True, mamba_type='both', token_num=4, group_num=4, use_att=True)
def mambahsi_plus(pretrained: bool = False, **kwargs) -> MambaHSI_Plus:
    """Constructs a MambaHSI_Plus model."""
    # Map standardized names to model-specific names
    if 'bands' in kwargs:
        kwargs['in_channels'] = kwargs.pop('bands')
    kwargs.pop('patch_size', None)  # MambaHSI_Plus doesn't use patch_size
    return MambaHSI_Plus(**kwargs)
