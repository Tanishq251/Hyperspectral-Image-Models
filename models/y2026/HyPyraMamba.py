"""
HyPyraMamba: Spatial-Spectral Mamba with Pyramid Attention for Hyperspectral Image Classification

Paper/Source: https://github.com/dekai-li/HyPyraMamba
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import typing as t
from einops import rearrange
from mamba_ssm import Mamba

from models.registry import register_model


class SCSA(nn.Module):
    def __init__(self,
                 dim: int,
                 head_num: int,
                 window_size: int = 7,
                 group_kernel_sizes: t.List[int] = [3, 5, 7, 9],
                 qkv_bias: bool = False,
                 down_sample_mode: str = 'avg_pool',
                 attn_drop_ratio: float = 0.,
                 gate_layer: str = 'sigmoid',
                 ):
        super(SCSA, self).__init__()

        self.dim = dim
        self.head_num = head_num
        self.head_dim = dim // head_num
        self.scaler = self.head_dim ** -0.5
        self.group_kernel_sizes = group_kernel_sizes
        self.window_size = window_size
        self.qkv_bias = qkv_bias
        self.down_sample_mode = down_sample_mode

        assert self.dim % 4 == 0, 'The dimension of input feature should be divisible by 4.'
        self.group_chans = self.dim // 4

        self.local_dwc = nn.Conv1d(self.group_chans, self.group_chans, kernel_size=self.group_kernel_sizes[0],
                                   padding=self.group_kernel_sizes[0] // 2, groups=self.group_chans)
        self.global_dwcs = nn.ModuleList([
            nn.Conv1d(self.group_chans, self.group_chans, kernel_size=size, padding=size // 2, groups=self.group_chans)
            for size in self.group_kernel_sizes[1:]
        ])

        self.sa_gate = nn.Softmax(dim=2) if gate_layer == 'softmax' else nn.Sigmoid()
        self.norm_h = nn.GroupNorm(4, dim)
        self.norm_w = nn.GroupNorm(4, dim)
        self.conv_d = nn.Identity()
        self.norm = nn.GroupNorm(1, dim)

        self.q = nn.Conv2d(in_channels=dim, out_channels=dim, kernel_size=1, bias=qkv_bias, groups=dim)
        self.k = nn.Conv2d(in_channels=dim, out_channels=dim, kernel_size=1, bias=qkv_bias, groups=dim)
        self.v = nn.Conv2d(in_channels=dim, out_channels=dim, kernel_size=1, bias=qkv_bias, groups=dim)
        self.attn_drop = nn.Dropout(attn_drop_ratio)
        self.ca_gate = nn.Softmax(dim=1) if gate_layer == 'softmax' else nn.Sigmoid()

        if window_size == -1:
            self.down_func = nn.AdaptiveAvgPool2d((1, 1))
        else:
            if down_sample_mode == 'avg_pool':
                self.down_func = nn.AvgPool2d(kernel_size=(window_size, window_size), stride=window_size)
            elif down_sample_mode == 'max_pool':
                self.down_func = nn.MaxPool2d(kernel_size=(window_size, window_size), stride=window_size)
            else:
                self.down_func = nn.AvgPool2d(kernel_size=(window_size, window_size), stride=window_size)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b, c, h_, w_ = x.size()

        x_h = x.mean(dim=3)
        l_x_h, g_x_h_s, g_x_h_m, g_x_h_l = torch.split(x_h, self.group_chans, dim=1)
        x_w = x.mean(dim=2)
        l_x_w, g_x_w_s, g_x_w_m, g_x_w_l = torch.split(x_w, self.group_chans, dim=1)

        x_h_attn = self.sa_gate(self.norm_h(torch.cat((
            self.local_dwc(l_x_h),
            *[gw(l_x_h) for gw in self.global_dwcs]
        ), dim=1)))
        x_h_attn = x_h_attn.view(b, c, h_, 1)

        x_w_attn = self.sa_gate(self.norm_w(torch.cat((
            self.local_dwc(l_x_w),
            *[gw(l_x_w) for gw in self.global_dwcs]
        ), dim=1)))
        x_w_attn = x_w_attn.view(b, c, 1, w_)

        x = x * x_h_attn * x_w_attn

        y = self.down_func(x)
        y = self.conv_d(y)
        _, _, h_, w_ = y.size()

        y = self.norm(y)
        q = self.q(y)
        k = self.k(y)
        v = self.v(y)

        q = rearrange(q, 'b (head_num head_dim) h w -> b head_num head_dim (h w)', head_num=self.head_num,
                      head_dim=self.head_dim)
        k = rearrange(k, 'b (head_num head_dim) h w -> b head_num head_dim (h w)', head_num=self.head_num,
                      head_dim=self.head_dim)
        v = rearrange(v, 'b (head_num head_dim) h w -> b head_num head_dim (h w)', head_num=self.head_num,
                      head_dim=self.head_dim)

        attn = q @ k.transpose(-2, -1) * self.scaler
        attn = self.attn_drop(attn.softmax(dim=-1))

        attn = attn @ v
        attn = rearrange(attn, 'b head_num head_dim (h w) -> b (head_num head_dim) h w', h=h_, w=w_)

        attn = attn.mean((2, 3), keepdim=True)
        attn = self.ca_gate(attn)

        return attn * x


class PyramidAttention(nn.Module):
    def __init__(self, dim, num_heads, bias):
        super(PyramidAttention, self).__init__()
        self.num_heads = num_heads
        self.temperature = nn.Parameter(torch.ones(num_heads, 1, 1))

        self.qkv = nn.Conv2d(dim, dim * 3, kernel_size=1, bias=bias)
        self.qkv_dwconv = nn.Conv2d(dim * 3, dim * 3, kernel_size=3, stride=1, dilation=2, padding=2, groups=dim * 3,
                                    bias=bias)
        self.project_out = nn.Conv2d(dim, dim, kernel_size=1, bias=bias)

    def forward(self, x):
        b, c, h, w = x.shape

        qkv = self.qkv_dwconv(self.qkv(x))
        q, k, v = qkv.chunk(3, dim=1)

        q = rearrange(q, 'b (head c) h w -> b head c (h w)', head=self.num_heads)
        k = rearrange(k, 'b (head c) h w -> b head c (h w)', head=self.num_heads)
        v = rearrange(v, 'b (head c) h w -> b head c (h w)', head=self.num_heads)

        q = torch.nn.functional.normalize(q, dim=-1)
        k = torch.nn.functional.normalize(k, dim=-1)

        attn = (q @ k.transpose(-2, -1)) * self.temperature
        attn = attn.softmax(dim=-1)

        out = (attn @ v)
        out = rearrange(out, 'b head c (h w) -> b (head c) h w', head=self.num_heads, h=h, w=w)
        out = self.project_out(out)

        return out


class PyramidRefinedChannelAttention(nn.Module):
    def __init__(self, dim, num_heads, bias, num_scales=3, num_layers=2):
        super(PyramidRefinedChannelAttention, self).__init__()

        self.num_heads = num_heads
        self.temperature = nn.Parameter(torch.ones(num_heads, 1, 1))

        self.attention_modules = nn.ModuleList([
            PyramidAttention(dim, num_heads, bias) for _ in range(num_scales)
        ])

        self.attention_layers = nn.ModuleList([
            nn.ModuleList([PyramidAttention(dim, num_heads, bias) for _ in range(num_layers)])
            for _ in range(num_scales)
        ])

        self.project_out = nn.Conv2d(dim * num_scales, dim, kernel_size=1, bias=bias)

    def forward(self, x):
        b, c, h, w = x.shape
        outputs = []

        for i, attention_module in enumerate(self.attention_modules):
            if i == 0:
                scaled_input = x
            else:
                scaled_input = nn.functional.avg_pool2d(x, kernel_size=2 ** i, stride=2 ** i)

            output = attention_module(scaled_input)

            for layer in self.attention_layers[i]:
                output = layer(output)

            if i > 0:
                output = torch.nn.functional.interpolate(output, size=(h, w), mode='bilinear', align_corners=False)

            outputs.append(output)

        out = torch.cat(outputs, dim=1)
        out = self.project_out(out)

        return out


class MultiScaleConv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(MultiScaleConv, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(in_channels, out_channels, kernel_size=5, padding=2)
        self.conv3 = nn.Conv2d(in_channels, out_channels, kernel_size=7, padding=3)
        self.conv4 = nn.Conv2d(in_channels, out_channels, kernel_size=1)
        self.final_conv = nn.Conv2d(out_channels * 4, out_channels, kernel_size=1)
        self.norm = nn.GroupNorm(1, out_channels)
        self.activation = nn.ReLU()
        self.attention = ChannelAttention(out_channels)

    def forward(self, x):
        out1 = self.conv1(x)
        out2 = self.conv2(x)
        out3 = self.conv3(x)
        out4 = self.conv4(x)
        out = torch.cat((out1, out2, out3, out4), dim=1)
        out = self.final_conv(out)
        out = self.norm(out)
        out = self.activation(out)
        return self.attention(out)


class DynamicConvBlock(nn.Module):
    def __init__(self, channels, kernel_size=3, num_experts=4, reduction=4, dropout=0.1):
        super(DynamicConvBlock, self).__init__()
        self.channels = channels
        self.num_experts = num_experts
        self.kernel_size = kernel_size

        self.attention = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(channels, channels // reduction, kernel_size=1),
            nn.ReLU(),
            nn.Conv2d(channels // reduction, num_experts, kernel_size=1),
            nn.Softmax(dim=1)
        )

        self.convs = nn.ModuleList([
            nn.Conv2d(channels, channels, kernel_size=kernel_size, padding=kernel_size // 2, groups=channels)
            for _ in range(num_experts)
        ])

        self.norm = nn.GroupNorm(1, channels)
        self.dropout = nn.Dropout(dropout)
        self.activation = nn.SiLU()

    def forward(self, x):
        B, C, H, W = x.shape

        attention_weights = self.attention(x).view(B, self.num_experts, 1, 1, 1)

        outputs = [conv(x).unsqueeze(1) for conv in self.convs]
        outputs = torch.cat(outputs, dim=1)

        x = (outputs * attention_weights).sum(dim=1)
        x = self.norm(x)
        x = self.activation(x)
        x = self.dropout(x)
        return x


class ChannelAttention(nn.Module):
    def __init__(self, in_channels, reduction=16):
        super(ChannelAttention, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc1 = nn.Conv2d(in_channels, in_channels // reduction, 1, bias=False)
        self.relu = nn.ReLU()
        self.fc2 = nn.Conv2d(in_channels // reduction, in_channels, 1, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        y = self.avg_pool(x)
        y = self.fc1(y)
        y = self.relu(y)
        y = self.fc2(y)
        return x * self.sigmoid(y)


class ImprovedSpeMamba(nn.Module):
    def __init__(self, channels, token_num=4, use_residual=True, group_num=4, num_scales=3, num_layers=2):
        super(ImprovedSpeMamba, self).__init__()
        self.token_num = token_num
        self.use_residual = use_residual
        self.group_channel_num = math.ceil(channels / token_num)
        self.channel_num = self.token_num * self.group_channel_num
        self.pyramid_refined_attention = PyramidRefinedChannelAttention(
            dim=self.channel_num,
            num_heads=4,
            bias=True,
            num_scales=num_scales,
            num_layers=num_layers
        )
        self.mamba = Mamba(
            d_model=self.group_channel_num,
            d_state=16,
            d_conv=4,
            expand=2,
        )
        self.proj = nn.Sequential(
            nn.GroupNorm(group_num, self.channel_num),
            nn.SiLU()
        )

    def padding_feature(self, x):
        B, C, H, W = x.shape
        if C < self.channel_num:
            pad_c = self.channel_num - C
            pad_features = torch.zeros((B, pad_c, H, W)).to(x.device)
            cat_features = torch.cat([x, pad_features], dim=1)
            return cat_features
        else:
            return x

    def forward(self, x):
        x_pad = self.padding_feature(x)
        x_re = self.pyramid_refined_attention(x_pad)

        B, C, H, W = x_re.shape
        x_re_flat = x_re.view(B * H * W, self.token_num, self.group_channel_num)
        x_recon = self.mamba(x_re_flat)

        x_recon = x_recon.view(B, C, H, W)
        x_recon = self.proj(x_recon)
        return x_recon + x if self.use_residual else x_recon


class ImprovedSpaMamba(nn.Module):
    def __init__(self, channels, use_residual=True, group_num=4, token_num=4, num_scales=3, num_layers=2):
        super(ImprovedSpaMamba, self).__init__()
        self.use_residual = use_residual
        self.token_num = token_num
        self.group_channel_num = math.ceil(channels / token_num)
        self.channel_num = self.token_num * self.group_channel_num
        self.pyramid_refined_attention = PyramidRefinedChannelAttention(
            dim=self.channel_num,
            num_heads=4,
            bias=True,
            num_scales=num_scales,
            num_layers=num_layers
        )

        self.mamba = Mamba(
            d_model=channels,
            d_state=16,
            d_conv=4,
            expand=2,
        )

        self.multi_scale_conv = MultiScaleConv(channels, channels)
        self.scsa = SCSA(dim=channels, head_num=4, window_size=7)

        self.proj = nn.Sequential(
            nn.GroupNorm(group_num, channels),
            nn.SiLU()
        )

    def forward(self, x):
        x_re = self.pyramid_refined_attention(x)

        B, C, H, W = x_re.shape
        x_flat = x_re.view(B * H * W, 1, C)
        x_flat = self.mamba(x_flat)

        x_recon = x_flat.view(B, C, H, W)
        x_recon = self.proj(x_recon)

        return x_recon + x if self.use_residual else x_recon


class ImprovedBothMamba(nn.Module):
    def __init__(self, channels, token_num, use_residual, group_num=4):
        super(ImprovedBothMamba, self).__init__()
        self.use_residual = use_residual

        self.spa_mamba = ImprovedSpaMamba(channels, use_residual=use_residual, group_num=group_num)
        self.spe_mamba = ImprovedSpeMamba(channels, token_num=token_num, use_residual=use_residual, group_num=group_num)

        self.attention = nn.Sequential(
            nn.Conv2d(2 * channels, channels, kernel_size=1),
            nn.SiLU(),
            nn.Conv2d(channels, 1, kernel_size=1),
            nn.Sigmoid()
        )

    def forward(self, x):
        spa_x = self.spa_mamba(x)
        spe_x = self.spe_mamba(x)

        fused_input = torch.cat([spa_x, spe_x], dim=1)
        attention_map = self.attention(fused_input)

        spa_x_attended = spa_x * attention_map
        spe_x_attended = spe_x * (1 - attention_map)

        fusion_x = spa_x_attended + spe_x_attended

        return fusion_x + x if self.use_residual else fusion_x


class HyPyraMamba(nn.Module):
    def __init__(self, in_channels=128, hidden_dim=64, num_classes=10, use_residual=True, mamba_type='both',
                 token_num=4, group_num=4):
        super(HyPyraMamba, self).__init__()
        self.mamba_type = mamba_type

        self.patch_embedding = nn.Sequential(
            nn.Conv2d(in_channels=in_channels, out_channels=hidden_dim, kernel_size=1, stride=1, padding=0),
            nn.GroupNorm(group_num, hidden_dim),
            nn.SiLU()
        )

        if mamba_type == 'spa':
            self.mamba = nn.Sequential(
                ImprovedSpaMamba(hidden_dim, use_residual=use_residual, group_num=group_num),
                nn.AvgPool2d(kernel_size=2, stride=2, padding=0),
            )
        elif mamba_type == 'spe':
            self.mamba = nn.Sequential(
                ImprovedSpeMamba(hidden_dim, token_num=token_num, use_residual=use_residual, group_num=group_num),
                nn.AvgPool2d(kernel_size=2, stride=2, padding=0),
            )
        elif mamba_type == 'both':
            self.mamba = nn.Sequential(
                ImprovedBothMamba(hidden_dim, token_num=token_num, use_residual=use_residual, group_num=group_num),
                nn.AvgPool2d(kernel_size=2, stride=2, padding=0),
            )

        self.dynamic_conv = DynamicConvBlock(channels=hidden_dim)

        self.cls_head = nn.Sequential(
            nn.Conv2d(in_channels=hidden_dim, out_channels=128, kernel_size=1, stride=1, padding=0),
            nn.GroupNorm(group_num, 128),
            nn.SiLU(),
            nn.Conv2d(in_channels=128, out_channels=num_classes, kernel_size=1, stride=1, padding=0)
        )
        self.global_pool = nn.AdaptiveAvgPool2d(1)
        self.flat = nn.Flatten()

    def forward(self, x):
        x = self.patch_embedding(x)
        x = self.mamba(x)
        x = self.dynamic_conv(x)
        logits = self.cls_head(x)
        logits = self.global_pool(logits)
        logits = self.flat(logits)
        return logits


@register_model('HyPyraMamba', expects_4d=True, hidden_dim=64, use_residual=True, mamba_type='both', token_num=4, group_num=4)
def hypyramamba_model(pretrained: bool = False, **kwargs) -> HyPyraMamba:
    """Constructs a HyPyraMamba model."""
    if 'bands' in kwargs:
        kwargs['in_channels'] = kwargs.pop('bands')
    kwargs.pop('patch_size', None)
    return HyPyraMamba(**kwargs)
