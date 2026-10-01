"""
DSFormer: Dual Selective Fusion Transformer Network for Hyperspectral Image Classification

Paper: https://arxiv.org/abs/2410.03171
GitHub: https://github.com/YichuXu/DSFormer
Venue: Neural Networks 2025
Year: 2025

Adapted from original repo to work with local model registry.
"""
from models.registry import register_model

import torch
from torch import nn
from torch.nn import functional as F
import numbers
from einops import rearrange
from einops.layers.torch import Reduce
from itertools import repeat
import collections.abc
import math
from timm.layers import DropPath


# ──────────────────────────────────────────────
# Helpers
# ──────────────────────────────────────────────

def _ntuple(n):
    def parse(x):
        if isinstance(x, collections.abc.Iterable):
            return x
        return tuple(repeat(x, n))
    return parse

to_1tuple = _ntuple(1)
to_2tuple = _ntuple(2)
to_3tuple = _ntuple(3)
to_4tuple = _ntuple(4)
to_ntuple = _ntuple


def to_3d(x):
    return rearrange(x, 'b c h w -> b (h w) c')


def to_4d(x, h, w):
    return rearrange(x, 'b (h w) c -> b c h w', h=h, w=w)


# ──────────────────────────────────────────────
# Layer Norms
# ──────────────────────────────────────────────

class BiasFree_LayerNorm(nn.Module):
    def __init__(self, normalized_shape):
        super(BiasFree_LayerNorm, self).__init__()
        if isinstance(normalized_shape, numbers.Integral):
            normalized_shape = (normalized_shape,)
        normalized_shape = torch.Size(normalized_shape)
        assert len(normalized_shape) == 1
        self.weight = nn.Parameter(torch.ones(normalized_shape))
        self.normalized_shape = normalized_shape

    def forward(self, x):
        sigma = x.var(-1, keepdim=True, unbiased=False)
        return x / torch.sqrt(sigma + 1e-5) * self.weight


class WithBias_LayerNorm(nn.Module):
    def __init__(self, normalized_shape):
        super(WithBias_LayerNorm, self).__init__()
        if isinstance(normalized_shape, numbers.Integral):
            normalized_shape = (normalized_shape,)
        normalized_shape = torch.Size(normalized_shape)
        assert len(normalized_shape) == 1
        self.weight = nn.Parameter(torch.ones(normalized_shape))
        self.bias = nn.Parameter(torch.zeros(normalized_shape))
        self.normalized_shape = normalized_shape

    def forward(self, x):
        mu = x.mean(-1, keepdim=True)
        sigma = x.var(-1, keepdim=True, unbiased=False)
        return (x - mu) / torch.sqrt(sigma + 1e-5) * self.weight + self.bias


class LayerNorm(nn.Module):
    def __init__(self, dim, LayerNorm_type):
        super(LayerNorm, self).__init__()
        if LayerNorm_type == 'BiasFree':
            self.body = BiasFree_LayerNorm(dim)
        else:
            self.body = WithBias_LayerNorm(dim)

    def forward(self, x):
        h, w = x.shape[-2:]
        return to_4d(self.body(to_3d(x)), h, w)


# ──────────────────────────────────────────────
# Patch Embedding
# ──────────────────────────────────────────────

class PatchEmbed(nn.Module):
    r""" Image to Patch Embedding

    Args:
        img_size (int): Image size.  Default: 224.
        patch_size (int): Patch token size. Default: 4.
        in_chans (int): Number of input image channels. Default: 3.
        embed_dim (int): Number of linear projection output channels. Default: 96.
        norm_layer (nn.Module, optional): Normalization layer. Default: None
    """
    def __init__(self, img_size=224, patch_size=4, in_chans=3, embed_dim=96, norm_layer=None):
        super().__init__()
        img_size = to_2tuple(img_size)
        patch_size = to_2tuple(patch_size)
        patches_resolution = [img_size[0] // patch_size[0], img_size[1] // patch_size[1]]
        self.img_size = img_size
        self.patch_size = patch_size
        self.patches_resolution = patches_resolution
        self.num_patches = patches_resolution[0] * patches_resolution[1]
        self.in_chans = in_chans
        self.embed_dim = embed_dim
        self.proj = nn.Conv2d(in_chans, embed_dim, kernel_size=patch_size, stride=patch_size)
        if norm_layer is not None:
            self.norm = norm_layer(embed_dim)
        else:
            self.norm = None

    def forward(self, x):
        x = self.proj(x)
        Hp, Wp = x.shape[2], x.shape[3]
        x = x.flatten(2).transpose(1, 2)  # b Ph*Pw c
        if self.norm is not None:
            x = self.norm(x)
        return x, (Hp, Wp)


# ──────────────────────────────────────────────
# Core Modules
# ──────────────────────────────────────────────

class Mlp(nn.Module):
    def __init__(self, in_features, hidden_features=None, out_features=None, act_layer=nn.GELU, drop=0.):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = act_layer()
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop = nn.Dropout(drop)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x


# Token Selective Attention
class Token_Selective_Attention(nn.Module):
    def __init__(self, dim, num_heads, bias, k, group_num):
        super(Token_Selective_Attention, self).__init__()
        self.num_heads = num_heads
        self.k = k
        self.group_num = group_num
        self.dim_group = dim // group_num
        self.temperature = nn.Parameter(torch.ones(1, num_heads, 1, 1))

        self.qkv = nn.Conv3d(self.group_num, self.group_num * 3, kernel_size=(1, 1, 1), bias=False)
        self.qkv_conv = nn.Conv3d(self.group_num * 3, self.group_num * 3, kernel_size=(1, 3, 3), padding=(0, 1, 1),
                                  groups=self.group_num * 3, bias=bias)
        self.project_out = nn.Conv2d(dim, dim, kernel_size=1, bias=bias)
        self.attn1 = torch.nn.Parameter(torch.tensor([0.2]), requires_grad=True)
        self.w = nn.Parameter(torch.ones(2))

    def forward(self, x):
        b, c, h, w = x.shape
        x = x.reshape(b, self.group_num, c // self.group_num, h, w)
        b, t, c, h, w = x.shape

        q, k, v = self.qkv_conv(self.qkv(x)).chunk(3, dim=1)

        q = rearrange(q, 'b t (head c) h w -> b head c (h w t)', head=self.num_heads)
        k = rearrange(k, 'b t (head c) h w -> b head c (h w t)', head=self.num_heads)
        v = rearrange(v, 'b t (head c) h w -> b head c (h w t)', head=self.num_heads)

        q = torch.nn.functional.normalize(q, dim=-1)
        k = torch.nn.functional.normalize(k, dim=-1)

        _, _, _, N = q.shape

        mask = torch.zeros(b, self.num_heads, N, N, device=x.device, requires_grad=False)

        attn = (q.transpose(-2, -1) @ k) * self.temperature

        index = torch.topk(attn, k=int(N * self.k), dim=-1, largest=True)[1]
        mask.scatter_(-1, index, 1.)
        attn = torch.where(mask > 0, attn, torch.full_like(attn, float('-inf')))
        attn = attn.softmax(dim=-1)

        out = (attn @ v.transpose(-2, -1)).transpose(-2, -1)

        out = rearrange(out, 'b head c (h w t) -> b t (head c) h w', head=self.num_heads, h=h, w=w)

        out = out.reshape(b, -1, h, w)
        out = self.project_out(out)

        return out


class Mix(nn.Module):
    def __init__(self, m=-0.80):
        super(Mix, self).__init__()
        w = torch.nn.Parameter(torch.FloatTensor([m]), requires_grad=True)
        w = torch.nn.Parameter(w, requires_grad=True)
        self.w = w
        self.mix_block = nn.Sigmoid()

    def forward(self, fea1, fea2):
        mix_factor = self.mix_block(self.w)
        out = fea1 * mix_factor.expand_as(fea1) + fea2 * (1 - mix_factor.expand_as(fea2))
        return out


class block(nn.Module):
    def __init__(self, dim, r=16, L=32):
        super().__init__()
        d = max(dim // r, L)
        self.conv0 = nn.Conv2d(dim, dim, 3, padding=1, groups=dim)
        self.conv_spatial = nn.Conv2d(dim, dim, 5, stride=1, padding=4, groups=dim, dilation=2)
        self.conv1 = nn.Conv2d(dim, dim // 2, 1)
        self.conv2 = nn.Conv2d(dim, dim // 2, 1)
        self.conv_squeeze = nn.Conv2d(2, 2, 7, padding=3)
        self.conv = nn.Conv2d(2, 2, 1)
        self.conv_proj = nn.Conv2d(dim // 2, dim, 1)

        self.global_pool = nn.AdaptiveAvgPool2d(1)
        self.global_maxpool = nn.AdaptiveMaxPool2d(1)
        self.fc1 = nn.Sequential(
            nn.Conv2d(dim, d, 1, bias=False),
            nn.BatchNorm2d(d),
            nn.ReLU(inplace=True)
        )
        self.fc2 = nn.Conv2d(d, dim, 1, 1, bias=False)
        self.softmax = nn.Softmax(dim=1)

    def forward(self, x):
        batch_size = x.size(0)
        dim = x.size(1)
        attn1 = self.conv0(x)
        attn2 = self.conv_spatial(attn1)

        attn1 = self.conv1(attn1)
        attn2 = self.conv2(attn2)

        attn = torch.cat([attn1, attn2], dim=1)
        avg_attn = torch.mean(attn, dim=1, keepdim=True)
        max_attn, _ = torch.max(attn, dim=1, keepdim=True)
        agg = torch.cat([avg_attn, max_attn], dim=1)
        agg = self.conv(agg).sigmoid()

        ch_attn1 = self.global_pool(attn)
        z = self.fc1(ch_attn1)
        a_b = self.fc2(z)
        a_b = a_b.reshape(batch_size, 2, dim // 2, -1)
        a_b = self.softmax(a_b)

        a1, a2 = a_b.chunk(2, dim=1)
        a1 = a1.reshape(batch_size, dim // 2, 1, 1)
        a2 = a2.reshape(batch_size, dim // 2, 1, 1)

        w1 = a1 * agg[:, 0, :, :].unsqueeze(1)
        w2 = a2 * agg[:, 1, :, :].unsqueeze(1)

        attn = attn1 * w1 + attn2 * w2
        attn = self.conv_proj(attn).sigmoid()

        return x * attn


class FFN(nn.Module):
    def __init__(self, dim, bias, kernel_size):
        super(FFN, self).__init__()
        if kernel_size not in [3, 5, 7]:
            raise ValueError("Invalid kernel_size. Must be 3, 5, or 7.")

        self.kernel_size = kernel_size
        hidden_features = 180

        self.project_in = nn.Conv2d(dim, hidden_features, kernel_size=1, bias=bias)
        self.dwconv3x3 = nn.Conv2d(hidden_features, hidden_features, kernel_size=3, stride=1, padding=1,
                                   groups=hidden_features, bias=bias)
        self.dwconv5x5 = nn.Conv2d(hidden_features, hidden_features, kernel_size=5, stride=1, padding=2,
                                   groups=hidden_features, bias=bias)
        self.dwconv7x7 = nn.Conv2d(hidden_features, hidden_features, kernel_size=7, stride=1, padding=3,
                                   groups=hidden_features, bias=bias)
        self.relu3 = nn.ReLU()
        self.project_out = nn.Conv2d(hidden_features, dim, kernel_size=1, bias=bias)

    def forward(self, x):
        x = self.project_in(x)
        if self.kernel_size == 3:
            x = self.relu3(self.dwconv3x3(x))
        elif self.kernel_size == 5:
            x = self.relu3(self.dwconv5x5(x))
        elif self.kernel_size == 7:
            x = self.relu3(self.dwconv7x7(x))
        x = self.project_out(x)
        return x


class Attention_KSB(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.proj_1 = nn.Conv2d(dim, dim, 1)
        self.activation = nn.GELU()
        self.spatial_gating_unit = block(dim)
        self.proj_2 = nn.Conv2d(dim, dim, 1)

    def forward(self, x):
        shorcut = x.clone()
        x = self.proj_1(x)
        x = self.activation(x)
        x = self.spatial_gating_unit(x)
        x = self.proj_2(x)
        x = x + shorcut
        return x


class Mlp_KSB(nn.Module):
    def __init__(self, in_features, hidden_features=None, out_features=None, act_layer=nn.GELU, drop=0.):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Conv2d(in_features, hidden_features, 1)
        self.dwconv = nn.Conv2d(hidden_features, hidden_features, 3, 1, 1, bias=True, groups=hidden_features)
        self.act = act_layer()
        self.fc2 = nn.Conv2d(hidden_features, out_features, 1)
        self.drop = nn.Dropout(drop)

    def forward(self, x):
        x = self.fc1(x)
        x = self.dwconv(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x


# ──────────────────────────────────────────────
# Transformer Blocks
# ──────────────────────────────────────────────

class Transformer_KSFA(nn.Module):
    def __init__(self, dim, mlp_ratio=1., drop=0., drop_path=0., act_layer=nn.GELU):
        super().__init__()
        self.norm1 = nn.BatchNorm2d(dim)
        self.norm2 = nn.BatchNorm2d(dim)
        self.attn = Attention_KSB(dim)
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = Mlp_KSB(in_features=dim, hidden_features=mlp_hidden_dim, act_layer=act_layer, drop=drop)
        layer_scale_init_value = 1e-2
        self.layer_scale_1 = nn.Parameter(
            layer_scale_init_value * torch.ones((dim)), requires_grad=True)
        self.layer_scale_2 = nn.Parameter(
            layer_scale_init_value * torch.ones((dim)), requires_grad=True)

    def forward(self, x):
        x = x + self.drop_path(self.layer_scale_1.unsqueeze(-1).unsqueeze(-1) * self.attn(self.norm1(x)))
        x = x + self.drop_path(self.layer_scale_2.unsqueeze(-1).unsqueeze(-1) * self.mlp(self.norm2(x)))
        return x


class Transformer_TSFA(nn.Module):
    def __init__(self, dim, num_heads, ffn_expansion_factor, bias, LayerNorm_type, kernel_size, k, group_num):
        super(Transformer_TSFA, self).__init__()
        self.norm1 = LayerNorm(dim, LayerNorm_type)
        self.attn_selective = Token_Selective_Attention(dim, num_heads, bias, k, group_num)
        self.norm2 = LayerNorm(dim, LayerNorm_type)
        self.ffn_conv = FFN(dim, bias, kernel_size)

    def forward(self, x):
        x = x + self.attn_selective(self.norm1(x))
        x = x + self.ffn_conv(self.norm2(x))
        return x


# ──────────────────────────────────────────────
# Main Model
# ──────────────────────────────────────────────

class DSFormerModel(nn.Module):
    """
    DSFormer: Dual Selective Fusion Transformer Network for HSI Classification.

    Input: [B, 1, C, H, W] (5D) where C = spectral bands
    Output: [B, num_classes]

    Args:
        num_classes (int): Number of output classes.
        bands (int): Number of input spectral bands.
        embed_dim (int): Embedding dimension. Default: 64.
        img_size (int): Image size for PatchEmbed. Default: 256.
        ps (int): Patch stride for PatchEmbed. Default: 2.
        num_heads (int): Number of attention heads. Default: 8.
        depth (int): Number of TSFA transformer blocks. Default: 6.
        drop_rate (float): Dropout rate. Default: 0.0.
        kernel_size (int): FFN kernel size (3, 5, or 7). Default: 3.
        k (float): Top-k ratio for token selective attention. Default: 0.8.
        group_num (int): Number of groups for grouped attention. Default: 4.
    """
    def __init__(self,
                 num_classes=9,
                 bands=30,
                 embed_dim=64,
                 img_size=256,
                 ps=2,
                 num_heads=8,
                 depth=6,
                 drop_rate=0.,
                 kernel_size=3,
                 k=0.8,
                 group_num=4
                 ):
        super(DSFormerModel, self).__init__()

        self.num_features = embed_dim
        self.conv0 = nn.Conv2d(bands, embed_dim, 3, 1, 1)

        self.patch_embed = PatchEmbed(
            img_size=img_size,
            patch_size=ps,
            in_chans=embed_dim,
            embed_dim=embed_dim,
            norm_layer=nn.LayerNorm)

        self.blocks_TSFA = nn.ModuleList([
            Transformer_TSFA(
                dim=embed_dim, num_heads=num_heads, ffn_expansion_factor=2, bias=False, LayerNorm_type='WithBias',
                kernel_size=kernel_size, k=k, group_num=group_num
            )
            for i in range(depth)])

        self.block_KSFA = Transformer_KSFA(embed_dim)
        self.pos_drop = nn.Dropout(p=drop_rate)
        self.mlp_head = nn.Sequential(
            Reduce('b d h w -> b d', 'mean'),
            nn.LayerNorm(embed_dim),
            nn.Linear(embed_dim, num_classes)
        )

    def forward(self, x):
        b, _, c, h, w = x.size()
        # Squeeze only the channel dim (dim=1). Bare torch.squeeze() also removes
        # the batch dim when b==1 (e.g. a trailing test batch of size 1), which
        # collapses [1,1,C,H,W]->[C,H,W] and breaks downstream shapes
        # ("tuple index out of range").
        x = x.squeeze(1)

        x = self.conv0(x)  # b, dim, PS, PS

        x1, (Hp, Wp) = self.patch_embed(x)
        _, _, dim = x1.size()
        x1 = self.pos_drop(x1)

        x1 = x1.view(b, Hp, Wp, dim)
        x1 = x1.permute(0, 3, 1, 2)

        for i, blk in enumerate(self.blocks_TSFA):
            if i == 0 or i == 3:
                x1 = self.block_KSFA(x1)
            x1 = blk(x1)

        x = self.mlp_head(x1)

        return x


# ──────────────────────────────────────────────
# Registry
# ──────────────────────────────────────────────

@register_model('DSFormer',
                embed_dim=128,
                img_size=256,
                ps=2,
                num_heads=8,
                depth=6,
                drop_rate=0.,
                kernel_size=3,
                k=0.8,
                group_num=4)
def dsformer_factory(pretrained: bool = False, **kwargs) -> DSFormerModel:
    """Construct DSFormer model.

    Standard kwargs: num_classes, bands, patch_size.
    DSFormer-specific: embed_dim, ps, k, group_num, kernel_size, depth.
    """
    # Map standardized 'patch_size' — not used by model directly, but pop it
    kwargs.pop('patch_size', None)

    return DSFormerModel(**kwargs)
