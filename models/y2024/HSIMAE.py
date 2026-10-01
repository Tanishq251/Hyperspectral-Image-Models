"""
HSIViT: Spatial-Spectral Vision Transformer for Hyperspectral Image Classification
Downstream supervised model from HSIMAE.

Paper: https://arxiv.org/abs/2403.02324
GitHub: https://github.com/Candy-CY/Hyperspectral-Image-Classification-Models
Year: 2024
"""

import math
import random
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from itertools import product
from einops import rearrange
from timm.layers import DropPath
from models.registry import register_model


# ──────────────────────────────────────────────
# 3D Sin-Cos Position Encoding
# ──────────────────────────────────────────────

def get_3d_sincos_pos_embed(embed_dim, t_size, grid_size, cls_token=False, scale_t=None):
    assert embed_dim % 4 == 0
    embed_dim_spatial = embed_dim // 2
    embed_dim_temporal = embed_dim // 2

    # spatial
    grid_h = np.arange(grid_size, dtype=np.float32)
    grid_w = np.arange(grid_size, dtype=np.float32)
    grid = np.meshgrid(grid_w, grid_h)  # here w goes first
    grid = np.stack(grid, axis=0)

    grid = grid.reshape([2, 1, grid_size, grid_size])
    pos_embed_spatial = get_2d_sincos_pos_embed_from_grid(
        embed_dim_spatial, grid
    )

    # temporal
    grid_t = np.arange(t_size, dtype=np.float32)
    pos_embed_temporal = get_1d_sincos_pos_embed_from_grid(
        embed_dim_temporal, grid_t, scale=scale_t
    )

    # concate: [T, H, W] order
    pos_embed_temporal = pos_embed_temporal[:, np.newaxis, :]
    pos_embed_temporal = np.repeat(pos_embed_temporal, grid_size**2, axis=1)  # [T, H*W, D // 2]

    pos_embed_spatial = pos_embed_spatial[np.newaxis, :, :]
    pos_embed_spatial = np.repeat(pos_embed_spatial, t_size, axis=0)  # [T, H*W, D // 2]

    pos_embed = np.concatenate([pos_embed_temporal, pos_embed_spatial], axis=-1)
    pos_embed = pos_embed.reshape([-1, embed_dim])  # [T*H*W, D]

    if cls_token:
        pos_embed = np.concatenate(
            [np.zeros([1, embed_dim]), pos_embed], axis=0
        )
    return torch.tensor(pos_embed, dtype=torch.float, requires_grad=False).unsqueeze(0)


def get_2d_sincos_pos_embed_from_grid(embed_dim, grid):
    assert embed_dim % 2 == 0

    # use half of dimensions to encode grid_h
    emb_h = get_1d_sincos_pos_embed_from_grid(
        embed_dim // 2, grid[0]
    )  # (H*W, D/2)
    emb_w = get_1d_sincos_pos_embed_from_grid(
        embed_dim // 2, grid[1]
    )  # (H*W, D/2)

    emb = np.concatenate([emb_h, emb_w], axis=1)  # (H*W, D)
    return emb


def get_1d_sincos_pos_embed_from_grid(embed_dim, pos, scale=None):
    assert embed_dim % 2 == 0
    omega = np.arange(embed_dim // 2, dtype=np.float32)
    omega /= embed_dim / 2.0
    omega = 1.0 / 10000 ** omega  # (D/2,)

    pos = pos.reshape(-1)  # (M,)
    if scale is not None:
        pos = pos * scale
    out = np.einsum("m,d->md", pos, omega)  # (M, D/2), outer product

    emb_sin = np.sin(out)  # (M, D/2)
    emb_cos = np.cos(out)  # (M, D/2)

    emb = np.concatenate([emb_sin, emb_cos], axis=1)  # (M, D)
    return emb


# ──────────────────────────────────────────────
# HSI 3D Patch Embedding
# ──────────────────────────────────────────────

class PatchEmbed(nn.Module):
    """HSI to Patch Embedding"""

    def __init__(
        self,
        img_size=9,
        patch_size=3,
        bands=200,
        b_patch_size=8,
        in_chans=1,
        embed_dim=128,
    ):
        super().__init__()
        img_size = (img_size, img_size)
        patch_size = (patch_size, patch_size)
        assert img_size[1] % patch_size[1] == 0
        assert img_size[0] % patch_size[0] == 0
        assert bands % b_patch_size == 0
        num_patches = (
            (img_size[1] // patch_size[1])
            * (img_size[0] // patch_size[0])
            * (bands // b_patch_size)
        )
        self.input_size = (
            bands // b_patch_size,
            img_size[0] // patch_size[0],
            img_size[1] // patch_size[1],
        )

        self.img_size = img_size
        self.patch_size = patch_size
        self.bands = bands
        self.b_patch_size = b_patch_size
        self.num_patches = num_patches

        self.grid_size = img_size[0] // patch_size[0]
        self.b_grid_size = bands // b_patch_size

        kernel_size = [b_patch_size] + list(patch_size)
        self.proj = nn.Conv3d(
            in_chans, embed_dim, kernel_size=kernel_size, stride=kernel_size,
        )

    def forward(self, x):
        B, C, T, H, W = x.shape
        assert (
            H == self.img_size[0] and W == self.img_size[1]
        ), f"Input image size ({H}*{W}) doesn't match model ({self.img_size[0]}*{self.img_size[1]})."
        assert T == self.bands
        x = self.proj(x).flatten(3)
        x = torch.einsum("ncts->ntsc", x)  # [N, T, H*W, C]
        self.output_size = x.shape
        return x


# ──────────────────────────────────────────────
# Attention & SwiGLU Block
# ──────────────────────────────────────────────

class Attention(nn.Module):
    def __init__(
        self,
        dim,
        num_heads=8,
        qkv_bias=False,
        qk_scale=None,
        attn_drop=0.0,
        proj_drop=0.0,
    ):
        super().__init__()
        assert dim % num_heads == 0, "dim should be divisible by num_heads"
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = qk_scale or head_dim**-0.5

        self.q = nn.Linear(dim, dim, bias=qkv_bias)
        self.k = nn.Linear(dim, dim, bias=qkv_bias)
        self.v = nn.Linear(dim, dim, bias=qkv_bias)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

    def forward(self, x, attn_bias=None):
        B, N, C = x.shape
        q = (
            self.q(x)
            .reshape(B, N, self.num_heads, C // self.num_heads)
            .permute(0, 2, 1, 3)
        )
        k = (
            self.k(x)
            .reshape(B, N, self.num_heads, C // self.num_heads)
            .permute(0, 2, 1, 3)
        )
        v = (
            self.v(x)
            .reshape(B, N, self.num_heads, C // self.num_heads)
            .permute(0, 2, 1, 3)
        )

        attn = (q @ k.transpose(-2, -1)) * self.scale
        if attn_bias is not None:
            attn += attn_bias
        attn = attn.softmax(dim=-1)

        x = (attn @ v).transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x


class SwiGLU(nn.Module):
    def __init__(self, dim, hidden_dim, multiple_of=4, dropout=0.):
        super().__init__()
        hidden_dim = int(multiple_of * ((2 * hidden_dim // 3 + multiple_of - 1) // multiple_of))
        self.w1 = nn.Linear(dim, hidden_dim, bias=True)
        self.w2 = nn.Linear(hidden_dim, dim, bias=True)
        self.w3 = nn.Linear(dim, hidden_dim, bias=True)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        return self.dropout(self.w2(F.silu(self.w1(x)) * self.w3(x)))


class Block(nn.Module):
    def __init__(
        self,
        dim,
        num_heads,
        mlp_ratio=4.0,
        qkv_bias=False,
        qk_scale=None,
        drop=0.0,
        attn_drop=0.0,
        drop_path=0.0,
        norm_layer=nn.LayerNorm,
    ):
        super().__init__()
        self.norm1 = norm_layer(dim)
        self.attn = Attention(
            dim,
            num_heads=num_heads,
            qkv_bias=qkv_bias,
            qk_scale=qk_scale,
            attn_drop=attn_drop,
            proj_drop=drop,
        )
        self.drop_path = DropPath(drop_path) if drop_path > 0.0 else nn.Identity()
        self.norm2 = norm_layer(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = SwiGLU(dim, mlp_hidden_dim, mlp_ratio, drop)

    def forward(self, x):
        x = x + self.drop_path(self.attn(self.norm1(x)))
        x = x + self.drop_path(self.mlp(self.norm2(x)))
        return x


# ──────────────────────────────────────────────
# HSIViT Class
# ──────────────────────────────────────────────

class HSIViT(nn.Module):
    def __init__(
        self,
        img_size=9,
        spatial_patch_size=3,
        in_chans=1,
        embed_dim=128,
        depth=12,
        num_heads=8,
        mlp_ratio=4.0,
        norm_layer=nn.LayerNorm,
        bands=200,
        b_patch_size=8,
        num_class=16,
        no_qkv_bias=False,
        trunc_init=True,
        drop_rate=0.,
        drop_path=0.2,
        s_depth=6,
        **kwargs,
    ):
        super().__init__()
        self.trunc_init = trunc_init
        self.b_pred_patch_size = b_patch_size
        self.b_patch_size = b_patch_size
        self.spatial_patch_size = spatial_patch_size

        # Precompute padded dimensions for shape-resiliency
        self.padded_bands = math.ceil(bands / b_patch_size) * b_patch_size
        self.padded_img_size = math.ceil(img_size / spatial_patch_size) * spatial_patch_size

        self.patch_embed = PatchEmbed(
            img_size=self.padded_img_size,
            patch_size=spatial_patch_size,
            bands=self.padded_bands,
            b_patch_size=b_patch_size,
            in_chans=in_chans,
            embed_dim=embed_dim,
        )

        num_patches = self.patch_embed.num_patches
        input_size = self.patch_embed.input_size
        self.input_size = input_size

        self.pos_embed = nn.Parameter(torch.zeros(1, num_patches, embed_dim))

        dpr = [x.item() for x in torch.linspace(0, drop_path, depth)]
        
        # Spatial Encoder Blocks
        if s_depth > 0:
            self.blocks_1 = nn.ModuleList(
                [
                    Block(
                        embed_dim,
                        num_heads,
                        mlp_ratio,
                        qkv_bias=not no_qkv_bias,
                        qk_scale=None,
                        norm_layer=norm_layer,
                        drop_path=dpr[i],
                    )
                    for i in range(s_depth)
                ]
            )

            self.blocks_2 = nn.ModuleList(
                [
                    Block(
                        embed_dim,
                        num_heads,
                        mlp_ratio,
                        qkv_bias=not no_qkv_bias,
                        qk_scale=None,
                        norm_layer=norm_layer,
                        drop_path=dpr[i],
                    )
                    for i in range(s_depth)
                ]
            )

        # Joint Spatial-Spectral Blocks
        if s_depth < depth:
            self.blocks = nn.ModuleList(
                [
                    Block(
                        embed_dim,
                        num_heads,
                        mlp_ratio,
                        qkv_bias=not no_qkv_bias,
                        qk_scale=None,
                        norm_layer=norm_layer,
                        drop_path=dpr[i],
                    )
                    for i in range(s_depth, depth)
                ]
            )

        self.norm = norm_layer(embed_dim)
        self.cls_head = nn.Linear(embed_dim * self.patch_embed.b_grid_size, num_class)

        self.s_depth = s_depth
        self.dim = embed_dim

        self.initialize_weights()

    def initialize_weights(self):
        pos_embed = get_3d_sincos_pos_embed(self.dim, self.input_size[0], self.input_size[1])
        self.pos_embed.data.copy_(pos_embed)
        self.pos_embed.requires_grad = False

        w = self.patch_embed.proj.weight.data
        if self.trunc_init:
            torch.nn.init.trunc_normal_(w, std=0.02)
        else:
            torch.nn.init.xavier_uniform_(w.view([w.shape[0], -1]))

        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            if self.trunc_init:
                nn.init.trunc_normal_(m.weight, std=0.02)
            else:
                torch.nn.init.xavier_uniform_(m.weight)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    def forward_encoder(self, x):
        x = self.patch_embed(x)
        N, T, L, C = x.shape

        x = x.view(N, -1, C) + self.pos_embed

        if self.s_depth > 0:
            x1 = rearrange(x, 'b (t l) c -> (b t) l c', t=T, l=L)
            x2 = rearrange(x, 'b (t l) c -> (b l) t c', t=T, l=L)

            for blk in self.blocks_1:
                x1 = blk(x1)

            for blk in self.blocks_2:
                x2 = blk(x2)

            x1 = rearrange(x1, '(b t) l c -> b (t l) c', b=N, t=T)
            x2 = rearrange(x2, '(b l) t c -> b (t l) c', b=N, l=L)
            x = x1 + x2

        if self.s_depth < len(self.blocks) + self.s_depth:
            for blk in self.blocks:
                x = blk(x)

        x = self.norm(x)
        return x

    def head(self, x, type='AGG'):
        N, T, L, C = self.patch_embed.output_size
        if type == 'GAP':
            x = x.reshape(N, -1, C)
        elif type == 'AGG':
            x = x.reshape(N, T, -1, C)
            x = x.permute(0, 2, 1, 3).flatten(2)
        x = x.mean(1)
        pred = self.cls_head(x)
        return pred, x

    def forward(self, x):
        # x input is standard 4D pipeline tensor [B, C, H, W]
        if x.dim() == 4:
            x = x.unsqueeze(1)  # reshape to [B, 1, C, H, W]

        B, C_in, T, H, W = x.shape

        # Zero padding to fit padded dimensions calculated in __init__
        pad_t = (self.padded_bands - T)
        pad_h = (self.padded_img_size - H)
        pad_w = (self.padded_img_size - W)

        if pad_t > 0 or pad_h > 0 or pad_w > 0:
            x = F.pad(x, (0, pad_w, 0, pad_h, 0, pad_t))

        latent = self.forward_encoder(x)
        pred, latent = self.head(latent)
        return pred


# ──────────────────────────────────────────────
# Registry Factory
# ──────────────────────────────────────────────

@register_model('HSIMAE',
                expects_4d=True,
                embed_dim=128,
                depth=12,
                num_heads=8,
                s_depth=6,
                b_patch_size=8,
                spatial_patch_size=3)
def hsimae_factory(pretrained: bool = False, **kwargs) -> HSIViT:
    """Constructs HSIMAE model."""
    # Standardize keywords passed by standard trainer pipeline
    if 'patch_size' in kwargs:
        kwargs['img_size'] = kwargs.pop('patch_size')
    if 'num_classes' in kwargs:
        kwargs['num_class'] = kwargs.pop('num_classes')
    return HSIViT(**kwargs)
