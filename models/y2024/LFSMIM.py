"""
LFSMIM: Low-frequency Spatial-spectral Masked Image Modeling
Downstream Vision Transformer classifier with CAF (Cross Attention Fusion) skip connections.

Paper/GitHub: https://github.com/Candy-CY/Hyperspectral-Image-Classification-Models
Year: 2024
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange, repeat
from timm.models.layers import trunc_normal_ as __call_trunc_normal_
import numpy as np
from models.registry import register_model


def trunc_normal_(tensor, mean=0., std=1.):
    __call_trunc_normal_(tensor, mean=mean, std=std, a=-std, b=std)


# ──────────────────────────────────────────────
# Transformer Submodules
# ──────────────────────────────────────────────

class Residual(nn.Module):
    def __init__(self, fn):
        super().__init__()
        self.fn = fn

    def forward(self, x, **kwargs):
        return self.fn(x, **kwargs) + x


class PreNorm(nn.Module):
    def __init__(self, dim, fn):
        super().__init__()
        self.norm = nn.LayerNorm(dim)
        self.fn = fn

    def forward(self, x, **kwargs):
        return self.fn(self.norm(x), **kwargs)


class FeedForward(nn.Module):
    def __init__(self, dim, hidden_dim, dropout=0.):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, dim),
            nn.Dropout(dropout)
        )

    def forward(self, x):
        return self.net(x)


class Attention(nn.Module):
    def __init__(self, dim, heads=4, dim_head=16, dropout=0.):
        super().__init__()
        inner_dim = dim_head * heads
        self.heads = heads
        self.temperature = nn.Parameter(torch.log(torch.tensor(dim_head ** -0.5)))
        self.norm = nn.LayerNorm(dim)
        self.attend = nn.Softmax(dim=-1)
        self.dropout = nn.Dropout(dropout)
        self.to_qkv = nn.Linear(dim, inner_dim * 3, bias=False)
        self.to_out = nn.Sequential(
            nn.Linear(inner_dim, dim),
            nn.Dropout(dropout)
        )

    def forward(self, x):
        qkv = self.to_qkv(x).chunk(3, dim=-1)
        q, k, v = map(lambda t: rearrange(t, 'b n (h d) -> b h n d', h=self.heads), qkv)
        dots = torch.matmul(q, k.transpose(-1, -2)) * self.temperature.exp()
        mask = torch.eye(dots.shape[-1], device=dots.device, dtype=torch.bool)
        mask_value = -torch.finfo(dots.dtype).max
        dots = dots.masked_fill(mask, mask_value)
        attn = self.attend(dots)
        attn = self.dropout(attn)
        out = torch.matmul(attn, v)
        out = rearrange(out, 'b h n d -> b n (h d)')
        return self.to_out(out)


class Attention_Re(nn.Module):
    def __init__(self, dim, heads, dim_head, dropout):
        super().__init__()
        inner_dim = dim_head * heads
        self.heads = heads
        self.scale = dim_head ** -0.5

        self.to_qkv = nn.Linear(dim, inner_dim * 3, bias=False)
        self.to_out = nn.Sequential(
            nn.Linear(inner_dim, dim),
            nn.Dropout(dropout)
        )

    def forward(self, x):
        b, n, _, h = *x.shape, self.heads
        qkv = self.to_qkv(x).chunk(3, dim=-1)
        q, k, v = map(lambda t: rearrange(t, 'b n (h d) -> b h n d', h=h), qkv)

        dots = torch.einsum('bhid,bhjd->bhij', q, k) * self.scale
        attn = dots.softmax(dim=-1)
        out = torch.einsum('bhij,bhjd->bhid', attn, v)
        out = rearrange(out, 'b h n d -> b n (h d)')
        out = self.to_out(out)
        return out


class Transformer(nn.Module):
    def __init__(self, dim, depth, heads, dim_head, mlp_head, dropout, num_channel, mode):
        super().__init__()
        self.layers = nn.ModuleList([])
        for _ in range(depth):
            self.layers.append(nn.ModuleList([
                Residual(PreNorm(dim, Attention(dim, heads=heads, dim_head=dim_head, dropout=dropout))),
                Residual(PreNorm(dim, Attention_Re(dim, heads=heads, dim_head=dim_head, dropout=dropout))),
                Residual(PreNorm(dim, FeedForward(dim, mlp_head, dropout=dropout)))
            ]))

        self.mode = mode
        self.skipcat = nn.ModuleList([])
        for _ in range(depth - 2):
            self.skipcat.append(nn.Conv2d(num_channel + 1, num_channel + 1, [1, 2], 1, 0))

    def forward(self, x, mask=None):
        if self.mode == 'ViT':
            for attn, attre, ff in self.layers:
                x = attn(x) + attre(x)
                x = ff(x)
        elif self.mode == 'CAF':
            last_output = []
            nl = 0
            for attn, attre, ff in self.layers:
                last_output.append(x)
                if nl > 1:
                    x = self.skipcat[nl - 2](torch.cat([x.unsqueeze(3), last_output[nl - 2].unsqueeze(3)], dim=3)).squeeze(3)
                x = attn(x) + attre(x)
                x = ff(x)
                nl += 1
        return x


# ──────────────────────────────────────────────
# VisionTransformerEncoder
# ──────────────────────────────────────────────

class VisionTransformerEncoder(nn.Module):
    def __init__(self,
                 image_size=11,
                 near_band=1,
                 num_patches=200,
                 num_classes=16,
                 dim=121,
                 depth=5,
                 heads=4,
                 mlp_dim=12,
                 pool='cls',
                 dim_head=24,
                 dropout=0.1,
                 emb_dropout=0.1,
                 mode='CAF',
                 ):
        super().__init__()
        patch_dim = image_size ** 2 * near_band
        self.use_cls = True
        self.num_classes = num_classes
        self.num_patches = num_patches
        
        if self.use_cls:
            patch_cls = 1
            self.cls_token = nn.Parameter(torch.randn(1, 1, dim))
        else:
            patch_cls = 0
            
        self.pos_embedding = nn.Parameter(torch.randn(1, num_patches + patch_cls, dim))
        self.patch_to_embedding = nn.Linear(patch_dim, dim)
        self.dropout = nn.Dropout(emb_dropout)
        
        self.transformer = Transformer(dim, depth, heads, dim_head, mlp_dim, dropout, num_patches, mode)
        self.pool = pool
        self.to_latent = nn.Identity()
        self.norm = nn.LayerNorm(dim)
        self.mlp_head = nn.Sequential(nn.LayerNorm(dim), nn.Linear(dim, num_classes)) if num_classes > 0 else nn.Identity()

        self.initialize_weights()

    def initialize_weights(self):
        trunc_normal_(self.pos_embedding, std=0.02)
        if self.use_cls:
            trunc_normal_(self.cls_token, std=0.02)

    def forward_features(self, x):
        x = self.patch_to_embedding(x)
        b, n, c = x.shape
        if self.use_cls:
            cls_tokens = repeat(self.cls_token, '() n d -> b n d', b=b)
            x = torch.cat((cls_tokens, x), dim=1)
            x += self.pos_embedding[:, :(n + 1)]
            x = self.dropout(x)
        else:
            x = self.dropout(x + self.pos_embedding)
        return x

    def forward(self, x):
        x = self.forward_features(x)
        x = self.transformer(x)

        if self.num_classes > 0:
            x = x.mean(axis=1) if self.pool == 'mean' else self.to_latent(x[:, 0])
        else:
            if self.use_cls:
                x = x[:, 1:]
            else:
                x = self.to_latent(x)
                
        return self.mlp_head(x)


# ──────────────────────────────────────────────
# LFSMIM Wrapper Model
# ──────────────────────────────────────────────

class LFSMIMModel(nn.Module):
    def __init__(self, num_classes=16, bands=200, patch_size=11, embed_dim=None, depth=5, heads=4, dropout=0.1, **kwargs):
        super().__init__()
        self.bands = bands
        self.image_size = patch_size

        if embed_dim is None:
            embed_dim = patch_size * patch_size

        self.encoder = VisionTransformerEncoder(
            image_size=patch_size,
            near_band=1,
            num_patches=bands,
            num_classes=num_classes,
            dim=embed_dim,
            depth=depth,
            heads=heads,
            dim_head=24,
            mlp_dim=12,
            dropout=dropout,
            emb_dropout=dropout,
            mode='CAF'
        )

    def forward(self, x):
        # Input shape: [B, C, H, W] where C is bands
        B, C, H, W = x.shape

        # Shape resilient padding to match expected image_size
        if H != self.image_size or W != self.image_size:
            if H < self.image_size or W < self.image_size:
                pad_h = self.image_size - H
                pad_w = self.image_size - W
                x = F.pad(x, (0, pad_w, 0, pad_h))
            else:
                x = x[:, :, :self.image_size, :self.image_size]
        
        B, C, H, W = x.shape
        x = x.reshape(B, C, H * W)

        # Resilient padding along the bands count (num_patches)
        if C < self.bands:
            pad_c = self.bands - C
            x = torch.cat([x, torch.zeros((B, pad_c, H * W), device=x.device)], dim=1)
        elif C > self.bands:
            x = x[:, :self.bands, :]

        return self.encoder(x)


# ──────────────────────────────────────────────
# Registry Factory
# ──────────────────────────────────────────────

@register_model('LFSMIM',
                expects_4d=True,
                depth=5,
                heads=4,
                dropout=0.1)
def lfsmim_factory(pretrained: bool = False, **kwargs) -> LFSMIMModel:
    """Constructs LFSMIM model."""
    if 'num_classes' in kwargs:
        kwargs['num_classes'] = kwargs.pop('num_classes')
    return LFSMIMModel(**kwargs)
