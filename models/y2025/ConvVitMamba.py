from models.registry import register_model
"""
ConvViTMamba: Convolutional Vision Transformer + Mamba block for Hyperspectral Image Classification

Paper: https://doi.org/10.1016/j.knosys.2025.113282
GitHub: https://github.com/mqalkhatib/ConvVitMamba
Venue: Knowledge-Based Systems
Year: 2025

Adaptation notes:
  - Original implementation is TensorFlow/Keras (functional API). Ported to
    PyTorch layer-for-layer, preserving the same pipeline: a multi-scale 3D
    conv front-end (MS_FE: spectral-only, spatial-only, and joint
    spectral-spatial 3D conv branches over the raw HSI cube) -> non-overlapping
    patch tokenization -> a 4-layer standard ViT transformer encoder -> an MLP
    head -> a gated-convolution "Mamba block" (the repo's own lightweight
    conv-gated sequence-mixing block, not `mamba_ssm`) -> global average pool
    -> classifier.
  - Keras `layers.MultiHeadAttention(num_heads, key_dim=projection_dim)` uses
    an internal q/k/v width of `num_heads*key_dim` distinct from the input
    embedding dim, with an output projection back down — this is reproduced
    exactly with a small custom attention module (`KerasStyleMHA`) since
    `nn.MultiheadAttention` assumes embed_dim is evenly split across heads.
  - The final Keras `Dense(num_classes, activation="softmax")` is replaced
    with raw logits (no softmax) per this framework's convention.
  - `MS_FE`'s 3D convs operate on the raw cube (H, W, bands, 1) in Keras'
    channels-last layout; this is the framework's `expects_4d=False` 5D input
    (B, 1, bands, H, W), so `expects_4d=False` is used here.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


# ──────────────────────────────────────────────
# Multi-scale 3D feature extraction (MS_FE)
# ──────────────────────────────────────────────

class Conv3DBlock(nn.Module):
    """Two stacked Conv3d(+ReLU) layers, 'same' padding, matching the
    original `conv3d_block` (Keras kernel order (kh,kw,kd) -> PyTorch (kd,kh,kw))."""
    def __init__(self, in_channels, filters, kernel_hwd):
        super().__init__()
        kh, kw, kd = kernel_hwd
        k = (kd, kh, kw)  # PyTorch Conv3d expects (D, H, W)
        pad = (kd // 2, kh // 2, kw // 2)
        self.conv1 = nn.Conv3d(in_channels, filters, kernel_size=k, padding=pad)
        self.conv2 = nn.Conv3d(filters, filters, kernel_size=k, padding=pad)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        return x


class MS_FE(nn.Module):
    """Multi-scale spectral / spatial / spectral-spatial 3D feature extractor."""
    def __init__(self, bands, num_filters=32):
        super().__init__()
        self.spe = Conv3DBlock(1, num_filters, kernel_hwd=(1, 1, 3))
        self.spa = Conv3DBlock(1, num_filters, kernel_hwd=(3, 3, 1))
        self.ss = Conv3DBlock(1, num_filters, kernel_hwd=(3, 3, 3))
        self.proj = nn.Conv2d(bands * num_filters * 3, num_filters * 3, kernel_size=1)

    def forward(self, x):
        # x: (B, 1, D, H, W)
        x_spe = self.spe(x)
        x_spa = self.spa(x)
        x_ss = self.ss(x)
        x_cat = torch.cat([x_spe, x_spa, x_ss], dim=1)  # (B, 3*F, D, H, W)
        B, C, D, H, W = x_cat.shape
        # Merge (D, C) into a single channel dim, D slower / C faster,
        # matching Keras' Reshape((H, W, D*C)) on a (B,H,W,D,C) tensor.
        x_cat = x_cat.permute(0, 2, 1, 3, 4).contiguous().view(B, D * C, H, W)
        x_out = F.relu(self.proj(x_cat))  # (B, 3*num_filters, H, W)
        return x_out


# ──────────────────────────────────────────────
# Patchify + positional encoding
# ──────────────────────────────────────────────

class PatchEmbed(nn.Module):
    def __init__(self, in_channels, token_size, image_size, projection_dim):
        super().__init__()
        self.token_size = token_size
        self.grid = image_size // token_size
        self.num_patches = self.grid * self.grid
        patch_dim = in_channels * token_size * token_size
        self.proj = nn.Linear(patch_dim, projection_dim)
        self.pos_embed = nn.Parameter(torch.zeros(1, self.num_patches, projection_dim))
        nn.init.trunc_normal_(self.pos_embed, std=0.02)

    def forward(self, x):
        # x: (B, C, H, W)
        crop = self.grid * self.token_size
        x = x[:, :, :crop, :crop]
        patches = F.unfold(x, kernel_size=self.token_size, stride=self.token_size)  # (B, C*ts*ts, N)
        patches = patches.transpose(1, 2)  # (B, N, C*ts*ts)
        x = self.proj(patches) + self.pos_embed
        return x


# ──────────────────────────────────────────────
# Transformer encoder (matches the Keras MultiHeadAttention width convention)
# ──────────────────────────────────────────────

class KerasStyleMHA(nn.Module):
    def __init__(self, embed_dim, num_heads, key_dim, dropout=0.1):
        super().__init__()
        inner = num_heads * key_dim
        self.num_heads = num_heads
        self.key_dim = key_dim
        self.q = nn.Linear(embed_dim, inner)
        self.k = nn.Linear(embed_dim, inner)
        self.v = nn.Linear(embed_dim, inner)
        self.out = nn.Linear(inner, embed_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        B, N, _ = x.shape
        q = self.q(x).view(B, N, self.num_heads, self.key_dim).transpose(1, 2)
        k = self.k(x).view(B, N, self.num_heads, self.key_dim).transpose(1, 2)
        v = self.v(x).view(B, N, self.num_heads, self.key_dim).transpose(1, 2)
        attn = (q @ k.transpose(-2, -1)) / (self.key_dim ** 0.5)
        attn = attn.softmax(dim=-1)
        attn = self.dropout(attn)
        out = attn @ v
        out = out.transpose(1, 2).contiguous().view(B, N, self.num_heads * self.key_dim)
        return self.out(out)


class MLP(nn.Module):
    def __init__(self, in_dim, hidden_units, dropout_rate):
        super().__init__()
        layers = []
        d = in_dim
        for u in hidden_units:
            layers.append(nn.Linear(d, u))
            layers.append(nn.GELU())
            layers.append(nn.Dropout(dropout_rate))
            d = u
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x)


class TransformerBlock(nn.Module):
    def __init__(self, projection_dim, num_heads, transformer_units):
        super().__init__()
        self.ln1 = nn.LayerNorm(projection_dim, eps=1e-6)
        self.attn = KerasStyleMHA(projection_dim, num_heads, key_dim=projection_dim, dropout=0.1)
        self.ln2 = nn.LayerNorm(projection_dim, eps=1e-6)
        self.mlp = MLP(projection_dim, transformer_units, dropout_rate=0.1)

    def forward(self, x):
        x1 = self.ln1(x)
        attn_out = self.attn(x1)
        x2 = attn_out + x
        x3 = self.ln2(x2)
        x3 = self.mlp(x3)
        return x3 + x2


# ──────────────────────────────────────────────
# Gated-conv "Mamba block" (repo's own lightweight sequence mixer)
# ──────────────────────────────────────────────

class MambaBlock(nn.Module):
    def __init__(self, d_model, expand_factor=3, conv_kernel_size=1):
        super().__init__()
        E = expand_factor * d_model
        self.E = E
        self.ln = nn.LayerNorm(d_model, eps=1e-6)
        self.in_proj = nn.Linear(d_model, 2 * E)
        self.conv = nn.Conv1d(E, E, kernel_size=conv_kernel_size, padding=conv_kernel_size // 2)
        self.out_proj = nn.Linear(E, d_model)

    def forward(self, x):
        # x: (B, N, d_model)
        x_norm = self.ln(x)
        x_proj = self.in_proj(x_norm)
        x_content, x_gate = x_proj.split(self.E, dim=-1)
        x_content = self.conv(x_content.transpose(1, 2)).transpose(1, 2)
        x_content = F.gelu(x_content)
        x_gate = torch.sigmoid(x_gate)
        x_mixed = x_content * x_gate
        x_out = self.out_proj(x_mixed)
        return x + x_out


# ──────────────────────────────────────────────
# Full model
# ──────────────────────────────────────────────

class ConvViTMamba(nn.Module):
    def __init__(
        self, in_channels=200, num_classes=16, patch_size=11, token_size=3,
        num_filters=32, projection_dim=32, num_heads=4, transformer_layers=4,
        mlp_head_units=(128, 64), **kwargs,
    ):
        super().__init__()
        image_size = patch_size
        transformer_units = (projection_dim * 2, projection_dim)

        self.fe = MS_FE(in_channels, num_filters=num_filters)
        fe_channels = num_filters * 3
        self.patch_embed = PatchEmbed(fe_channels, token_size, image_size, projection_dim)

        self.blocks = nn.ModuleList([
            TransformerBlock(projection_dim, num_heads, transformer_units)
            for _ in range(transformer_layers)
        ])
        self.final_ln = nn.LayerNorm(projection_dim, eps=1e-6)
        self.head_mlp = MLP(projection_dim, list(mlp_head_units), dropout_rate=0.25)

        d_model = mlp_head_units[-1]
        self.mamba = MambaBlock(d_model, expand_factor=3, conv_kernel_size=1)

        self.dropout = nn.Dropout(0.5)
        self.pre_classifier = nn.Linear(d_model, 64)
        self.classifier = nn.Linear(64, num_classes)

    def forward(self, x):
        # x: (B, 1, D, H, W)
        x = self.fe(x)
        x = self.patch_embed(x)
        for blk in self.blocks:
            x = blk(x)
        x = self.final_ln(x)
        x = self.head_mlp(x)
        x = self.mamba(x)
        x = x.mean(dim=1)  # GlobalAveragePooling1D
        x = self.dropout(x)
        x = F.gelu(self.pre_classifier(x))
        logits = self.classifier(x)
        return logits


@register_model('ConvVitMamba', expects_4d=False, token_size=3, num_filters=32,
                 projection_dim=32, num_heads=4, transformer_layers=4, mlp_head_units=(128, 64))
def convvitmamba(pretrained: bool = False, **kwargs) -> ConvViTMamba:
    """Constructs a ConvViTMamba model."""
    if 'bands' in kwargs:
        kwargs['in_channels'] = kwargs.pop('bands')
    # patch_size here = the spatial window size (H, W); mapped through, not popped.
    return ConvViTMamba(**kwargs)
