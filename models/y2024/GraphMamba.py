from models.registry import register_model
"""
GraphMamba: Graph-enhanced Mamba for Hyperspectral Image Classification

Paper: GraphMamba — An Efficient Graph Structure Learning Vision Mamba
GitHub: https://github.com/YAT-Graph-Mamba/GraphMamba

Architecture:
  - Linear spectral embedding: projects raw spectral bands → embed_dim
  - Stacked Mamba SSM blocks (depth layers)
  - Parallel GCN branches at each layer
  - Cross-attention-fusion (CAF) skip connections via 1×2 conv
  - Per-pixel classification head (center pixel extraction)

The adjacency matrix for the GCN is computed on-the-fly from spectral
similarity within the input patch, so no external pre-processing is needed.

Input : 5D tensor [B, 1, bands, patch_size, patch_size]  (expects_4d=False)
Output: class logits [B, num_classes]

Adapted to the unified HSI classification framework:
  * Removed hard-coded .cuda() calls  → device-agnostic
  * forward() accepts standard 5D input and returns only logits
  * Adjacency matrix computed internally from patch data
  * All hyper-parameters exposed via register_model() defaults

Helper modules defined in:
  models/helpers/graphmamba_help.py
"""

import math
import torch
import torch.nn as nn
from functools import partial
from typing import Optional

from timm.layers import DropPath, trunc_normal_

from models.helpers.graphmamba_help import (
    GCN, create_block, _init_weights, segm_init_weights,
    compute_local_adjacency,
)
from models.helpers.rope import VisionRotaryEmbeddingFast

try:
    from mamba_ssm.ops.triton.layernorm import RMSNorm, layer_norm_fn, rms_norm_fn
except ImportError:
    # Fallback if mamba_ssm not installed
    RMSNorm, layer_norm_fn, rms_norm_fn = None, None, None


# ──────────────────────────────────────────────────────────────────────────────
# GraphMamba  (VisionMamba in the original repository)
# ──────────────────────────────────────────────────────────────────────────────
class GraphMambaHSI(nn.Module):
    """
    GraphMamba: Graph-enhanced Mamba for HSI classification.

    Mirrors the VisionMamba class from the original repository,
    adapted for the unified framework:

      * forward() takes standard 5-D [B, 1, bands, H, W] input
      * Adjacency matrix computed from spectral similarity on-the-fly
      * Center pixel position determined automatically
      * Removed hard-coded .cuda() → device-agnostic
      * forward() returns only logits (not features)

    Args:
        in_chans (int)        : Number of input spectral bands (default 30).
        num_classes (int)     : Number of output classes.
        patch_size (int)      : Spatial patch size (default 11).
        depth (int)           : Number of Mamba + GCN stages (default 6).
        embed_dim (int)       : Embedding dimension (default 64).
        gcn_layers (int)      : Number of GCN layers per branch (default 3).
        sigma (float)         : Gaussian bandwidth for spectral similarity (default 10.0).
        neighbor_l (int)      : Neighborhood kernel size for adjacency (default 3).
        ssm_cfg (dict)        : SSM config passed to Mamba blocks.
        drop_rate (float)     : Dropout rate (default 0.0).
        drop_path_rate (float): Drop-path rate (default 0.1).
        rms_norm (bool)       : Use RMSNorm instead of LayerNorm (default True).
        fused_add_norm (bool) : Use fused add+norm kernels (default True).
        residual_in_fp32 (bool): Keep residual in fp32 (default True).
        if_abs_pos_embed (bool): Use absolute position embedding (default True).
        if_rope (bool)        : Use rotary position embedding (default False).
        if_rope_residual (bool): Apply RoPE to residual (default True).
        if_cls_token (bool)   : Use cls token (default False).
    """

    def __init__(
        self,
        in_chans: int = 30,
        num_classes: int = 16,
        patch_size: int = 11,
        depth: int = 6,
        embed_dim: int = 64,
        gcn_layers: int = 3,
        sigma: float = 10.0,
        neighbor_l: int = 3,
        ssm_cfg=None,
        drop_rate: float = 0.0,
        drop_path_rate: float = 0.1,
        norm_epsilon: float = 1e-5,
        rms_norm: bool = True,
        fused_add_norm: bool = True,
        residual_in_fp32: bool = True,
        if_abs_pos_embed: bool = True,
        if_rope: bool = False,
        if_rope_residual: bool = True,
        if_cls_token: bool = False,
        pt_hw_seq_len: int = 14,
        bimamba_type: str = "v2",
        final_pool_type: str = 'all',
        device=None,
        dtype=None,
        **kwargs,
    ):
        factory_kwargs = {"device": device, "dtype": dtype}
        super().__init__()

        self.residual_in_fp32 = residual_in_fp32
        self.fused_add_norm = fused_add_norm
        self.final_pool_type = final_pool_type
        self.if_abs_pos_embed = if_abs_pos_embed
        self.if_rope = if_rope
        self.if_rope_residual = if_rope_residual
        self.if_cls_token = if_cls_token
        self.num_tokens = 1 if if_cls_token else 0
        self.num_classes = num_classes
        self.d_model = self.num_features = self.embed_dim = embed_dim
        self.patch_size = patch_size
        self.sigma = sigma
        self.neighbor_l = neighbor_l
        self.depth = depth

        # Graceful fallback: if RMSNorm not available, use LayerNorm
        use_rms = rms_norm and (RMSNorm is not None)
        use_fused = fused_add_norm and (RMSNorm is not None)
        self.fused_add_norm = use_fused

        # ── spectral embedding ─────────────────────────────────────────────
        self.patch_to_embedding = nn.Linear(in_chans, embed_dim)
        num_patches = patch_size * patch_size

        # ── optional cls token ──────────────────────────────────────────────
        if if_cls_token:
            self.cls_token = nn.Parameter(torch.zeros(1, 1, self.embed_dim))

        # ── absolute position embedding ─────────────────────────────────────
        if if_abs_pos_embed:
            self.pos_embed = nn.Parameter(
                torch.randn(1, num_patches + self.num_tokens, self.embed_dim)
            )
            self.pos_drop = nn.Dropout(p=drop_rate)

        # ── rotary position embedding (optional) ───────────────────────────
        if if_rope:
            half_head_dim = embed_dim // 2
            self.rope = VisionRotaryEmbeddingFast(
                dim=half_head_dim,
                pt_seq_len=pt_hw_seq_len,
                ft_seq_len=patch_size,
            )

        # ── classification head ─────────────────────────────────────────────
        self.head = nn.Linear(self.num_features, num_classes) if num_classes > 0 else nn.Identity()

        # ── Mamba blocks ────────────────────────────────────────────────────
        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, depth)]
        inter_dpr = [0.0] + dpr
        self.drop_path = DropPath(drop_path_rate) if drop_path_rate > 0. else nn.Identity()

        self.layers = nn.ModuleList([
            create_block(
                embed_dim,
                ssm_cfg=ssm_cfg,
                norm_epsilon=norm_epsilon,
                rms_norm=rms_norm,
                residual_in_fp32=residual_in_fp32,
                fused_add_norm=fused_add_norm,
                layer_idx=i,
                bimamba_type=bimamba_type,
                drop_path=inter_dpr[i],
                **factory_kwargs,
            )
            for i in range(depth)
        ])

        # ── GCN branches ───────────────────────────────────────────────────
        self.layer_GCN = nn.ModuleList([
            GCN(height=patch_size, width=patch_size, changel=embed_dim,
                layers_count=gcn_layers)
            for _ in range(depth)
        ])

        # ── final norm ──────────────────────────────────────────────────────
        self.norm_f = (RMSNorm if use_rms else nn.LayerNorm)(
            embed_dim, eps=norm_epsilon, **factory_kwargs
        )

        self.pre_logits = nn.Identity()

        # ── skip connections (CAF-style) ────────────────────────────────────
        self.skipcat = nn.ModuleList([
            nn.Conv2d(num_patches, num_patches, [1, 2], 1, 0)
            for _ in range(depth - 2)
        ])

        # ── weight init ────────────────────────────────────────────────────
        self.apply(segm_init_weights)
        self.head.apply(segm_init_weights)
        if if_abs_pos_embed:
            trunc_normal_(self.pos_embed, std=.02)

        self.apply(
            partial(_init_weights, n_layer=depth)
        )

    # ── forward helpers ────────────────────────────────────────────────────

    def forward_features(self, x, batch_A, inference_params=None):
        """
        Forward through Mamba + GCN blocks.

        Args:
            x: [B, N, C]  —  embedded patch tokens
            batch_A: [B, N, N]  —  adjacency matrix
        Returns:
            hidden_states: [B, N, embed_dim]
        """
        x = self.patch_to_embedding(x)

        if self.if_abs_pos_embed:
            x = x + self.pos_embed[:, :x.shape[1], :]
            x = self.pos_drop(x)

        residual = None
        hidden_states = x
        last_output = []
        nl = 0

        for mamba_block, gcn_block in zip(self.layers, self.layer_GCN):
            last_output.append(hidden_states)
            if nl > 1:
                hidden_states = self.skipcat[nl - 2](
                    torch.cat([
                        hidden_states.unsqueeze(3),
                        last_output[nl - 2].unsqueeze(3)
                    ], dim=3)
                ).squeeze(3)
            hidden_states, residual = mamba_block(
                hidden_states, residual, inference_params=inference_params
            )
            hidden_states = gcn_block(hidden_states, batch_A)
            nl += 1

        # Final normalization
        if not self.fused_add_norm:
            if residual is None:
                residual = hidden_states
            else:
                residual = residual + self.drop_path(hidden_states)
            hidden_states = self.norm_f(residual.to(dtype=self.norm_f.weight.dtype))
        else:
            fused_add_norm_fn = rms_norm_fn if isinstance(self.norm_f, RMSNorm) else layer_norm_fn
            hidden_states = fused_add_norm_fn(
                self.drop_path(hidden_states),
                self.norm_f.weight,
                self.norm_f.bias,
                eps=self.norm_f.eps,
                residual=residual,
                prenorm=False,
                residual_in_fp32=self.residual_in_fp32,
            )

        return hidden_states

    def forward(self, x, inference_params=None):
        """
        Full forward pass.

        Args:
            x: [B, 1, bands, H, W]  or  [B, bands, H, W]
        Returns:
            logits: [B, num_classes]
        """
        # ── reshape input ──────────────────────────────────────────────────
        if x.dim() == 5 and x.shape[1] == 1:
            x = x.squeeze(1)  # [B, bands, H, W]

        B, C, H, W = x.shape

        # Flatten to [B, H*W, C]  (pixels as tokens, spectral as features)
        x_flat = x.permute(0, 2, 3, 1).reshape(B, H * W, C)

        # ── compute adjacency matrix on-the-fly ───────────────────────────
        batch_A = compute_local_adjacency(
            x_flat, patch_size=self.patch_size,
            neighbor_l=self.neighbor_l, sigma=self.sigma
        )

        # ── center pixel position (always the middle of the patch) ────────
        center_pos_val = (H // 2) * W + (W // 2)

        # ── forward through Mamba + GCN ───────────────────────────────────
        hidden_states = self.forward_features(x_flat, batch_A, inference_params)

        # ── classification head on hidden states ──────────────────────────
        logits = self.head(hidden_states)  # [B, N, num_classes]

        # ── extract center pixel prediction ───────────────────────────────
        x_out = logits[:, center_pos_val, :]  # [B, num_classes]

        return x_out


# ──────────────────────────────────────────────────────────────────────────────
# Registry
# ──────────────────────────────────────────────────────────────────────────────
@register_model(
    'GraphMamba',
    expects_4d=False,            # takes 5-D input [B, 1, bands, H, W] directly
    depth=6,
    embed_dim=64,
    gcn_layers=3,
    sigma=10.0,
    neighbor_l=3,
    drop_rate=0.0,
    drop_path_rate=0.1,
    rms_norm=True,
    fused_add_norm=True,
    residual_in_fp32=True,
    if_abs_pos_embed=True,
    if_rope=False,
    if_rope_residual=True,
    if_cls_token=False,
    bimamba_type='v2',
    final_pool_type='all',
)
def graphmamba(pretrained: bool = False, **kwargs) -> GraphMambaHSI:
    """Constructs a GraphMamba model.

    Standardized kwarg mapping:
        bands       → in_chans     (number of spectral bands after PCA)
        num_classes → num_classes   (pass-through)
        patch_size  → patch_size    (pass-through)

    Registry-injected extras that the model doesn't accept are silently dropped.
    """
    if 'bands' in kwargs:
        kwargs['in_chans'] = kwargs.pop('bands')
    kwargs.pop('expects_4d', None)
    kwargs.pop('dual_input', None)
    return GraphMambaHSI(**kwargs)
