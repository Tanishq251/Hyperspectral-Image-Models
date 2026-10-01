from models.registry import register_model
"""
IGroupSS-Mamba: Interval Group Spatial-Spectral Mamba for Hyperspectral Image Classification

Paper: https://ieeexplore.ieee.org/document/10816559
GitHub: https://github.com/IIP-Team/IGroupSS-Mamba
Venue: IEEE Transactions on Geoscience and Remote Sensing (TGRS)
Year: 2024

Architecture:
  - 3D convolution front-end to extract spatial-spectral features
  - Interval Group S6 Mechanism (IGSM):
      * Block_Group  : spatial  group Mamba scanning (4 interval directions)
      * Block_SpeGroup: spectral group Mamba scanning (4 interval directions)
  - Interval Group Spatial-Spectral Block (IGSSB):
      * Cascades one spatial + one spectral IGSM operator
      * Followed by an MLP feed-forward block
  - Pixel-aggregation downsampling between stages
  - AdaptiveAvgPool2d + Linear head

Input : 5D tensor [B, 1, bands, patch_size, patch_size]  (expects_4d=False)
Output: class logits [B, num_classes]

Adapted to the unified HSI classification framework.
Block_Group, Block_SpeGroup and MLP_Block are defined in:
  models/helpers/igroupss_mamba_help.py
"""

import torch
import torch.nn as nn
from einops import rearrange
from timm.layers import DropPath, trunc_normal_

from models.helpers.igroupss_mamba_help import Block_Group, Block_SpeGroup, MLP_Block


# ──────────────────────────────────────────────────────────────────────────────
# IGroupSS-Mamba  (VisionMamba in the original paper repository)
# ──────────────────────────────────────────────────────────────────────────────
class IGroupSSMamba(nn.Module):
    """
    IGroupSS-Mamba: Interval Group Spatial-Spectral Mamba.

    Directly mirrors the VisionMamba class in the original
    IGroupSS-Mamba/models/videomamba.py, adapted for the unified framework:

      * Removed hard-coded .cuda() calls  → device-agnostic
      * forward() returns only logits (not (logits, features) tuple)
      * All hyper-parameters are exposed via register_model() defaults

    Args:
        in_chans (int)        : Number of input spectral bands (default 30).
        num_classes (int)     : Number of output classes.
        patch_size (int)      : Spatial patch size (default 13).
        depth (int)           : Number of IGSSB stages (default 3).
        embed_dim (int)       : Embedding dimension (default 32).
        d_state (int)         : SSM state dimension (default 16).
        ssm_ratio (int)       : SSM expansion ratio (default 1).
        k_group (int)         : Number of interval scan groups (default 4).
        scan_type (str)       : Scan pattern ('Interval').
        group_type (str)      : Group type for spatial scan ('Patch').
        conv3D_channel (int)  : 3-D conv output channels (default 32).
        conv3D_kernel (tuple) : 3-D conv kernel (default (3, 3, 3)).
        spa_downks (list)     : Pixel-aggr downsampling [patch, stride] (default [2,1]).
        drop_rate (float)     : Dropout rate (default 0.0).
        drop_path_rate (float): Drop-path rate (default 0.1).
    """

    def __init__(
        self,
        in_chans: int = 30,
        num_classes: int = 9,
        patch_size: int = 13,
        depth: int = 3,
        embed_dim: int = 32,
        d_state: int = 16,
        ssm_ratio: int = 1,
        k_group: int = 4,
        scan_type: str = 'Interval',
        group_type: str = 'Patch',
        conv3D_channel: int = 32,
        conv3D_kernel: tuple = (3, 3, 3),
        spa_downks=None,
        drop_rate: float = 0.0,
        drop_path_rate: float = 0.1,
        **kwargs,
    ):
        super().__init__()

        if spa_downks is None:
            spa_downks = [2, 1]

        self.num_classes  = num_classes
        self.embed_dim    = embed_dim
        self.depth        = depth
        self.scan_type    = scan_type
        self.group_type   = group_type
        self.spa_downks   = spa_downks

        # ── derived spatial / spectral sizes ───────────────────────────────
        # After 3-D conv: spectral dim = in_chans - k[0] + 1
        #                  spatial dim = patch_size - k[1] + 1
        dim_linear = in_chans  - conv3D_kernel[0] + 1   # reduced spectral length
        dim_patch  = patch_size - conv3D_kernel[1] + 1   # reduced spatial size

        # per-stage spectral embedding sizes (spatial downsampling shrinks patch)
        embed_dims_spe = [dim_patch]
        cur = dim_patch
        for _ in range(depth - 1):
            cur = ((cur - spa_downks[0]) // spa_downks[1]) + 1
            embed_dims_spe.append(max(cur, 1))
        # pad to depth+2 to be safe (matches original code)
        while len(embed_dims_spe) < depth + 2:
            nxt = ((embed_dims_spe[-1] - spa_downks[0]) // spa_downks[1]) + 1
            embed_dims_spe.append(max(nxt, 1))

        # ── 3-D convolution front-end ──────────────────────────────────────
        self.conv3d_features = nn.Sequential(
            nn.Conv3d(1, out_channels=conv3D_channel, kernel_size=conv3D_kernel),
            nn.BatchNorm3d(conv3D_channel),
            nn.ReLU(),
        )

        # ── embeddings ─────────────────────────────────────────────────────
        # spatial embedding: flatten (conv3D_channel × dim_linear) → embed_dim
        self.embedding_spatial  = nn.Linear(conv3D_channel * dim_linear, embed_dim)
        # spectral embedding (kept for completeness, not used in main forward)
        self.embedding_spectral = nn.Linear(dim_patch * dim_patch, embed_dim)

        # ── misc layers ────────────────────────────────────────────────────
        self.norm        = nn.LayerNorm(embed_dim)
        self.avgpool     = nn.AdaptiveAvgPool2d(1)
        self.flatten     = nn.Flatten(1)
        self.pos_drop    = nn.Dropout(p=drop_rate)
        self.drop_path   = DropPath(drop_path_rate) if drop_path_rate > 0. else nn.Identity()
        self.head_drop   = nn.Dropout(drop_rate) if drop_rate > 0 else nn.Identity()
        self.head        = nn.Linear(embed_dim, num_classes) if num_classes > 0 else nn.Identity()

        # positional / cls params (kept to match original weight layout)
        self.cls_token              = nn.Parameter(torch.zeros(1, 1, embed_dim))
        self.pos_embed              = nn.Parameter(torch.zeros(1, 1793, embed_dim))
        self.temporal_pos_embedding = nn.Parameter(torch.zeros(1, 28,   embed_dim))

        # ── IGSSB stages ───────────────────────────────────────────────────
        # spatial group Mamba blocks
        self.layers = nn.ModuleList([
            Block_Group(
                scan_type=scan_type,
                group_type=group_type,
                k_group=k_group,
                dim=embed_dim,
                d_state=d_state,
                d_model=embed_dim,
                ssm_ratio=ssm_ratio,
            )
            for _ in range(depth)
        ])

        # spectral group Mamba blocks (each uses its stage-specific spe dim)
        self.layers_spe = nn.ModuleList([
            Block_SpeGroup(
                scan_type=scan_type,
                k_group=k_group,
                dim=embed_dim,
                d_state=d_state,
                d_model=embed_dim,
                d_model_spe=embed_dims_spe[i],
                ssm_ratio=ssm_ratio,
            )
            for i in range(depth)
        ])

        # FFN blocks
        self.FFNs = nn.ModuleList([
            MLP_Block(in_features=embed_dim, hidden_features=embed_dim)
            for _ in range(depth)
        ])

    # ── helpers ────────────────────────────────────────────────────────────

    def _scan_embed(self, x):
        """Rearrange 3-D conv output and project to embed_dim.

        Input : [B, conv3D_channel, dim_linear, H', W']
        Output: [B, H', W', embed_dim]
        """
        # merge channel and spectral axes → (B, C*T, H', W') → (B, H', W', C*T)
        x = rearrange(x, 'b c t h w -> b (c t) h w')
        x = rearrange(x, 'b c h w -> b h w c')
        x = self.embedding_spatial(x)   # (B, H', W', embed_dim)
        return x

    def _downsample(self, x):
        """Pixel-aggregation downsampling (device-agnostic).

        Input : [B, H, W, C]
        Output: [B, H', W', C]  where H' = (H - spa_downks[0]) // spa_downks[1] + 1
        """
        x = rearrange(x, 'b h w c -> b c h w')
        B, C, H, W = x.shape
        ks, st = self.spa_downks
        sz = ((H - ks) // st) + 1
        # allocate on the same device as x
        out = torch.zeros(B, C, sz, sz, device=x.device, dtype=x.dtype)
        for i in range(sz):
            for j in range(sz):
                patch = x[:, :, i*st: i*st+ks, j*st: j*st+ks]
                out[:, :, i, j] = patch.mean(dim=[2, 3])
        return rearrange(out, 'b c h w -> b h w c')

    # ── forward ────────────────────────────────────────────────────────────

    def forward_features(self, x):
        x = self.conv3d_features(x)       # (B,1,C,H,W) → (B, ch, T, H', W')
        x = self._scan_embed(x)           # → (B, H', W', embed_dim)
        x = self.pos_drop(x)

        for i in range(self.depth):
            # spatial IGSM
            x = x + self.drop_path(self.layers[i](self.norm(x)))
            # spectral IGSM
            x = x + self.drop_path(self.layers_spe[i](self.norm(x), group_type='Patch'))
            # FFN
            x = x + self.drop_path(self.FFNs[i](self.norm(x)))
            # pixel-aggregation downsampling (skip at last stage)
            if i != self.depth - 1:
                x = self._downsample(x)

        # (B, H', W', C) → pool → (B, C)
        return self.flatten(self.avgpool(x.permute(0, 3, 1, 2)))

    def forward(self, x):
        feat = self.forward_features(x)
        return self.head(self.head_drop(feat))


# ──────────────────────────────────────────────────────────────────────────────
# Registry
# ──────────────────────────────────────────────────────────────────────────────
@register_model(
    'IGroupSS-Mamba',
    expects_4d=False,          # takes 5-D input [B, 1, bands, H, W] directly
    depth=3,
    embed_dim=32,
    d_state=16,
    ssm_ratio=1,
    k_group=4,
    scan_type='Interval',
    group_type='Patch',
    conv3D_channel=32,
    conv3D_kernel=(3, 3, 3),
    spa_downks=[2, 1],
    drop_rate=0.0,
    drop_path_rate=0.1,
)
def igroupss_mamba(pretrained: bool = False, **kwargs) -> IGroupSSMamba:
    """Constructs an IGroupSS-Mamba model.

    Standardized kwarg mapping:
        bands      → in_chans     (number of spectral bands after PCA)
        num_classes → num_classes  (pass-through)
        patch_size  → patch_size   (pass-through)

    Registry-injected extras that the model doesn't accept are silently dropped.
    """
    if 'bands' in kwargs:
        kwargs['in_chans'] = kwargs.pop('bands')
    kwargs.pop('expects_4d', None)
    kwargs.pop('dual_input', None)
    return IGroupSSMamba(**kwargs)
