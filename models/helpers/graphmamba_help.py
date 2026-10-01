"""
GraphMamba helper modules: GCNLayer, GCN, MambaBlock, create_block.

Adapted from: https://github.com/YAT-Graph-Mamba/GraphMamba
Paper: GraphMamba: An Efficient Graph Structure Learning Vision Mamba for HSI Classification

All .cuda() calls removed → device-agnostic.
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from functools import partial
from torch import Tensor
from typing import Optional

from timm.layers import DropPath, trunc_normal_

from mamba_ssm.modules.mamba_simple import Mamba

try:
    from mamba_ssm.ops.triton.layernorm import RMSNorm, layer_norm_fn, rms_norm_fn
except ImportError:
    RMSNorm, layer_norm_fn, rms_norm_fn = None, None, None


# ──────────────────────────────────────────────────────────────────────────────
# GCN components (from GraphMamba/GCN.py — device-agnostic)
# ──────────────────────────────────────────────────────────────────────────────

class GCNLayer(nn.Module):
    """Single GCN layer with batch-norm and LeakyReLU."""
    def __init__(self, input_dim: int, output_dim: int):
        super().__init__()
        self.BN = nn.BatchNorm1d(input_dim)
        self.Activition = nn.LeakyReLU()
        self.sigma1 = nn.Parameter(torch.tensor([0.1], requires_grad=True))
        self.GCN_liner_theta_1 = nn.Sequential(nn.Linear(input_dim, 256))
        self.GCN_liner_out_1 = nn.Sequential(nn.Linear(input_dim, output_dim))

    def A_to_D_inv(self, A: torch.Tensor):
        """Compute D^{-1/2} from adjacency matrix A (device-agnostic)."""
        D = A.sum(2)
        batch, l = D.shape
        D1 = D.reshape(batch * l, 1).squeeze(1)
        # Clamp to avoid division by zero
        D2 = torch.pow(D1.clamp(min=1e-8), -0.5)
        D2 = D2.reshape(batch, l)
        D_hat = torch.zeros(batch, l, l, dtype=torch.float, device=A.device)
        for i in range(batch):
            D_hat[i] = torch.diag(D2[i])
        return D_hat

    def forward(self, H, A):
        nodes_count = A.shape[1]
        I = torch.eye(nodes_count, nodes_count, requires_grad=False, device=A.device)
        A = A + I
        batch, l, c = H.shape
        H1 = H.reshape(batch * l, c)
        H2 = self.BN(H1)
        H = H2.reshape(batch, l, c)
        D_hat = self.A_to_D_inv(A)
        A_hat = torch.matmul(D_hat, torch.matmul(A, D_hat))
        output = torch.matmul(A_hat, self.GCN_liner_out_1(H))
        output = self.Activition(output)
        return output


class GCN(nn.Module):
    """Multi-layer GCN branch used in GraphMamba (device-agnostic)."""
    def __init__(self, height: int, width: int, changel: int, layers_count: int):
        super().__init__()
        self.channel = changel
        self.height = height
        self.width = width
        self.GCN_Branch = nn.Sequential()
        for i in range(layers_count):
            self.GCN_Branch.add_module(
                'GCN_Branch' + str(i),
                GCNLayer(self.channel, self.channel)
            )
        self.BN = nn.BatchNorm1d(changel)

    def forward(self, x: torch.Tensor, A: torch.Tensor):
        H = x
        for i in range(len(self.GCN_Branch)):
            H = self.GCN_Branch[i](H, A)
        return H


# ──────────────────────────────────────────────────────────────────────────────
# Mamba Block (from GraphMamba/Mamba.py — device-agnostic)
# ──────────────────────────────────────────────────────────────────────────────

class MambaBlock(nn.Module):
    """
    Prenorm Mamba block: Add → LN → Mixer, returning hidden_states and residual.
    """
    def __init__(self, dim, mixer_cls, norm_cls=nn.LayerNorm,
                 fused_add_norm=False, residual_in_fp32=False, drop_path=0.):
        super().__init__()
        self.residual_in_fp32 = residual_in_fp32
        self.fused_add_norm = fused_add_norm
        self.mixer = mixer_cls(dim)
        self.norm = norm_cls(dim)
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        if self.fused_add_norm:
            assert RMSNorm is not None, "RMSNorm import fails"
            assert isinstance(self.norm, (nn.LayerNorm, RMSNorm)), \
                "Only LayerNorm and RMSNorm are supported for fused_add_norm"

    def forward(self, hidden_states: Tensor, residual: Optional[Tensor] = None,
                inference_params=None):
        if not self.fused_add_norm:
            if residual is None:
                residual = hidden_states
            else:
                residual = residual + self.drop_path(hidden_states)
            hidden_states = self.norm(residual.to(dtype=self.norm.weight.dtype))
            if self.residual_in_fp32:
                residual = residual.to(torch.float32)
        else:
            fused_add_norm_fn = rms_norm_fn if isinstance(self.norm, RMSNorm) else layer_norm_fn
            if residual is None:
                hidden_states, residual = fused_add_norm_fn(
                    hidden_states, self.norm.weight, self.norm.bias,
                    residual=residual, prenorm=True,
                    residual_in_fp32=self.residual_in_fp32, eps=self.norm.eps,
                )
            else:
                hidden_states, residual = fused_add_norm_fn(
                    self.drop_path(hidden_states), self.norm.weight, self.norm.bias,
                    residual=residual, prenorm=True,
                    residual_in_fp32=self.residual_in_fp32, eps=self.norm.eps,
                )
        hidden_states = self.mixer(hidden_states, inference_params=inference_params)
        return hidden_states, residual

    def allocate_inference_cache(self, batch_size, max_seqlen, dtype=None, **kwargs):
        return self.mixer.allocate_inference_cache(batch_size, max_seqlen, dtype=dtype, **kwargs)


def create_block(
    d_model, ssm_cfg=None, norm_epsilon=1e-5, drop_path=0.,
    rms_norm=False, residual_in_fp32=False, fused_add_norm=False,
    layer_idx=None, device=None, dtype=None, bimamba_type="none",
):
    """Create a single Mamba block."""
    if ssm_cfg is None:
        ssm_cfg = {}
    factory_kwargs = {"device": device, "dtype": dtype}
    mixer_cls = partial(Mamba, layer_idx=layer_idx, **ssm_cfg, **factory_kwargs)

    # Graceful fallback: if RMSNorm is unavailable, use LayerNorm and disable fused ops
    use_rms = rms_norm and (RMSNorm is not None)
    use_fused = fused_add_norm and (RMSNorm is not None)

    norm_cls = partial(
        RMSNorm if use_rms else nn.LayerNorm,
        eps=norm_epsilon, **factory_kwargs
    )
    block = MambaBlock(
        d_model, mixer_cls, norm_cls=norm_cls, drop_path=drop_path,
        fused_add_norm=use_fused, residual_in_fp32=residual_in_fp32,
    )
    block.layer_idx = layer_idx
    return block


# ──────────────────────────────────────────────────────────────────────────────
# Weight initialization helpers
# ──────────────────────────────────────────────────────────────────────────────

def _init_weights(module, n_layer, initializer_range=0.02,
                  rescale_prenorm_residual=True, n_residuals_per_layer=1):
    if isinstance(module, nn.Linear):
        if module.bias is not None:
            if not getattr(module.bias, "_no_reinit", False):
                nn.init.zeros_(module.bias)
    elif isinstance(module, nn.Embedding):
        nn.init.normal_(module.weight, std=initializer_range)

    if rescale_prenorm_residual:
        for name, p in module.named_parameters():
            if name in ["out_proj.weight", "fc2.weight"]:
                nn.init.kaiming_uniform_(p, a=math.sqrt(5))
                with torch.no_grad():
                    p /= math.sqrt(n_residuals_per_layer * n_layer)


def segm_init_weights(m):
    if isinstance(m, nn.Linear):
        trunc_normal_(m.weight, std=0.02)
        if m.bias is not None:
            nn.init.constant_(m.bias, 0)
    elif isinstance(m, nn.LayerNorm):
        nn.init.constant_(m.bias, 0)
        nn.init.constant_(m.weight, 1.0)


# ──────────────────────────────────────────────────────────────────────────────
# Adjacency matrix computation (from GraphMamba/functions.py — device-agnostic)
# ──────────────────────────────────────────────────────────────────────────────

def compute_local_adjacency(x_patch, patch_size, neighbor_l=3, sigma=10.0):
    """
    Compute spectral-similarity weighted adjacency matrix from patch data.

    Args:
        x_patch: [B, H*W, C]  —  flattened patch pixels with spectral features
        patch_size: int  —  spatial size (H = W = patch_size)
        neighbor_l: int  —  neighborhood kernel size (default 3 → 3×3 window)
        sigma: float  —  Gaussian kernel bandwidth

    Returns:
        A: [B, H*W, H*W]  —  adjacency matrix
    """
    B, N, C = x_patch.shape
    device = x_patch.device
    H = W = patch_size

    # Build binary neighborhood mask (same for all samples)
    B_mask = torch.zeros(H * W, H * W, dtype=torch.float32, device=device)
    half_l = (neighbor_l - 1) // 2
    for i in range(H):
        for j in range(W):
            m = i * W + j
            for k in range(neighbor_l):
                for q in range(neighbor_l):
                    ni = i + (k - half_l)
                    nj = j + (q - half_l)
                    n = ni * W + nj
                    if 0 <= ni < H and 0 <= nj < W and m != n:
                        B_mask[m, n] = 1.0

    # Compute pairwise spectral similarity per sample
    # x_patch: [B, N, C]
    # prod[b] = x_patch[b] @ x_patch[b].T  →  [N, N]
    prod = torch.bmm(x_patch, x_patch.transpose(1, 2))       # [B, N, N]
    norm = prod.diagonal(dim1=1, dim2=2).unsqueeze(2)          # [B, N, 1]
    dist_sq = (norm + norm.transpose(1, 2) - 2 * prod).clamp(min=0)  # [B, N, N]
    similarity = torch.exp(-dist_sq / (sigma ** 2))            # [B, N, N]

    # Mask with neighborhood pattern
    A = similarity * B_mask.unsqueeze(0)  # [B, N, N]

    return A
