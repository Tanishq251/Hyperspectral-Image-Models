from models.registry import register_model
"""
HyperMamba: A Spectral-Spatial Adaptive Mamba for Hyperspectral Image Classification

Paper: https://ieeexplore.ieee.org/document/10614183
GitHub: https://github.com/chiangliu/HyperMamba
Venue: IEEE Transactions on Geoscience and Remote Sensing (TGRS)
Year: 2024

Adaptation notes:
  - The original repo builds on a VMamba (VSSM) backbone with two custom CUDA
    kernels (selective_scan_cuda_core / selective_scan_cuda_oflex) compiled as
    part of the VMamba project itself (not the pip `mamba_ssm` package, and
    not available in this environment). These are replaced here with the
    algebraically-equivalent pure-PyTorch reference scan
    (`mamba_ssm.ops.selective_scan_interface.selective_scan_ref`), which
    supports the same grouped B/C broadcasting the kernels used. This is a
    drop-in numerical substitute, not an architectural change.
  - The model also uses `mmcv.ops.DeformRoIPool` for a 4-scale RoI pooling +
    Gumbel-softmax routing front-end. mmcv (compiled ops) is not installed in
    this environment. Because the original code calls DeformRoIPool with no
    offset tensor, it degenerates to standard box RoI pooling, so it is
    replaced with `torchvision.ops.roi_align` over the same boxes — same
    operation, no mmcv dependency.
  - Triton cross-scan kernels, fvcore FLOP counters, and openmmlab
    backbone-compat wrappers (unused at inference) were dropped as dead code.
"""

import math
from functools import partial
from typing import Callable, Any
from collections import OrderedDict

import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import repeat
from torchvision.ops import roi_align
from mamba_ssm.ops.selective_scan_interface import selective_scan_ref


# ──────────────────────────────────────────────
# Cross-scan / cross-merge (pure PyTorch, no triton)
# ──────────────────────────────────────────────

class CrossScan1D(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x: torch.Tensor):
        B, C, L = x.shape
        ctx.shape = (B, C, L)
        xs = x.new_empty((B, 2, C, L))
        xs[:, 0] = x
        xs[:, 1] = torch.flip(xs[:, 0], dims=[-1])
        return xs

    @staticmethod
    def backward(ctx, ys: torch.Tensor):
        B, C, L = ctx.shape
        y = ys[:, 0] + ys[:, 1].flip(dims=[-1])
        return y


class CrossMerge1D(torch.autograd.Function):
    @staticmethod
    def forward(ctx, ys: torch.Tensor):
        B, K, D, L = ys.shape
        ctx.shape = L
        ys = ys[:, 0] + ys[:, 1].flip(dims=[-1])
        return ys

    @staticmethod
    def backward(ctx, x: torch.Tensor):
        L = ctx.shape
        B, D, L = x.shape
        xs = x.new_empty((B, 2, D, L))
        xs[:, 0] = x
        xs[:, 1] = torch.flip(xs[:, 0], dims=[-1])
        return xs


class CrossScan(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x: torch.Tensor):
        B, C, H, W = x.shape
        ctx.shape = (B, C, H, W)
        xs = x.new_empty((B, 4, C, H * W))
        xs[:, 0] = x.flatten(2, 3)
        xs[:, 1] = x.transpose(dim0=2, dim1=3).flatten(2, 3)
        xs[:, 2:4] = torch.flip(xs[:, 0:2], dims=[-1])
        return xs

    @staticmethod
    def backward(ctx, ys: torch.Tensor):
        B, C, H, W = ctx.shape
        L = H * W
        ys = ys[:, 0:2] + ys[:, 2:4].flip(dims=[-1]).view(B, 2, -1, L)
        y = ys[:, 0] + ys[:, 1].view(B, -1, W, H).transpose(dim0=2, dim1=3).contiguous().view(B, -1, L)
        return y.view(B, -1, H, W)


class CrossMerge(torch.autograd.Function):
    @staticmethod
    def forward(ctx, ys: torch.Tensor):
        B, K, D, H, W = ys.shape
        ctx.shape = (H, W)
        ys = ys.view(B, K, D, -1)
        ys = ys[:, 0:2] + ys[:, 2:4].flip(dims=[-1]).view(B, 2, D, -1)
        y = ys[:, 0] + ys[:, 1].view(B, -1, W, H).transpose(dim0=2, dim1=3).contiguous().view(B, D, -1)
        return y

    @staticmethod
    def backward(ctx, x: torch.Tensor):
        H, W = ctx.shape
        B, C, L = x.shape
        xs = x.new_empty((B, 4, C, L))
        xs[:, 0] = x
        xs[:, 1] = x.view(B, C, H, W).transpose(dim0=2, dim1=3).flatten(2, 3)
        xs[:, 2:4] = torch.flip(xs[:, 0:2], dims=[-1])
        xs = xs.view(B, 4, C, H, W)
        return xs


def att(x):
    """Center-pixel cosine-similarity spatial gating used before op_spectral."""
    x_normalized = F.normalize(x, p=2, dim=3)
    batch_size, H, W, D = x_normalized.shape
    center_h = (H // 2)
    center_w = (W // 2)
    center_vector = x_normalized[:, center_h, center_w, :]
    similarities = torch.einsum('ijkl,il->ijk', (x_normalized, center_vector))
    normalized_similarity = (similarities + 1) / 2
    x = x * normalized_similarity.unsqueeze(-1)
    x = x.mean(dim=[1, 2], keepdim=True)
    return x


# ──────────────────────────────────────────────
# Pure-PyTorch selective-scan bridge
# (replaces selective_scan_cuda_core / selective_scan_cuda_oflex)
# ──────────────────────────────────────────────

class SelectiveScanRef:
    @staticmethod
    def apply(u, delta, A, B, C, D=None, delta_bias=None, delta_softplus=False,
              nrows=1, backnrows=1, oflex=True):
        out, last_state = selective_scan_ref(
            u, delta, A, B, C, D=D, delta_bias=delta_bias,
            delta_softplus=delta_softplus, return_last_state=True,
        )
        return out, last_state


# ──────────────────────────────────────────────
# Basic building blocks
# ──────────────────────────────────────────────

class Linear2d(nn.Linear):
    def forward(self, x: torch.Tensor):
        return F.conv2d(x, self.weight[:, :, None, None], self.bias)


class LayerNorm2d(nn.LayerNorm):
    def forward(self, x: torch.Tensor):
        x = x.permute(0, 2, 3, 1)
        x = nn.functional.layer_norm(x, self.normalized_shape, self.weight, self.bias, self.eps)
        x = x.permute(0, 3, 1, 2)
        return x


class Permute(nn.Module):
    def __init__(self, *args):
        super().__init__()
        self.args = args

    def forward(self, x: torch.Tensor):
        return x.permute(*self.args)


class Mlp(nn.Module):
    def __init__(self, in_features, hidden_features=None, out_features=None, act_layer=nn.GELU, drop=0., channels_first=False):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        Linear = Linear2d if channels_first else nn.Linear
        self.fc1 = Linear(in_features, hidden_features)
        self.act = act_layer()
        self.fc2 = Linear(hidden_features, out_features)
        self.drop = nn.Dropout(drop)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x


class gMlp(nn.Module):
    def __init__(self, in_features, hidden_features=None, out_features=None, act_layer=nn.GELU, drop=0., channels_first=False):
        super().__init__()
        self.channel_first = channels_first
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        Linear = Linear2d if channels_first else nn.Linear
        self.fc1 = Linear(in_features, 2 * hidden_features)
        self.act = act_layer()
        self.fc2 = Linear(hidden_features, out_features)
        self.drop = nn.Dropout(drop)

    def forward(self, x: torch.Tensor):
        x = self.fc1(x)
        x, z = x.chunk(2, dim=(1 if self.channel_first else -1))
        x = self.fc2(x * self.act(z))
        x = self.drop(x)
        return x


class MultiScaleROIPool(nn.Module):
    """Replacement for mmcv.ops.DeformRoIPool (no learned offsets => plain
    box RoI pooling). See module docstring."""
    def __init__(self, output_size=(7, 7)):
        super().__init__()
        self.output_size = output_size

    def forward(self, x, rois):
        return roi_align(x, rois, output_size=self.output_size, spatial_scale=1.0,
                          sampling_ratio=-1, aligned=False)


# ──────────────────────────────────────────────
# Spectral (1D) selective-scan module
# ──────────────────────────────────────────────

class SS1D(nn.Module):
    def __init__(
        self, d_model=96, d_state=16, ssm_ratio=2.0, dt_rank="auto",
        act_layer=nn.SiLU, d_conv=3, conv_bias=True, dropout=0.0, bias=False,
        dt_min=0.001, dt_max=0.1, dt_init="random", dt_scale=1.0, dt_init_floor=1e-4,
        initialize="v0", forward_type="v2", channel_first=False, **kwargs,
    ):
        super().__init__()
        factory_kwargs = {"device": None, "dtype": None}
        d_inner = int(ssm_ratio * d_model)
        dt_rank = math.ceil(d_model / 16) if dt_rank == "auto" else dt_rank
        self.d_conv = d_conv
        self.channel_first = channel_first
        Linear = nn.Linear
        self.out_norm_shape = "v0"
        self.out_norm = nn.LayerNorm(d_inner)
        self.out_norm_state = nn.LayerNorm(d_state)
        self.forward_spectral = partial(self.forward_spectral_core, force_fp32=True, SelectiveScan=SelectiveScanRef)

        k_group = 2
        d_proj = d_inner * 2
        self.in_proj = Linear(d_model, d_proj, bias=bias, **factory_kwargs)
        self.act: nn.Module = act_layer()
        if d_conv > 1:
            self.conv1d = nn.Conv1d(
                in_channels=d_inner, out_channels=d_inner, groups=d_inner, bias=conv_bias,
                kernel_size=d_conv, padding=(d_conv - 1) // 2, **factory_kwargs,
            )
        self.x_proj = [
            nn.Linear(d_inner, (dt_rank + d_state * 2), bias=False, **factory_kwargs)
            for _ in range(k_group)
        ]
        self.x_proj_weight = nn.Parameter(torch.stack([t.weight for t in self.x_proj], dim=0))
        del self.x_proj

        self.out_proj = Linear(d_inner, d_model, bias=bias, **factory_kwargs)
        self.dropout = nn.Dropout(dropout) if dropout > 0. else nn.Identity()

        self.dt_projs = [
            self.dt_init(dt_rank, d_inner, dt_scale, dt_init, dt_min, dt_max, dt_init_floor, **factory_kwargs)
            for _ in range(k_group)
        ]
        self.dt_projs_weight = nn.Parameter(torch.stack([t.weight for t in self.dt_projs], dim=0))
        self.dt_projs_bias = nn.Parameter(torch.stack([t.bias for t in self.dt_projs], dim=0))
        del self.dt_projs

        self.A_logs = self.A_log_init(d_state, d_inner, copies=k_group, merge=True)
        self.Ds = self.D_init(d_inner, copies=k_group, merge=True)
        self.spectral_conv1 = nn.Conv1d(1, d_inner * 2, kernel_size=d_conv, padding=(d_conv - 1) // 2)
        self.spectral_conv2 = nn.Conv1d(d_inner, 1, kernel_size=d_conv, padding=(d_conv - 1) // 2)

    @staticmethod
    def dt_init(dt_rank, d_inner, dt_scale=1.0, dt_init="random", dt_min=0.001, dt_max=0.1, dt_init_floor=1e-4, **factory_kwargs):
        dt_proj = nn.Linear(dt_rank, d_inner, bias=True, **factory_kwargs)
        dt_init_std = dt_rank ** -0.5 * dt_scale
        if dt_init == "constant":
            nn.init.constant_(dt_proj.weight, dt_init_std)
        elif dt_init == "random":
            nn.init.uniform_(dt_proj.weight, -dt_init_std, dt_init_std)
        else:
            raise NotImplementedError
        dt = torch.exp(
            torch.rand(d_inner, **factory_kwargs) * (math.log(dt_max) - math.log(dt_min)) + math.log(dt_min)
        ).clamp(min=dt_init_floor)
        inv_dt = dt + torch.log(-torch.expm1(-dt))
        with torch.no_grad():
            dt_proj.bias.copy_(inv_dt)
        return dt_proj

    @staticmethod
    def A_log_init(d_state, d_inner, copies=-1, device=None, merge=True):
        A = repeat(torch.arange(1, d_state + 1, dtype=torch.float32, device=device), "n -> d n", d=d_inner).contiguous()
        A_log = torch.log(A)
        if copies > 0:
            A_log = repeat(A_log, "d n -> r d n", r=copies)
            if merge:
                A_log = A_log.flatten(0, 1)
        A_log = nn.Parameter(A_log)
        A_log._no_weight_decay = True
        return A_log

    @staticmethod
    def D_init(d_inner, copies=-1, device=None, merge=True):
        D = torch.ones(d_inner, device=device)
        if copies > 0:
            D = repeat(D, "n1 -> r n1", r=copies)
            if merge:
                D = D.flatten(0, 1)
        D = nn.Parameter(D)
        D._no_weight_decay = True
        return D

    def forward_spectral_core(
        self, x=None, delta_softplus=True, out_norm=None, out_norm_shape="v0",
        to_dtype=True, force_fp32=False, nrows=-1, backnrows=-1, ssoflex=True,
        SelectiveScan=None, CrossScan=CrossScan1D, CrossMerge=CrossMerge1D, **kwargs,
    ):
        x_proj_weight = self.x_proj_weight
        dt_projs_weight = self.dt_projs_weight
        dt_projs_bias = self.dt_projs_bias
        A_logs = self.A_logs
        Ds = self.Ds
        out_norm = getattr(self, "out_norm", None)

        B, D, L = x.shape
        D, N = A_logs.shape
        K, D, R = dt_projs_weight.shape

        def selective_scan(u, delta, A, B, C, D=None, delta_bias=None, delta_softplus=True):
            return SelectiveScan.apply(u, delta, A, B, C, D, delta_bias, delta_softplus, nrows, backnrows, ssoflex)

        xs = CrossScan.apply(x)
        x_dbl = torch.einsum("b k d l, k c d -> b k c l", xs, x_proj_weight)
        dts, Bs, Cs = torch.split(x_dbl, [R, N, N], dim=2)
        dts = torch.einsum("b k r l, k d r -> b k d l", dts, dt_projs_weight)

        xs = xs.view(B, -1, L)
        dts = dts.contiguous().view(B, -1, L)
        As = -torch.exp(A_logs.to(torch.float))
        Bs = Bs.contiguous().view(B, K, N, L)
        Cs = Cs.contiguous().view(B, K, N, L)
        Ds = Ds.to(torch.float)
        delta_bias = dt_projs_bias.view(-1).to(torch.float)

        if force_fp32:
            xs = xs.to(torch.float)
            dts = dts.to(torch.float)
            Bs = Bs.to(torch.float)
            Cs = Cs.to(torch.float)

        ys, last_state = selective_scan(xs, dts, As, Bs, Cs, Ds, delta_bias, delta_softplus)
        last_state = last_state.view(B, K, -1, N)
        ys = ys.view(B, K, -1, L)
        y = CrossMerge.apply(ys)
        last_state_spectral = last_state[:, 0] + last_state[:, 1]

        y = y.transpose(dim0=1, dim1=2).contiguous()
        y = out_norm(y).view(B, L, -1)
        y = y.to(x.dtype) if to_dtype else y
        last_state_spectral = last_state_spectral.unsqueeze(1)
        last_state_spectral = self.out_norm_state(last_state_spectral)
        last_state_spectral = last_state_spectral.to(x.dtype) if to_dtype else last_state_spectral
        return y, last_state_spectral

    def forward(self, x: torch.Tensor, **kwargs):
        with_dconv = (self.d_conv > 1)
        B, H, W, C = x.shape
        x = x.view(B * H * W, C).unsqueeze(1)
        x = self.spectral_conv1(x)
        x, z = x.chunk(2, dim=1)
        z = self.act(z.permute(0, 2, 1).contiguous())
        if with_dconv:
            x = self.conv1d(x)
        x = self.act(x)
        y, last_state = self.forward_spectral(x)
        last_state = last_state.view(B, H, W, -1)
        y = y * z
        out = self.dropout(self.out_proj(y))
        out = self.spectral_conv2(out.permute(0, 2, 1)).view(B, H, W, C)
        return out, last_state


# ──────────────────────────────────────────────
# Spatial (2D) selective-scan module
# ──────────────────────────────────────────────

class SS2D(nn.Module):
    def __init__(
        self, d_model=96, d_state=16, ssm_ratio=2.0, dt_rank="auto",
        act_layer=nn.SiLU, d_conv=3, conv_bias=True, dropout=0.0, bias=False,
        dt_min=0.001, dt_max=0.1, dt_init="random", dt_scale=1.0, dt_init_floor=1e-4,
        initialize="v0", forward_type="v2", channel_first=False, **kwargs,
    ):
        super().__init__()
        factory_kwargs = {"device": None, "dtype": None}
        d_inner = int(ssm_ratio * d_model)
        dt_rank = math.ceil(d_model / 16) if dt_rank == "auto" else dt_rank
        self.d_conv = d_conv
        self.channel_first = channel_first
        Linear = nn.Linear
        self.out_norm_shape = "v0"
        self.out_norm = nn.LayerNorm(d_inner)
        self.forward_spatial = partial(self.forward_spatial_core, force_fp32=True, SelectiveScan=SelectiveScanRef)

        k_group = 4
        d_proj = d_inner * 2
        self.in_proj = Linear(d_model, d_proj, bias=bias, **factory_kwargs)
        self.act: nn.Module = act_layer()
        if d_conv > 1:
            self.conv2d = nn.Conv2d(
                in_channels=d_inner, out_channels=d_inner, groups=d_inner, bias=conv_bias,
                kernel_size=d_conv, padding=(d_conv - 1) // 2, **factory_kwargs,
            )
        self.x_proj = [
            nn.Linear(d_inner, (dt_rank + d_state * 2), bias=False, **factory_kwargs)
            for _ in range(k_group)
        ]
        self.x_proj_weight = nn.Parameter(torch.stack([t.weight for t in self.x_proj], dim=0))
        del self.x_proj

        self.out_proj = Linear(d_inner, d_model, bias=bias, **factory_kwargs)
        self.dropout = nn.Dropout(dropout) if dropout > 0. else nn.Identity()

        self.dt_projs = [
            self.dt_init(dt_rank, d_inner, dt_scale, dt_init, dt_min, dt_max, dt_init_floor, **factory_kwargs)
            for _ in range(k_group)
        ]
        self.dt_projs_weight = nn.Parameter(torch.stack([t.weight for t in self.dt_projs], dim=0))
        self.dt_projs_bias = nn.Parameter(torch.stack([t.bias for t in self.dt_projs], dim=0))
        del self.dt_projs

        self.A_logs = self.A_log_init(d_state, d_inner, copies=k_group, merge=True)
        self.Ds = self.D_init(d_inner, copies=k_group, merge=True)

    dt_init = staticmethod(SS1D.dt_init)
    A_log_init = staticmethod(SS1D.A_log_init)
    D_init = staticmethod(SS1D.D_init)

    def forward_spatial_core(
        self, x=None, delta_softplus=True, out_norm=None, out_norm_shape="v0",
        to_dtype=True, force_fp32=False, nrows=-1, backnrows=-1, ssoflex=True,
        SelectiveScan=None, CrossScan=CrossScan, CrossMerge=CrossMerge, **kwargs,
    ):
        x_proj_weight = self.x_proj_weight
        dt_projs_weight = self.dt_projs_weight
        dt_projs_bias = self.dt_projs_bias
        A_logs = self.A_logs
        Ds = self.Ds
        out_norm = getattr(self, "out_norm", None)

        B, D, H, W = x.shape
        D, N = A_logs.shape
        K, D, R = dt_projs_weight.shape
        L = H * W

        def selective_scan(u, delta, A, B, C, D=None, delta_bias=None, delta_softplus=True):
            return SelectiveScan.apply(u, delta, A, B, C, D, delta_bias, delta_softplus, nrows, backnrows, ssoflex)

        xs = CrossScan.apply(x)
        x_dbl = torch.einsum("b k d l, k c d -> b k c l", xs, x_proj_weight)
        dts, Bs, Cs = torch.split(x_dbl, [R, N, N], dim=2)
        dts = torch.einsum("b k r l, k d r -> b k d l", dts, dt_projs_weight)

        xs = xs.view(B, -1, L)
        dts = dts.contiguous().view(B, -1, L)
        As = -torch.exp(A_logs.to(torch.float))
        Bs = Bs.contiguous().view(B, K, N, L)
        Cs = Cs.contiguous().view(B, K, N, L)
        Ds = Ds.to(torch.float)
        delta_bias = dt_projs_bias.view(-1).to(torch.float)

        if force_fp32:
            xs = xs.to(torch.float)
            dts = dts.to(torch.float)
            Bs = Bs.to(torch.float)
            Cs = Cs.to(torch.float)

        ys, last_state = selective_scan(xs, dts, As, Bs, Cs, Ds, delta_bias, delta_softplus)
        ys = ys.view(B, K, -1, H, W)
        y = CrossMerge.apply(ys)

        y = y.transpose(dim0=1, dim1=2).contiguous()
        y = out_norm(y).view(B, H, W, -1)
        y = y.to(x.dtype) if to_dtype else y
        last_state = last_state.to(x.dtype) if to_dtype else last_state
        return y, last_state

    def forward(self, x: torch.Tensor, **kwargs):
        with_dconv = (self.d_conv > 1)
        x = self.in_proj(x)
        x, z = x.chunk(2, dim=-1)
        z = self.act(z)
        x = x.permute(0, 3, 1, 2).contiguous()
        if with_dconv:
            x = self.conv2d(x)
        x = self.act(x)
        y, last_state = self.forward_spatial(x)
        y = y * z
        out = self.dropout(self.out_proj(y))
        return out


# ──────────────────────────────────────────────
# VSS block / stage / backbone
# ──────────────────────────────────────────────

class VSSBlock(nn.Module):
    def __init__(
        self, hidden_dim=0, drop_path=0., norm_layer=nn.LayerNorm, channel_first=False,
        ssm_d_state=16, ssm_ratio=2.0, ssm_dt_rank: Any = "auto", ssm_act_layer=nn.SiLU,
        ssm_conv=3, ssm_conv_bias=True, ssm_drop_rate=0., ssm_init="v0", forward_type="v2",
        mlp_ratio=4.0, mlp_act_layer=nn.GELU, mlp_drop_rate=0.0, gmlp=False,
        use_checkpoint=False, post_norm=False, **kwargs,
    ):
        super().__init__()
        from timm.layers import DropPath
        self.ssm_branch = ssm_ratio > 0
        self.mlp_branch = mlp_ratio > 0
        self.use_checkpoint = use_checkpoint
        self.post_norm = post_norm

        if self.ssm_branch:
            self.norm_spatial = norm_layer(hidden_dim)
            self.op_spatial = SS2D(
                d_model=hidden_dim, d_state=ssm_d_state, ssm_ratio=ssm_ratio, dt_rank=ssm_dt_rank,
                act_layer=ssm_act_layer, d_conv=ssm_conv, conv_bias=ssm_conv_bias,
                dropout=ssm_drop_rate, initialize=ssm_init,
            )
            self.norm_spectral = norm_layer(hidden_dim)
            self.op_spectral = SS1D(
                d_model=hidden_dim, d_state=ssm_d_state, ssm_ratio=ssm_ratio, dt_rank=ssm_dt_rank,
                act_layer=ssm_act_layer, d_conv=ssm_conv, conv_bias=ssm_conv_bias,
                dropout=ssm_drop_rate,
            )

        self.drop_path_spatial = DropPath(drop_path)
        self.drop_path_spectral = DropPath(drop_path)

        if self.mlp_branch:
            _MLP = Mlp if not gmlp else gMlp
            self.norm2 = norm_layer(hidden_dim)
            mlp_hidden_dim = int(hidden_dim * mlp_ratio)
            self.mlp = _MLP(in_features=hidden_dim, hidden_features=mlp_hidden_dim, act_layer=mlp_act_layer, drop=mlp_drop_rate, channels_first=channel_first)

    def _forward(self, input: torch.Tensor):
        if self.ssm_branch:
            if self.post_norm:
                x = input + self.drop_path_spatial(self.norm_spatial(self.op_spatial(input)))
            else:
                x_spatial = self.drop_path_spatial(self.op_spatial(self.norm_spatial((input))))
                x = input + x_spatial
        else:
            x = input
        if self.mlp_branch:
            if self.post_norm:
                x = x + self.drop_path_spatial(self.norm2(self.mlp(x)))
            else:
                x = x + self.drop_path_spatial(self.mlp(self.norm2(x)))
        return x

    def forward(self, input: torch.Tensor):
        return self._forward(input)


class VSSM(nn.Module):
    def __init__(
        self, patch_size=1, in_chans=200, num_classes=16, depths=(1,), dims=96,
        ssm_d_state=16, ssm_ratio=1.0, ssm_dt_rank="auto", ssm_act_layer="silu",
        ssm_conv=3, ssm_conv_bias=True, ssm_drop_rate=0.0, ssm_init="v0", forward_type="v2",
        mlp_ratio=0.0, mlp_act_layer="gelu", mlp_drop_rate=0.0, gmlp=False,
        drop_path_rate=0.2, patch_norm=True, norm_layer="ln",
        downsample_version="v2", patchembed_version="v1", use_checkpoint=False, **kwargs,
    ):
        super().__init__()
        self.channel_first = (norm_layer.lower() in ["bn", "ln2d"])
        self.num_classes = num_classes
        depths = list(depths)
        self.num_layers = len(depths)
        if isinstance(dims, int):
            dims = [int(dims) for _ in range(self.num_layers)]
        self.num_features = dims[-1]
        self.dims = dims
        self.drop_path = None

        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, sum(depths))]

        _NORMLAYERS = dict(ln=nn.LayerNorm, ln2d=LayerNorm2d, bn=nn.BatchNorm2d)
        _ACTLAYERS = dict(silu=nn.SiLU, gelu=nn.GELU, relu=nn.ReLU, sigmoid=nn.Sigmoid)

        norm_layer_cls = _NORMLAYERS.get(norm_layer.lower(), nn.LayerNorm)
        ssm_act_layer_cls = _ACTLAYERS.get(ssm_act_layer.lower(), nn.SiLU)
        mlp_act_layer_cls = _ACTLAYERS.get(mlp_act_layer.lower(), nn.GELU)

        self.patch_embed = self._make_patch_embed(in_chans, dims[0], patch_size, patch_norm, norm_layer_cls, channel_first=self.channel_first)

        self.deformpool = MultiScaleROIPool(output_size=(7, 7))
        self.deform_fc_channels = 128
        self.router = nn.Sequential(
            nn.Linear(in_chans, self.deform_fc_channels),
            nn.ReLU(inplace=True),
            nn.Linear(self.deform_fc_channels, self.deform_fc_channels),
            nn.ReLU(inplace=True),
            nn.Linear(self.deform_fc_channels, 4),
        )

        self.op_spectral = SS1D(
            d_model=dims[0], d_state=ssm_d_state, ssm_ratio=ssm_ratio, dt_rank=ssm_dt_rank,
            act_layer=ssm_act_layer_cls, d_conv=ssm_conv, conv_bias=ssm_conv_bias, dropout=ssm_drop_rate,
        )

        self.layers = nn.ModuleList()
        for i_layer in range(self.num_layers):
            downsample = nn.Identity()
            blocks = [
                VSSBlock(
                    hidden_dim=dims[i_layer], drop_path=dpr[sum(depths[:i_layer]) + d],
                    norm_layer=norm_layer_cls, channel_first=self.channel_first,
                    ssm_d_state=ssm_d_state, ssm_ratio=ssm_ratio, ssm_dt_rank=ssm_dt_rank,
                    ssm_act_layer=ssm_act_layer_cls, ssm_conv=ssm_conv, ssm_conv_bias=ssm_conv_bias,
                    ssm_drop_rate=ssm_drop_rate, ssm_init=ssm_init, forward_type=forward_type,
                    mlp_ratio=mlp_ratio, mlp_act_layer=mlp_act_layer_cls, mlp_drop_rate=mlp_drop_rate,
                    gmlp=gmlp, use_checkpoint=use_checkpoint,
                ) for d in range(depths[i_layer])
            ]
            self.layers.append(nn.Sequential(OrderedDict(blocks=nn.Sequential(*blocks), downsample=downsample)))

        self.norm = norm_layer_cls(self.num_features)
        self.drop_path_final = None
        from timm.layers import DropPath
        self.drop_path_final = DropPath(drop_path_rate)
        self.classifier = nn.Sequential(OrderedDict(
            norm=norm_layer_cls(self.num_features),
            permute=(Permute(0, 3, 1, 2) if not self.channel_first else nn.Identity()),
            avgpool=nn.AdaptiveAvgPool2d(1),
            flatten=nn.Flatten(1),
            head=nn.Linear(self.num_features, num_classes),
        ))

        self.apply(self._init_weights)

    def _init_weights(self, m: nn.Module):
        if isinstance(m, nn.Linear):
            from timm.layers import trunc_normal_
            trunc_normal_(m.weight, std=.02)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    @staticmethod
    def _make_patch_embed(in_chans=3, embed_dim=96, patch_size=4, patch_norm=True, norm_layer=nn.LayerNorm, channel_first=False):
        return nn.Sequential(
            nn.Conv2d(in_chans, embed_dim, kernel_size=patch_size, stride=patch_size, bias=True),
            (nn.Identity() if channel_first else Permute(0, 2, 3, 1)),
            (norm_layer(embed_dim) if patch_norm else nn.Identity()),
        )

    def forward(self, x: torch.Tensor):
        batch_size, D, H, W = x.shape
        device = x.device
        center_h = (H // 2)
        center_w = (W // 2)
        batchs = torch.arange(0, batch_size, device=device)

        def make_rois(inset):
            r = torch.tensor([[0, float(inset), float(inset), float(H - inset), float(W - inset)]], dtype=torch.float32, device=device)
            r = r.repeat(batch_size, 1)
            r[:, 0] = batchs
            return r

        rois_1 = make_rois(0)
        rois_2 = make_rois(1)
        rois_3 = make_rois(2)
        rois_4 = make_rois(3)

        route_logits = self.router(x[:, :, center_h, center_w])
        route_scores = F.gumbel_softmax(route_logits, hard=True).unsqueeze(-1).unsqueeze(-1).unsqueeze(-1)
        x1 = self.deformpool(x, rois_1)
        x2 = self.deformpool(x, rois_2)
        x3 = self.deformpool(x, rois_3)
        x4 = self.deformpool(x, rois_4)
        x = x1 * route_scores[:, 0] + x2 * route_scores[:, 1] + x3 * route_scores[:, 2] + x4 * route_scores[:, 3]

        x = self.patch_embed(x)
        x = self.layers[0](x)
        x_spatial = att(x)
        x_spectral = self.op_spectral(x_spatial)[0]
        x = x_spatial + self.drop_path_final(self.norm(x_spectral))
        x = self.classifier(x)
        return x


# ──────────────────────────────────────────────
# Register model
# ──────────────────────────────────────────────

@register_model('HyperMamba', expects_4d=True, patch_size=1, depths=(1,), dims=96,
                 ssm_d_state=16, ssm_ratio=1.0, ssm_dt_rank="auto", ssm_act_layer="silu",
                 ssm_conv=3, mlp_ratio=0.0, drop_path_rate=0.2, patch_norm=True,
                 norm_layer="ln", downsample_version="v2", patchembed_version="v1")
def hypermamba(pretrained: bool = False, **kwargs) -> VSSM:
    """Constructs a HyperMamba model."""
    if 'bands' in kwargs:
        kwargs['in_chans'] = kwargs.pop('bands')
    # `patch_size` here is HyperMamba's own patch-embedding conv kernel/stride
    # (kept at 1 to preserve the small spatial window), not the benchmark's
    # spatial patch_size (the 11x11 window is already the model's input H,W).
    kwargs.pop('patch_size', None)
    return VSSM(**kwargs)
