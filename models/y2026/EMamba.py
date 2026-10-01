from models.registry import register_model
"""
E-Mamba: Efficient Cross-Modal Mamba for HSI-LiDAR Fusion Classification (single-modality HSI variant)

Paper: https://doi.org/10.1016/j.inffus.2025.103328
GitHub: https://github.com/zhangyiyan001/E-Mamba
Venue: Information Fusion
Year: 2026

Adaptation notes:
  - E-Mamba is originally an HSI+LiDAR fusion model: the original
    `MultimodalClassier.forward(x1, x2)` runs an HSI cube (x1) and a LiDAR
    cube (x2, projected to the same channel count) through four *shared*
    VSSBlock stacks and then fuses them with a cross-Mamba attention block +
    a concat-Mamba fusion block, before a linear classifier head.
  - This benchmark provides only single-modality HSI input, so the LiDAR
    branch (`conv_lidar`) and both fusion blocks (`CrossMambaFusionBlock`,
    `ConcatMambaFusionBlock`) are dropped. What remains is exactly the HSI
    stream of the original model — the same four VSSBlocks applied to x1 in
    sequence — followed by the same average-pool classifier + linear head.
    No new layers are introduced; only the parts that require a second
    modality are removed.
  - The custom CUDA kernel `selective_scan_cuda_core` (bundled with this
    repo's own build, not the pip `mamba_ssm` package) is unavailable here.
    It is replaced with the algebraically equivalent pure-PyTorch reference
    scan `mamba_ssm.ops.selective_scan_interface.selective_scan_ref`, used
    exactly as the VMamba-style `forward_corev2` selective-scan needs it.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from functools import partial
from typing import Callable, Any
from collections import OrderedDict
from einops import repeat
from timm.layers import DropPath
from mamba_ssm.ops.selective_scan_interface import selective_scan_ref


# ──────────────────────────────────────────────
# Cross-scan / cross-merge (pure PyTorch)
# ──────────────────────────────────────────────

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


def cross_selective_scan(
    x, x_proj_weight, x_proj_bias, dt_projs_weight, dt_projs_bias,
    A_logs, Ds, out_norm, softmax_version=False, nrows=-1, delta_softplus=True,
):
    B, D, H, W = x.shape
    D, N = A_logs.shape
    K, D, R = dt_projs_weight.shape
    L = H * W

    xs = CrossScan.apply(x)
    x_dbl = torch.einsum("b k d l, k c d -> b k c l", xs, x_proj_weight)
    if x_proj_bias is not None:
        x_dbl = x_dbl + x_proj_bias.view(1, K, -1, 1)
    dts, Bs, Cs = torch.split(x_dbl, [R, N, N], dim=2)
    dts = torch.einsum("b k r l, k d r -> b k d l", dts, dt_projs_weight)

    xs = xs.view(B, -1, L).to(torch.float)
    dts = dts.contiguous().view(B, -1, L).to(torch.float)
    As = -torch.exp(A_logs.to(torch.float))
    Bs = Bs.contiguous().to(torch.float).view(B, K, N, L)
    Cs = Cs.contiguous().to(torch.float).view(B, K, N, L)
    Ds = Ds.to(torch.float)
    delta_bias = dt_projs_bias.view(-1).to(torch.float)

    ys = selective_scan_ref(
        xs, dts, As, Bs, Cs, D=Ds, delta_bias=delta_bias,
        delta_softplus=delta_softplus, return_last_state=False,
    ).view(B, K, -1, H, W)

    y = CrossMerge.apply(ys)

    if softmax_version:
        y = torch.softmax(y, dim=-1).to(x.dtype)
        y = y.transpose(dim0=1, dim1=2).contiguous().view(B, H, W, -1)
    else:
        y = y.transpose(dim0=1, dim1=2).contiguous().view(B, H, W, -1)
        y = out_norm(y).to(x.dtype)
    return y


# ──────────────────────────────────────────────
# Basic blocks
# ──────────────────────────────────────────────

class Permute(nn.Module):
    def __init__(self, *args):
        super().__init__()
        self.args = args

    def forward(self, x):
        return x.permute(*self.args)


class Mlp(nn.Module):
    def __init__(self, in_features, hidden_features=None, out_features=None, act_layer=nn.GELU, drop=0., channels_first=False):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        Linear = partial(nn.Conv2d, kernel_size=1, padding=0) if channels_first else nn.Linear
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


class SS2D(nn.Module):
    def __init__(
        self, d_model=96, d_state=16, ssm_ratio=2, dt_rank="auto", d_conv=3, conv_bias=True,
        dropout=0., bias=False, dt_min=0.001, dt_max=0.1, dt_init="random", dt_scale=1.0,
        dt_init_floor=1e-4, softmax_version=False, **kwargs,
    ):
        factory_kwargs = {"device": None, "dtype": None}
        super().__init__()
        self.softmax_version = softmax_version
        self.d_model = d_model
        self.d_state = d_state
        self.d_conv = d_conv
        self.expand = ssm_ratio
        self.d_inner = int(self.expand * self.d_model)
        self.dt_rank = math_ceil(self.d_model / 16) if dt_rank == "auto" else dt_rank
        self.in_proj = nn.Linear(self.d_model, self.d_inner * 2, bias=bias, **factory_kwargs)

        if self.d_conv > 1:
            self.conv2d = nn.Conv2d(
                in_channels=self.d_inner, out_channels=self.d_inner, groups=self.d_inner,
                bias=conv_bias, kernel_size=d_conv, padding=(d_conv - 1) // 2, **factory_kwargs,
            )
            self.act = nn.SiLU()

        self.K = 4
        self.x_proj = [
            nn.Linear(self.d_inner, (self.dt_rank + self.d_state * 2), bias=False, **factory_kwargs)
            for _ in range(self.K)
        ]
        self.x_proj_weight = nn.Parameter(torch.stack([t.weight for t in self.x_proj], dim=0))
        del self.x_proj

        self.dt_projs = [
            self.dt_init(self.dt_rank, self.d_inner, dt_scale, dt_init, dt_min, dt_max, dt_init_floor, **factory_kwargs)
            for _ in range(self.K)
        ]
        self.dt_projs_weight = nn.Parameter(torch.stack([t.weight for t in self.dt_projs], dim=0))
        self.dt_projs_bias = nn.Parameter(torch.stack([t.bias for t in self.dt_projs], dim=0))
        del self.dt_projs

        self.K2 = self.K
        self.A_logs = self.A_log_init(self.d_state, self.d_inner, copies=self.K2, merge=True)
        self.Ds = self.D_init(self.d_inner, copies=self.K2, merge=True)

        if not self.softmax_version:
            self.out_norm = nn.LayerNorm(self.d_inner)
        self.out_proj = nn.Linear(self.d_inner, self.d_model, bias=bias, **factory_kwargs)
        self.dropout = nn.Dropout(dropout) if dropout > 0. else nn.Identity()

    @staticmethod
    def dt_init(dt_rank, d_inner, dt_scale=1.0, dt_init="random", dt_min=0.001, dt_max=0.1, dt_init_floor=1e-4, **factory_kwargs):
        import math
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

    def forward_core(self, x: torch.Tensor, nrows=-1):
        return cross_selective_scan(
            x, self.x_proj_weight, None, self.dt_projs_weight, self.dt_projs_bias,
            self.A_logs, self.Ds, getattr(self, "out_norm", None), self.softmax_version, nrows=nrows,
        )

    def forward(self, x: torch.Tensor, **kwargs):
        xz = self.in_proj(x)
        if self.d_conv > 1:
            x, z = xz.chunk(2, dim=-1)
            x = x.permute(0, 3, 1, 2).contiguous()
            x = self.act(self.conv2d(x))
            y = self.forward_core(x)
            y = y * F.silu(z) if not self.softmax_version else y * z
        else:
            xz = F.silu(xz)
            x, z = xz.chunk(2, dim=-1)
            x = x.permute(0, 3, 1, 2).contiguous()
            y = self.forward_core(x)
            y = y * z
        out = self.dropout(self.out_proj(y))
        return out


def math_ceil(v):
    import math
    return math.ceil(v)


class VSSBlock(nn.Module):
    def __init__(
        self, dim=0, drop_path=0., norm_layer: Callable[..., torch.nn.Module] = partial(nn.LayerNorm, eps=1e-6),
        attn_drop_rate=0, d_state=16, dt_rank: Any = "auto", ssm_ratio=2.0,
        softmax_version=False, use_checkpoint=False, mlp_ratio=4.0, act_layer=nn.GELU, drop=0.0, **kwargs,
    ):
        super().__init__()
        self.use_checkpoint = use_checkpoint
        self.norm = norm_layer(dim)
        self.op = SS2D(
            d_model=dim, dropout=attn_drop_rate, d_state=d_state, ssm_ratio=ssm_ratio,
            dt_rank=dt_rank, softmax_version=softmax_version, **kwargs
        )
        self.drop_path = DropPath(drop_path)

        self.mlp_branch = mlp_ratio > 0
        if self.mlp_branch:
            self.norm2 = norm_layer(dim)
            mlp_hidden_dim = int(dim * mlp_ratio)
            self.mlp = Mlp(in_features=dim, hidden_features=mlp_hidden_dim, act_layer=act_layer, drop=drop, channels_first=False)

    def forward(self, input: torch.Tensor):
        x = input + self.drop_path(self.op(self.norm(input.permute(0, 2, 3, 1)))).permute(0, 3, 1, 2)
        if self.mlp_branch:
            x = x + self.drop_path(self.mlp(self.norm2(x.permute(0, 2, 3, 1)))).permute(0, 3, 1, 2)
        return x


# ──────────────────────────────────────────────
# E-Mamba (single-modality HSI variant)
# ──────────────────────────────────────────────

class EMambaHSI(nn.Module):
    """HSI-only stream of the original HSI+LiDAR E-Mamba MultimodalClassier:
    the same 4 stacked VSSBlocks applied to the HSI cube, then the same
    average-pool + linear classifier head. The LiDAR branch and cross-modal
    fusion blocks (which require a second modality) are dropped.
    """
    def __init__(self, in_channels=144, num_classes=15, **kwargs):
        super().__init__()
        dim = in_channels
        self.vssblock1 = VSSBlock(dim=dim, drop_path=0.1, d_state=16, mlp_ratio=2.0)
        self.vssblock2 = VSSBlock(dim=dim, drop_path=0.1, d_state=16, mlp_ratio=2.0)
        self.vssblock3 = VSSBlock(dim=dim, drop_path=0.1, d_state=16, mlp_ratio=2.0)
        self.vssblock4 = VSSBlock(dim=dim, drop_path=0.1, d_state=16, mlp_ratio=2.0)
        self.classifier = nn.Sequential(OrderedDict(
            avgpool=nn.AdaptiveAvgPool2d(1),
            flatten=nn.Flatten(1),
        ))
        self.linear = nn.Linear(dim, num_classes, bias=False)

    def forward(self, x):
        x = self.vssblock1(x)
        x = self.vssblock2(x)
        x = self.vssblock3(x)
        x = self.vssblock4(x)
        out = self.classifier(x)
        out = self.linear(out)
        return out


@register_model('EMamba', expects_4d=True)
def emamba(pretrained: bool = False, **kwargs) -> EMambaHSI:
    """Constructs an E-Mamba (single-modality HSI) model."""
    if 'bands' in kwargs:
        kwargs['in_channels'] = kwargs.pop('bands')
    kwargs.pop('patch_size', None)  # EMamba doesn't use patch_size
    return EMambaHSI(**kwargs)
