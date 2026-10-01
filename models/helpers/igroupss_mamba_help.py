"""
IGroupSS-Mamba helper modules: Block_Group, Block_SpeGroup, MLP_Block

Adapted from: https://github.com/IGroupSS/IGroupSS-Mamba
Paper: IGroupSS-Mamba: Interval Group Spatial-Spectral Mamba for Hyperspectral Image Classification
"""

import math
import torch
import torch.nn as nn
from torch import Tensor
from typing import Optional
from einops import rearrange, repeat
from timm.layers import DropPath

from mamba_ssm.modules.mamba_simple import Mamba


# ──────────────────────────────────────────────
# Selective Scan import (same as IGroupSS-Mamba)
# ──────────────────────────────────────────────
try:
    import selective_scan_cuda
except Exception:
    selective_scan_cuda = None


class SelectiveScanMamba(torch.autograd.Function):
    """Selective scan forward/backward using mamba CUDA kernels."""
    @staticmethod
    @torch.amp.custom_fwd(device_type='cuda')
    def forward(ctx, u, delta, A, B, C, D=None, delta_bias=None, delta_softplus=False, nrows=1, backnrows=1, oflex=True):
        ctx.delta_softplus = delta_softplus
        out, x, *rest = selective_scan_cuda.fwd(u, delta, A, B, C, D, None, delta_bias, delta_softplus)
        ctx.save_for_backward(u, delta, A, B, C, D, delta_bias, x)
        return out

    @staticmethod
    @torch.amp.custom_bwd(device_type='cuda')
    def backward(ctx, dout, *args):
        u, delta, A, B, C, D, delta_bias, x = ctx.saved_tensors
        if dout.stride(-1) != 1:
            dout = dout.contiguous()
        du, ddelta, dA, dB, dC, dD, ddelta_bias, *rest = selective_scan_cuda.bwd(
            u, delta, A, B, C, D, None, delta_bias, dout, x, None, None, ctx.delta_softplus, False
        )
        return (du, ddelta, dA, dB, dC, dD, ddelta_bias, None, None, None, None)


# ──────────────────────────────────────────────
# Mamba initialization helpers
# ──────────────────────────────────────────────
class mamba_init:
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
        A = repeat(
            torch.arange(1, d_state + 1, dtype=torch.float32, device=device),
            "n -> d n", d=d_inner,
        ).contiguous()
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


# ──────────────────────────────────────────────
# Block_Group — Spatial group scanning
# ──────────────────────────────────────────────
class Block_Group(nn.Module, mamba_init):
    def __init__(self,
                 scan_type=None,
                 group_type=None,
                 k_group=None,
                 dim=None,
                 dt_rank="auto",
                 d_inner=None,
                 d_state=None,
                 d_model=None,
                 ssm_ratio=None,
                 bimamba=None,
                 seq=False,
                 force_fp32=True,
                 dropout=0.0,
                 **kwargs):
        super().__init__()
        act_layer = nn.SiLU
        dt_min = 0.001
        dt_max = 0.1
        dt_init = "random"
        dt_scale = 1.0
        dt_init_floor = 1e-4
        bias = False
        self.force_fp32 = force_fp32
        self.seq = seq
        self.k_group = k_group
        self.group_type = group_type
        self.scan_type = scan_type
        d_inner = int(ssm_ratio * d_model)

        self.fc1 = nn.Linear(dim, 4, bias=True)
        self.fc2 = nn.Linear(4, dim, bias=True)
        self.relu = nn.ReLU()
        self.sigmoid = nn.Sigmoid()

        # in proj
        self.in_proj = nn.Linear(dim, d_inner * 2, bias=bias)
        self.act = act_layer()
        self.forward_conv1d = nn.Conv1d(in_channels=d_inner, out_channels=d_inner, kernel_size=1)
        self.conv2d = nn.Conv2d(in_channels=d_inner, out_channels=d_inner, groups=d_inner, bias=True, kernel_size=(1, 1))
        self.conv3d = nn.Conv3d(in_channels=d_inner, out_channels=d_inner, groups=d_inner, bias=True, kernel_size=(1, 1, 1))

        # out proj
        self.out_norm = nn.LayerNorm(d_inner)
        self.out_proj = nn.Linear(d_inner, dim, bias=bias)
        self.dropout = nn.Dropout(dropout) if dropout > 0. else nn.Identity()

        # x proj
        d_inner = int(ssm_ratio * (d_model // 4))
        dt_rank = math.ceil((d_model // 4) / 16) if dt_rank == "auto" else dt_rank
        self.x_proj = [nn.Linear(d_inner, (dt_rank + d_state * 2), bias=False) for _ in range(k_group)]
        self.x_proj_weight = nn.Parameter(torch.stack([t.weight for t in self.x_proj], dim=0))
        del self.x_proj

        # dt proj
        self.dt_projs = [self.dt_init(dt_rank, d_inner, dt_scale, dt_init, dt_min, dt_max, dt_init_floor) for _ in range(k_group)]
        self.dt_projs_weight = nn.Parameter(torch.stack([t.weight for t in self.dt_projs], dim=0))
        self.dt_projs_bias = nn.Parameter(torch.stack([t.bias for t in self.dt_projs], dim=0))
        del self.dt_projs

        # A, D
        self.A_logs = self.A_log_init(d_state, d_inner, copies=k_group, merge=True)
        self.Ds = self.D_init(d_inner, copies=k_group, merge=True)

    def scan(self, x, scan_type=None, group_type=None, route=None):
        if scan_type == 'Interval':
            x1 = x[:, 0::4, :, :]
            x2 = x[:, 1::4, :, :]
            x3 = x[:, 2::4, :, :]
            x4 = x[:, 3::4, :, :]
            xs1 = x1.view(self.B, -1, self.L).view(self.B, 1, -1, self.L)
            xs2 = torch.transpose(x2, dim0=2, dim1=3).contiguous().view(self.B, -1, self.L).view(self.B, 1, -1, self.L)
            xs3 = x3.view(self.B, -1, self.L).view(self.B, 1, -1, self.L)
            xs3 = torch.flip(xs3, dims=[-1])
            xs4 = torch.transpose(x4, dim0=2, dim1=3).contiguous().view(self.B, -1, self.L).view(self.B, 1, -1, self.L)
            xs4 = torch.flip(xs4, dims=[-1])
            xs = torch.stack([xs1, xs2, xs3, xs4], dim=1).view(self.B, 4, -1, self.L)
        return xs

    def Interval_Combine(self, vectors):
        num = len(vectors)
        B, H, W, L = vectors[0].shape
        device = vectors[0].device
        merged_vector = torch.zeros(B, H, W, num * L, device=device)
        for j in range(L):
            for i in range(num):
                merged_vector[:, :, :, j * num + i] = vectors[i][:, :, :, j]
        return merged_vector

    def forward(self, x: Tensor, route=None):
        x = self.in_proj(x)
        x, z = x.chunk(2, dim=-1)
        z = self.act(z)

        if self.group_type == 'Patch':
            x = x.permute(0, 3, 1, 2).contiguous()
            x = self.conv2d(x)
            x = self.act(x)

        zz = x.mean(dim=2).mean(dim=2)
        fc_out_1 = self.relu(self.fc1(zz))
        fc_out_2 = self.sigmoid(self.fc2(fc_out_1))

        def selective_scan(u, delta, A, B, C, D=None, delta_bias=None, delta_softplus=True, nrows=1):
            return SelectiveScanMamba.apply(u, delta, A, B, C, D, delta_bias, delta_softplus, nrows, False)

        if len(x.size()) == 4:
            B, D, H, W = x.shape
            L = H * W
        elif len(x.size()) == 3:
            B, D, L = x.shape
        elif len(x.size()) == 5:
            B, D, T, H, W = x.shape
            L = T * H * W
        self.B = B
        self.L = L
        D_A, N = self.A_logs.shape
        K, D_dt, R = self.dt_projs_weight.shape

        xs = self.scan(x, scan_type=self.scan_type, group_type=self.group_type, route=route)

        x_dbl = torch.einsum("b k d l, k c d -> b k c l", xs, self.x_proj_weight)
        dts, Bs, Cs = torch.split(x_dbl, [R, N, N], dim=2)
        dts = torch.einsum("b k r l, k d r -> b k d l", dts, self.dt_projs_weight)

        xs = xs.view(B, -1, L)
        dts = dts.contiguous().view(B, -1, L)
        Bs = Bs.contiguous()
        Cs = Cs.contiguous()

        As = -torch.exp(self.A_logs.float())
        Ds = self.Ds.float()
        dt_projs_bias = self.dt_projs_bias.float().view(-1)

        to_fp32 = lambda *args: (_a.to(torch.float32) for _a in args)

        if self.force_fp32:
            xs, dts, Bs, Cs = to_fp32(xs, dts, Bs, Cs)

        if self.seq:
            out_y = []
            for i in range(self.k_group):
                yi = selective_scan(
                    xs.view(B, K, -1, L)[:, i], dts.view(B, K, -1, L)[:, i],
                    As.view(K, -1, N)[i], Bs[:, i].unsqueeze(1), Cs[:, i].unsqueeze(1), Ds.view(K, -1)[i],
                    delta_bias=dt_projs_bias.view(K, -1)[i],
                    delta_softplus=True,
                ).view(B, -1, L)
                out_y.append(yi)
            out_y = torch.stack(out_y, dim=1)
        else:
            out_y = selective_scan(
                xs, dts, As, Bs, Cs, Ds,
                delta_bias=dt_projs_bias,
                delta_softplus=True,
            ).view(B, K, -1, L)
        assert out_y.dtype == torch.float

        if self.scan_type == 'Interval':
            x_mamba1 = rearrange(out_y[:, 0].view(B, -1, W, H), 'b c h w -> b h w c')
            x_mamba2 = rearrange(torch.transpose(out_y[:, 1].view(B, -1, W, H), dim0=2, dim1=3), 'b c h w -> b h w c')
            inv_y = torch.flip(out_y[:, 2:4], dims=[-1]).view(B, 2, -1, L)
            x_mamba3 = rearrange(inv_y[:, 0].view(B, -1, W, H), 'b c h w -> b h w c')
            x_mamba4 = rearrange(torch.transpose(inv_y[:, 1].view(B, -1, W, H), dim0=2, dim1=3), 'b c h w -> b h w c')
            y = self.Interval_Combine([x_mamba1, x_mamba2, x_mamba3, x_mamba4])
            y = y * fc_out_2.unsqueeze(1).unsqueeze(1)
            y = self.out_norm(y)

        y = y * z
        out = self.dropout(self.out_proj(y))
        return out


# ──────────────────────────────────────────────
# Block_SpeGroup — Spectral group scanning
# ──────────────────────────────────────────────
class Block_SpeGroup(nn.Module, mamba_init):
    def __init__(self,
                 scan_type=None,
                 k_group=None,
                 dim=None,
                 dt_rank="auto",
                 d_state=None,
                 d_model=None,
                 d_model_spe=None,
                 ssm_ratio=None,
                 bimamba=None,
                 seq=False,
                 force_fp32=True,
                 dropout=0.0,
                 **kwargs):
        super().__init__()
        act_layer = nn.SiLU
        dt_min = 0.001
        dt_max = 0.1
        dt_init = "random"
        dt_scale = 1.0
        dt_init_floor = 1e-4
        bias = False
        self.force_fp32 = force_fp32
        self.seq = seq
        self.k_group = k_group
        self.scan_type = scan_type
        d_inner = int(ssm_ratio * d_model)

        self.fc1 = nn.Linear(dim, 4, bias=True)
        self.fc2 = nn.Linear(4, dim, bias=True)
        self.relu = nn.ReLU()
        self.sigmoid = nn.Sigmoid()

        # in proj
        self.in_proj = nn.Linear(dim, d_inner * 2, bias=bias)
        self.act = act_layer()
        self.forward_conv1d = nn.Conv1d(in_channels=d_inner, out_channels=d_inner, kernel_size=1)
        self.conv2d = nn.Conv2d(in_channels=d_inner, out_channels=d_inner, groups=d_inner, bias=True, kernel_size=(1, 1))
        self.conv3d = nn.Conv3d(in_channels=d_inner, out_channels=d_inner, groups=d_inner, bias=True, kernel_size=(1, 1, 1))

        # out proj
        self.out_norm = nn.LayerNorm(d_inner)
        self.out_proj = nn.Linear(d_inner, dim, bias=bias)
        self.dropout = nn.Dropout(dropout) if dropout > 0. else nn.Identity()

        # x proj
        d_inner = int(ssm_ratio * d_model_spe)
        dt_rank = math.ceil(d_model_spe / 16) if dt_rank == "auto" else dt_rank
        self.x_proj = [nn.Linear(d_inner, (dt_rank + d_state * 2), bias=False) for _ in range(k_group)]
        self.x_proj_weight = nn.Parameter(torch.stack([t.weight for t in self.x_proj], dim=0))
        del self.x_proj

        # dt proj
        self.dt_projs = [self.dt_init(dt_rank, d_inner, dt_scale, dt_init, dt_min, dt_max, dt_init_floor) for _ in range(k_group)]
        self.dt_projs_weight = nn.Parameter(torch.stack([t.weight for t in self.dt_projs], dim=0))
        self.dt_projs_bias = nn.Parameter(torch.stack([t.bias for t in self.dt_projs], dim=0))
        del self.dt_projs

        # A, D
        self.A_logs = self.A_log_init(d_state, d_inner, copies=k_group, merge=True)
        self.Ds = self.D_init(d_inner, copies=k_group, merge=True)

    def scan(self, x, scan_type=None, group_type=None, route=None):
        if scan_type == 'Interval':
            x1 = x[:, 0::4, :, :].permute(0, 2, 1, 3).contiguous()
            x2 = x[:, 1::4, :, :].permute(0, 2, 1, 3).contiguous()
            x3 = x[:, 2::4, :, :].permute(0, 2, 1, 3).contiguous()
            x4 = x[:, 3::4, :, :].permute(0, 2, 1, 3).contiguous()
            xs1 = x1.view(self.B, -1, self.L).view(self.B, 1, -1, self.L)
            xs2 = torch.transpose(x2, dim0=2, dim1=3).contiguous().view(self.B, -1, self.L).view(self.B, 1, -1, self.L)
            xs3 = x3.view(self.B, -1, self.L).view(self.B, 1, -1, self.L)
            xs3 = torch.flip(xs3, dims=[-1])
            xs4 = torch.transpose(x4, dim0=2, dim1=3).contiguous().view(self.B, -1, self.L).view(self.B, 1, -1, self.L)
            xs4 = torch.flip(xs4, dims=[-1])
            xs = torch.stack([xs1, xs2, xs3, xs4], dim=1).view(self.B, 4, -1, self.L)
        return xs

    def Interval_Combine(self, vectors):
        num = len(vectors)
        B, H, W, L = vectors[0].shape
        device = vectors[0].device
        merged_vector = torch.zeros(B, H, W, num * L, device=device)
        for j in range(L):
            for i in range(num):
                merged_vector[:, :, :, j * num + i] = vectors[i][:, :, :, j]
        return merged_vector

    def forward(self, x: Tensor, group_type=None, route=None):
        x = self.in_proj(x)
        x, z = x.chunk(2, dim=-1)
        z = self.act(z)

        if group_type == 'Patch':
            x = x.permute(0, 3, 1, 2).contiguous()
            x = self.conv2d(x)
            x = self.act(x)

        def selective_scan(u, delta, A, B, C, D=None, delta_bias=None, delta_softplus=True, nrows=1):
            return SelectiveScanMamba.apply(u, delta, A, B, C, D, delta_bias, delta_softplus, nrows, False)

        zz = x.mean(dim=2).mean(dim=2)
        fc_out_1 = self.relu(self.fc1(zz))
        fc_out_2 = self.sigmoid(self.fc2(fc_out_1))

        B, D, H, W = x.shape
        D_A, N = self.A_logs.shape
        K, D_dt, R = self.dt_projs_weight.shape
        L = x.size(1) // 4 * x.size(2)
        self.B = B
        self.L = L

        xs = self.scan(x, scan_type=self.scan_type, group_type=group_type, route=route)

        x_dbl = torch.einsum("b k d l, k c d -> b k c l", xs, self.x_proj_weight)
        dts, Bs, Cs = torch.split(x_dbl, [R, N, N], dim=2)
        dts = torch.einsum("b k r l, k d r -> b k d l", dts, self.dt_projs_weight)

        xs = xs.view(B, -1, L)
        dts = dts.contiguous().view(B, -1, L)
        Bs = Bs.contiguous()
        Cs = Cs.contiguous()

        As = -torch.exp(self.A_logs.float())
        Ds = self.Ds.float()
        dt_projs_bias = self.dt_projs_bias.float().view(-1)

        to_fp32 = lambda *args: (_a.to(torch.float32) for _a in args)

        if self.force_fp32:
            xs, dts, Bs, Cs = to_fp32(xs, dts, Bs, Cs)

        if self.seq:
            out_y = []
            for i in range(self.k_group):
                yi = selective_scan(
                    xs.view(B, K, -1, L)[:, i], dts.view(B, K, -1, L)[:, i],
                    As.view(K, -1, N)[i], Bs[:, i].unsqueeze(1), Cs[:, i].unsqueeze(1), Ds.view(K, -1)[i],
                    delta_bias=dt_projs_bias.view(K, -1)[i],
                    delta_softplus=True,
                ).view(B, -1, L)
                out_y.append(yi)
            out_y = torch.stack(out_y, dim=1)
        else:
            out_y = selective_scan(
                xs, dts, As, Bs, Cs, Ds,
                delta_bias=dt_projs_bias,
                delta_softplus=True,
            ).view(B, K, -1, L)
        assert out_y.dtype == torch.float

        if self.scan_type == 'Interval':
            x_mamba1 = out_y[:, 0]
            x_mamba1 = x_mamba1.transpose(dim0=1, dim1=2).contiguous().view(B, W, -1, H).permute(0, 3, 1, 2)
            x_mamba2 = torch.transpose(out_y[:, 1].view(B, -1, W, H), dim0=2, dim1=3).contiguous().view(B, -1, L)
            x_mamba2 = x_mamba2.transpose(dim0=1, dim1=2).contiguous().view(B, W, -1, H).permute(0, 3, 1, 2)
            inv_y = torch.flip(out_y[:, 2:4], dims=[-1]).view(B, 2, -1, L)
            x_mamba3 = inv_y[:, 0]
            x_mamba3 = x_mamba3.transpose(dim0=1, dim1=2).contiguous().view(B, W, -1, H).permute(0, 3, 1, 2)
            x_mamba4 = torch.transpose(inv_y[:, 1].view(B, -1, W, H), dim0=2, dim1=3).contiguous().view(B, -1, L)
            x_mamba4 = x_mamba4.transpose(dim0=1, dim1=2).contiguous().view(B, W, -1, H).permute(0, 3, 1, 2)
            y = self.Interval_Combine([x_mamba1, x_mamba2, x_mamba3, x_mamba4])
            y = y * fc_out_2.unsqueeze(1).unsqueeze(1)
            y = self.out_norm(y)

        y = y * z
        out = self.dropout(self.out_proj(y))
        return out


# ──────────────────────────────────────────────
# MLP Block
# ──────────────────────────────────────────────
class MLP_Block(nn.Module):
    def __init__(self, in_features, hidden_features, dropout=0.1):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(in_features, hidden_features),
            nn.GELU(),
            nn.Dropout(dropout) if dropout > 0. else nn.Identity(),
            nn.Linear(hidden_features, in_features)
        )

    def forward(self, x):
        return self.mlp(x)
