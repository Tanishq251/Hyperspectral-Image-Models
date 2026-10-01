"""
MambaMoE: Mixture-of-Experts Mamba Network for Hyperspectral Image Classification

Paper/Source: https://github.com/Candy-CY/Hyperspectral-Image-Classification-Models/2025/MambaMoE
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange, repeat
from einops.layers.torch import Rearrange
from typing import List, Callable
from functools import partial

from timm.models.layers import DropPath
from mamba_ssm import Mamba

try:
    from mamba_ssm.ops.selective_scan_interface import selective_scan_fn
except ImportError:
    selective_scan_fn = None

from models.registry import register_model


def to_2tuple(x):
    if isinstance(x, (list, tuple)):
        return x
    return (x, x)


class PatchEmbed(nn.Module):
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
        x = x.flatten(2).transpose(1, 2)
        if self.norm is not None:
            x = self.norm(x)
        return x, (Hp, Wp)


class FeatureFusionBlock(nn.Module):
    def __init__(self, features):
        super(FeatureFusionBlock, self).__init__()
        self.resConfUnit1 = ResidualConvUnit(features)
        self.resConfUnit2 = ResidualConvUnit(features)

    def forward(self, *xs):
        output = xs[0]
        if len(xs) == 2:
            output = F.interpolate(output, size=xs[1].size()[2:], mode="bilinear", align_corners=True)
            output += self.resConfUnit1(xs[1])
        output = self.resConfUnit2(output)
        return output


class ResidualConvUnit(nn.Module):
    def __init__(self, features):
        super(ResidualConvUnit, self).__init__()
        self.conv1 = nn.Conv2d(features, features, kernel_size=3, stride=1, padding=1, bias=True)
        self.conv2 = nn.Conv2d(features, features, kernel_size=3, stride=1, padding=1, bias=True)
        self.relu = nn.ReLU()

    def forward(self, x):
        out = self.relu(x)
        out = self.conv1(out)
        out = self.relu(out)
        out = self.conv2(out)
        return out + x


class Uncertainty_Guide_Enhancement(nn.Module):
    def __init__(self):
        super(Uncertainty_Guide_Enhancement, self).__init__()

    def forward(self, features, guide_map):
        guide_map = F.interpolate(guide_map, size=features.size()[2:], mode="bilinear", align_corners=True)
        prob_map = torch.softmax(guide_map, dim=1)
        max_map, _ = torch.max(prob_map, dim=1)

        uncertainty = -1 * torch.log(max_map + 1e-6) * max_map
        uncertainty_normal = (uncertainty - uncertainty.min()) / (
                    uncertainty.max() - uncertainty.min() + 1e-6)

        enhanced_features = uncertainty_normal * features + features
        return enhanced_features, prob_map, max_map, uncertainty


def uncertainty_map(input: torch.Tensor) -> torch.Tensor:
    input = F.softmax(input, dim=1)
    max_prob, _ = torch.max(input, dim=1)
    un = -1 * torch.log(max_prob + 1e-6) * max_prob
    un = un.unsqueeze(1)
    max_prob = max_prob.unsqueeze(1)
    return un, max_prob, input


def UARB(uncertainty_map: torch.Tensor, gt: torch.Tensor, ignore_index: int = 255) -> torch.Tensor:
    B, _, H, W = uncertainty_map.shape

    std_map_reshaped = uncertainty_map.view(B, -1)
    min_val = std_map_reshaped.min(dim=1, keepdim=True)[0]
    max_val = std_map_reshaped.max(dim=1, keepdim=True)[0]
    norm_std_map = (std_map_reshaped - min_val) / (max_val - min_val + 1e-6)
    prob_map = norm_std_map.view(B, 1, H, W)

    random_sample = torch.rand_like(prob_map)
    mask_map = (random_sample < prob_map).float()

    gt = gt.unsqueeze(1).float()
    masked_gt = gt * mask_map
    masked_gt = masked_gt.squeeze(1).long()
    final_gt = masked_gt.masked_fill(mask_map.squeeze(1) == 0, ignore_index)

    if (final_gt != ignore_index).sum() == 0:
        final_gt = gt.squeeze(1).long()

    mask_map = mask_map.squeeze()
    return final_gt, mask_map


class LayerNorm(nn.Module):
    def __init__(self, normalized_shape, eps=1e-6, data_format="channels_last"):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(normalized_shape))
        self.bias = nn.Parameter(torch.zeros(normalized_shape))
        self.eps = eps
        self.data_format = data_format
        if self.data_format not in ["channels_last", "channels_first"]:
            raise NotImplementedError
        self.normalized_shape = (normalized_shape,)

    def forward(self, x):
        if self.data_format == "channels_last":
            return F.layer_norm(x, self.normalized_shape, self.weight, self.bias, self.eps)
        elif self.data_format == "channels_first":
            u = x.mean(1, keepdim=True)
            s = (x - u).pow(2).mean(1, keepdim=True)
            x = (x - u) / torch.sqrt(s + self.eps)
            x = self.weight[:, None, None] * x + self.bias[:, None, None]
            return x


class BSS(nn.Module):
    def __init__(
            self,
            d_model,
            d_state=16,
            d_conv=3,
            expand=2,
            dt_rank="auto",
            dt_min=0.001,
            dt_max=0.1,
            dt_init="random",
            dt_scale=1.0,
            dt_init_floor=1e-4,
            dropout=0.,
            d_spectral=121,
            conv_bias=True,
            bias=False,
            device=None,
            dtype=None,
            **kwargs,
    ):
        factory_kwargs = {"device": device, "dtype": dtype}
        super().__init__()
        self.d_model = d_model
        self.d_state = d_state
        self.d_conv = d_conv
        self.expand = expand
        self.d_inner = int(self.expand * self.d_model)
        self.d_spectral = d_spectral
        self.dt_rank = math.ceil(self.d_model / 16) if dt_rank == "auto" else dt_rank

        self.in_proj = nn.Linear(self.d_model, self.d_inner * 2, bias=bias, **factory_kwargs)

        self.conv2d = nn.Conv2d(
            in_channels=self.d_inner,
            out_channels=self.d_inner,
            groups=self.d_inner,
            bias=conv_bias,
            kernel_size=d_conv,
            padding=(d_conv - 1) // 2,
            **factory_kwargs,
        )
        self.act = nn.SiLU()

        self.x_proj_spectral = (
            nn.Linear(self.d_spectral, (self.dt_rank + self.d_state * 2), bias=False, **factory_kwargs),
            nn.Linear(self.d_spectral, (self.dt_rank + self.d_state * 2), bias=False, **factory_kwargs),
        )
        self.x_proj_weight_spectral = nn.Parameter(torch.stack([t.weight for t in self.x_proj_spectral], dim=0))
        del self.x_proj_spectral

        self.dt_projs_spectral = (
            self.dt_init(self.dt_rank, self.d_spectral, dt_scale, dt_init, dt_min, dt_max, dt_init_floor,
                         **factory_kwargs),
            self.dt_init(self.dt_rank, self.d_spectral, dt_scale, dt_init, dt_min, dt_max, dt_init_floor,
                         **factory_kwargs),
        )
        self.dt_projs_weight_spectral = nn.Parameter(torch.stack([t.weight for t in self.dt_projs_spectral], dim=0))
        self.dt_projs_bias_spectral = nn.Parameter(torch.stack([t.bias for t in self.dt_projs_spectral], dim=0))
        del self.dt_projs_spectral

        self.A_logs_spectral = self.A_log_init(self.d_state, self.d_spectral, copies=2, merge=True)
        self.Ds_spectral = self.D_init(self.d_spectral, copies=2, merge=True)

        self.selective_scan = selective_scan_fn

        self.out_norm_spectral = nn.LayerNorm(self.d_spectral)
        self.out_proj = nn.Linear(self.d_inner, self.d_model, bias=bias, **factory_kwargs)
        self.dropout = nn.Dropout(dropout) if dropout > 0. else None

        self.conv = nn.Conv1d(
            in_channels=self.d_inner,
            out_channels=self.d_inner,
            groups=self.d_inner if self.d_inner == 64 else 1,
            bias=conv_bias,
            kernel_size=d_conv,
            padding=(d_conv - 1) // 2,
            **factory_kwargs,
        )

    @staticmethod
    def dt_init(dt_rank, d_inner, dt_scale=1.0, dt_init="random", dt_min=0.001, dt_max=0.1, dt_init_floor=1e-4,
                **factory_kwargs):
        dt_proj = nn.Linear(dt_rank, d_inner, bias=True, **factory_kwargs)
        dt_init_std = dt_rank ** -0.5 * dt_scale
        if dt_init == "constant":
            nn.init.constant_(dt_proj.weight, dt_init_std)
        elif dt_init == "random":
            nn.init.uniform_(dt_proj.weight, -dt_init_std, dt_init_std)
        else:
            raise NotImplementedError
        dt = torch.exp(
            torch.rand(d_inner, **factory_kwargs) * (math.log(dt_max) - math.log(dt_min))
            + math.log(dt_min)
        ).clamp(min=dt_init_floor)
        inv_dt = dt + torch.log(-torch.expm1(-dt))
        with torch.no_grad():
            dt_proj.bias.copy_(inv_dt)
        dt_proj.bias._no_reinit = True
        return dt_proj

    @staticmethod
    def A_log_init(d_state, d_inner, copies=1, device=None, merge=True):
        A = repeat(
            torch.arange(1, d_state + 1, dtype=torch.float32, device=device),
            "n -> d n",
            d=d_inner,
        ).contiguous()
        A_log = torch.log(A)
        if copies > 1:
            A_log = repeat(A_log, "d n -> r d n", r=copies)
            if merge:
                A_log = A_log.flatten(0, 1)
        A_log = nn.Parameter(A_log)
        A_log._no_weight_decay = True
        return A_log

    @staticmethod
    def D_init(d_inner, copies=1, device=None, merge=True):
        D = torch.ones(d_inner, device=device)
        if copies > 1:
            D = repeat(D, "n1 -> r n1", r=copies)
            if merge:
                D = D.flatten(0, 1)
        D = nn.Parameter(D)
        D._no_weight_decay = True
        return D

    def forward_spectral(self, x: torch.Tensor):
        old_B, old_C, old_H, old_W = x.shape
        x = torch.transpose(x.contiguous().view(old_B, old_C, -1), 1, 2).view(old_B, old_H * old_W, old_C, 1)
        B, C, H, W = x.shape
        L = H * W
        K = 2
        xs = torch.cat([x, torch.flip(x, dims=[-1])], dim=1)
        x_dbl = torch.einsum("b k d l, k c d -> b k c l", xs.view(B, K, -1, L), self.x_proj_weight_spectral)
        dts, Bs, Cs = torch.split(x_dbl, [self.dt_rank, self.d_state, self.d_state], dim=2)
        dts = torch.einsum("b k r l, k d r -> b k d l", dts.view(B, K, -1, L), self.dt_projs_weight_spectral)
        xs = xs.view(B, -1, L)
        dts = dts.contiguous().view(B, -1, L)
        Bs = Bs.view(B, K, -1, L)
        Cs = Cs.view(B, K, -1, L)
        As = -torch.exp(self.A_logs_spectral.float()).view(-1, self.d_state)
        Ds = self.Ds_spectral.view(-1)
        dt_projs_bias = self.dt_projs_bias_spectral.view(-1)

        if self.selective_scan is not None:
            out_y = self.selective_scan(
                xs, dts,
                As, Bs, Cs, Ds,
                delta_bias=dt_projs_bias,
                delta_softplus=True,
            ).view(B, K, -1, L)
        else:
            out_y = xs.view(B, K, -1, L)

        inv_y = torch.flip(out_y, dims=[-1])
        y = out_y[:, 0].float() + inv_y[:, 1].float()

        y = torch.transpose(y, dim0=1, dim1=2).contiguous().view(B, H, W, -1)
        y = self.out_norm_spectral(y).to(x.dtype)
        y = torch.transpose(y.view(B, L, -1), 1, 2).view(old_B, old_H, old_W, old_C)
        return y

    def forward(self, x: torch.Tensor, **kwargs):
        B, H, W, C = x.shape
        xz = self.in_proj(x)
        x, z = xz.chunk(2, dim=-1)
        dim = x.size(-1)
        x = x.permute(0, 3, 1, 2).contiguous().view(B, dim, -1)
        x = self.act(self.conv(x))
        x = x.view(B, self.d_inner, H, W)

        y_spectral = self.forward_spectral(x)
        y = y_spectral * F.gelu(z)
        out = self.out_proj(y)
        if self.dropout is not None:
            out = self.dropout(out)
        return out


class SS2D(nn.Module):
    def __init__(
            self,
            d_model,
            d_state=16,
            d_conv=3,
            expand=2,
            dt_rank="auto",
            dt_min=0.001,
            dt_max=0.1,
            dt_init="random",
            dt_scale=1.0,
            dt_init_floor=1e-4,
            dropout=0.,
            conv_bias=True,
            bias=False,
            device=None,
            dtype=None,
            low_dim=16,
            **kwargs,
    ):
        factory_kwargs = {"device": device, "dtype": dtype}
        super().__init__()
        self.d_model = d_model
        self.d_state = d_state
        self.d_conv = d_conv
        self.expand = expand
        self.d_inner = low_dim
        self.dt_rank = math.ceil(self.d_model / 16) if dt_rank == "auto" else dt_rank

        self.in_proj = nn.Linear(self.d_model, self.d_inner * 2, bias=bias, **factory_kwargs)
        self.conv2d = nn.Conv2d(
            in_channels=self.d_inner,
            out_channels=self.d_inner,
            groups=self.d_inner,
            bias=conv_bias,
            kernel_size=d_conv,
            padding=(d_conv - 1) // 2,
            **factory_kwargs,
        )
        self.act = nn.SiLU()

        self.x_proj = nn.Linear(self.d_inner, (self.dt_rank + self.d_state * 2), bias=False, **factory_kwargs)
        self.dt_proj = self.dt_init(self.dt_rank, self.d_inner, dt_scale, dt_init, dt_min, dt_max, dt_init_floor, **factory_kwargs)

        self.A_logs = self.A_log_init(self.d_state, self.d_inner)
        self.Ds = self.D_init(self.d_inner)

        self.selective_scan = selective_scan_fn
        self.out_norm = nn.LayerNorm(self.d_inner)
        self.out_proj = nn.Linear(self.d_inner, self.d_model, bias=bias, **factory_kwargs)
        self.dropout = nn.Dropout(dropout) if dropout > 0. else None

    @staticmethod
    def dt_init(dt_rank, d_inner, dt_scale=1.0, dt_init="random", dt_min=0.001, dt_max=0.1, dt_init_floor=1e-4,
                **factory_kwargs):
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
        dt_proj.bias._no_reinit = True
        return dt_proj

    @staticmethod
    def A_log_init(d_state, d_inner, device=None):
        A = repeat(
            torch.arange(1, d_state + 1, dtype=torch.float32, device=device),
            "n -> d n",
            d=d_inner,
        ).contiguous()
        A_log = torch.log(A)
        A_log = nn.Parameter(A_log)
        A_log._no_weight_decay = True
        return A_log

    @staticmethod
    def D_init(d_inner, device=None):
        D = torch.ones(d_inner, device=device)
        D = nn.Parameter(D)
        D._no_weight_decay = True
        return D

    def forward_core(self, x: torch.Tensor):
        B, C, H, W = x.shape
        L = H * W
        x_seq = x.view(B, -1, L)
        x_dbl = torch.einsum("b d l, c d -> b c l", x_seq, self.x_proj.weight)
        dts, Bs, Cs = torch.split(x_dbl, [self.dt_rank, self.d_state, self.d_state], dim=1)
        dts = torch.einsum("b r l, d r -> b d l", dts, self.dt_proj.weight)

        xs = x_seq.float()
        dts = dts.contiguous().float()
        Bs = Bs.float()
        Cs = Cs.float()
        Ds = self.Ds.float().view(-1)
        As = -torch.exp(self.A_logs.float()).view(-1, self.d_state)
        dt_proj_bias = self.dt_proj.bias.float().view(-1)

        if self.selective_scan is not None:
            out_y = self.selective_scan(
                xs, dts,
                As, Bs, Cs, Ds, z=None,
                delta_bias=dt_proj_bias,
                delta_softplus=True,
                return_last_state=False,
            ).view(B, -1, L)
        else:
            out_y = xs

        return out_y

    def forward(self, x: torch.Tensor, **kwargs):
        B, H, W, C = x.shape
        xz = self.in_proj(x)
        x, z = xz.chunk(2, dim=-1)

        x = x.permute(0, 3, 1, 2).contiguous()
        x = self.act(self.conv2d(x))
        y = self.forward_core(x)
        y = torch.transpose(y, dim0=1, dim1=2).contiguous().view(B, H, W, -1)
        y = self.out_norm(y)
        y = y * F.silu(z)
        out = self.out_proj(y)
        if self.dropout is not None:
            out = self.dropout(out)
        return out


class FFN(nn.Module):
    def __init__(self, dim, bias, kernel_size):
        super(FFN, self).__init__()
        if kernel_size not in [3, 5, 7]:
            raise ValueError("Invalid kernel_size. Must be 3, 5, or 7.")

        self.kernel_size = kernel_size
        hidden_features = 180

        self.project_in = nn.Conv2d(dim, hidden_features, kernel_size=1, bias=bias)
        self.dwconv3x3 = nn.Conv2d(hidden_features, hidden_features, kernel_size=3, stride=1, padding=1, groups=hidden_features, bias=bias)
        self.dwconv5x5 = nn.Conv2d(hidden_features, hidden_features, kernel_size=5, stride=1, padding=2, groups=hidden_features, bias=bias)
        self.dwconv7x7 = nn.Conv2d(hidden_features, hidden_features, kernel_size=7, stride=1, padding=3, groups=hidden_features, bias=bias)
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


class MMoEB(nn.Module):
    def __init__(
            self,
            hidden_dim: int = 0,
            drop_path: float = 0,
            norm_layer: Callable[..., torch.nn.Module] = partial(nn.LayerNorm, eps=1e-6),
            topk: int = 1,
            attn_drop_rate: float = 0,
            d_state: int = 16,
            expand: float = 2.,
            **kwargs,
    ):
        super().__init__()
        self.ln_1 = norm_layer(hidden_dim)
        self.drop_path = DropPath(drop_path)
        self.skip_scale = nn.Parameter(torch.ones(hidden_dim))
        self.moe = MoEFFN(hidden_dim, num_experts=4, topk=2)
        self.ln_2 = nn.LayerNorm(hidden_dim)
        self.skip_scale2 = nn.Parameter(torch.ones(hidden_dim))
        self.ffn = Mlp(in_features=hidden_dim, hidden_features=256, out_features=hidden_dim, act_layer=nn.GELU)

    def forward(self, input):
        x = input.permute(0, 2, 3, 1).contiguous()
        x = x * self.skip_scale + self.moe(self.ln_2(x).permute(0, 3, 1, 2).contiguous()).permute(0, 2, 3, 1).contiguous()
        x = self.ln_1(x)
        x = x * self.skip_scale2 + self.ffn(x)
        x = x.permute(0, 3, 1, 2).contiguous()
        return x


class MoEFFN(nn.Module):
    def __init__(self,
                 in_ch: int,
                 num_experts: int,
                 topk: int,
                 lr_space: str = "exp",
                 use_shuffle: bool = False
                 ):
        super().__init__()
        self.norm_1 = LayerNorm(in_ch, data_format='channels_first')
        self.block = MoEBlock(in_ch=in_ch, num_experts=num_experts, topk=topk, use_shuffle=use_shuffle, lr_space=lr_space)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        res = x
        x = self.block(self.norm_1(x)) + res
        return x


class MoEBlock(nn.Module):
    def __init__(self,
                 in_ch: int,
                 num_experts: int,
                 topk: int,
                 use_shuffle: bool = False,
                 lr_space: str = "exp"
                 ):
        super().__init__()
        self.use_shuffle = use_shuffle
        self.conv_1 = nn.Conv2d(in_ch, in_ch * 2, kernel_size=1, padding=0)

        if lr_space == "linear":
            grow_func = lambda i: i + 2
        elif lr_space == "exp":
            grow_func = lambda i: 2 ** (i + 1)
        elif lr_space == "double":
            grow_func = lambda i: 2 * i + 2
        else:
            raise NotImplementedError(f"lr_space {lr_space} not implemented")

        self.moe_layer = MoELayer(
            experts=[SS2D(d_model=in_ch, low_dim=grow_func(i)) for i in range(num_experts)],
            gate=Router(in_ch=in_ch, num_experts=num_experts),
            num_expert=topk,
        )

        self.proj = nn.Conv2d(in_ch * 2, in_ch, kernel_size=1, padding=0)
        self.norm = LayerNorm(in_ch, data_format='channels_first')
        self.norms = nn.ModuleList([
            nn.LayerNorm(in_ch),
            nn.LayerNorm(in_ch),
        ])
        self.shared = None

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, C, H, W = x.shape
        if self.shared is None or self.shared.d_spectral != H * W:
            self.shared = BSS(d_model=C, d_spectral=H * W).to(x.device)

        x = self.conv_1(x)
        x_s, x_a = torch.chunk(x, chunks=2, dim=1)

        x_a = self.moe_layer(x_a)

        x_s = x_s.permute(0, 2, 3, 1)
        x_s = self.shared(x_s)
        x_s = x_s.permute(0, 3, 1, 2)

        x = torch.cat([x_s, x_a], 1)
        x = self.proj(x)
        return x


class MoELayer(nn.Module):
    def __init__(self, experts: List[nn.Module], gate: nn.Module, num_expert: int = 1):
        super().__init__()
        assert len(experts) > 0
        self.experts = nn.ModuleList(experts)
        self.gate = gate
        self.num_expert = num_expert

    def _generate_directions(self, x: torch.Tensor):
        B, C, H, W = x.shape
        L = H * W
        x_hwwh = torch.stack([
            x.view(B, -1, L),
            torch.transpose(x, dim0=2, dim1=3).contiguous().view(B, -1, L)
        ], dim=1).view(B, 2, -1, L)
        xs = torch.cat([x_hwwh, torch.flip(x_hwwh, dims=[-1])], dim=1)
        return xs

    def forward(self, inputs: torch.Tensor):
        B, C, H, W = inputs.shape
        out = self.gate(inputs)
        weights = F.softmax(out, dim=1, dtype=torch.float).to(inputs.dtype)
        topk_weights, topk_experts = torch.topk(weights, self.num_expert)
        result = inputs.clone()

        directional_inputs = self._generate_directions(inputs)

        if self.training:
            exp_weights = torch.zeros_like(weights)
            exp_weights.scatter_(1, topk_experts, weights.gather(1, topk_experts))
            for i, expert in enumerate(self.experts):
                dir_idx = i % 4
                x_dir = directional_inputs[:, dir_idx].view(B, C, H, W).permute(0, 2, 3, 1)
                result += expert(x_dir).permute(0, 3, 1, 2) * exp_weights[:, i:i+1, None, None]
        else:
            for i in range(self.num_expert):
                exp_idx = topk_experts[:, i]
                w = topk_weights[:, i]
                for batch_idx in range(B):
                    idx = exp_idx[batch_idx].item()
                    expert = self.experts[idx]
                    dir_idx = idx % 4
                    x_dir = directional_inputs[batch_idx, dir_idx].view(1, C, H, W).permute(0, 2, 3, 1)
                    res_expert = expert(x_dir).permute(0, 3, 1, 2)
                    result[batch_idx] += res_expert[0] * w[batch_idx]
        return result


class Router(nn.Module):
    def __init__(self, in_ch: int, num_experts: int):
        super().__init__()
        self.body = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            Rearrange('b c 1 1 -> b c'),
            nn.Linear(in_ch, num_experts, bias=False),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.body(x)


class MambaMoE(nn.Module):
    def __init__(self,
                 num_features=103,
                 embed_dim=64,
                 img_size=128,
                 patch_size=1,
                 norm_layer=nn.LayerNorm,
                 drop_rate=0.,
                 num_classes=9,
                 group_num=4,
                 patch_norm=True
                 ):
        super(MambaMoE, self).__init__()

        self.patch_norm = patch_norm
        self.bn = nn.BatchNorm2d(embed_dim)
        self.relu = nn.ReLU(inplace=True)
        self.patch_embed = PatchEmbed(
            img_size=img_size,
            patch_size=patch_size,
            in_chans=embed_dim,
            embed_dim=embed_dim,
            norm_layer=norm_layer if self.patch_norm else None)
        self.pos_drop = nn.Dropout(p=drop_rate)

        self.patch_embedding = nn.Sequential(
            nn.Conv2d(in_channels=num_features, out_channels=embed_dim, kernel_size=1, stride=1, padding=0),
            nn.GroupNorm(group_num, embed_dim),
            nn.ReLU(inplace=True))

        self.feature_1 = nn.Sequential(
            nn.Conv2d(in_channels=num_features, out_channels=embed_dim, kernel_size=1, stride=1, padding=0),
            nn.GroupNorm(group_num, embed_dim),
            nn.ReLU(inplace=True),
            nn.AvgPool2d(kernel_size=2, stride=2, padding=0)
        )

        self.feature_2 = nn.Sequential(
            nn.Conv2d(in_channels=embed_dim, out_channels=embed_dim, kernel_size=1, stride=1, padding=0),
            nn.GroupNorm(group_num, embed_dim),
            nn.ReLU(inplace=True),
            nn.AvgPool2d(kernel_size=2, stride=2, padding=0)
        )

        self.feature_3 = nn.Sequential(
            nn.Conv2d(in_channels=embed_dim, out_channels=embed_dim, kernel_size=1, stride=1, padding=0),
            nn.GroupNorm(group_num, embed_dim),
            nn.ReLU(inplace=True),
            nn.AvgPool2d(kernel_size=2, stride=2, padding=0)
        )

        self.block_1 = MMoEB(embed_dim, num_experts=4, d_state=16)
        self.conv_1 = nn.AvgPool2d(kernel_size=2, stride=2, padding=0)

        self.block_2 = MMoEB(embed_dim, num_experts=4, d_state=16)
        self.conv_2 = nn.AvgPool2d(kernel_size=2, stride=2, padding=0)

        self.block_3 = MMoEB(embed_dim, num_experts=4, d_state=16)

        self.path_blocks = nn.ModuleList([
            FeatureFusionBlock(embed_dim) for _ in range(3)
        ])

        self.cls_head_1 = nn.Sequential(
            nn.Conv2d(in_channels=embed_dim, out_channels=128, kernel_size=1, stride=1, padding=0),
            nn.GroupNorm(group_num, 128),
            nn.SiLU(),
            nn.Conv2d(in_channels=128, out_channels=num_classes, kernel_size=1, stride=1, padding=0)
        )

        self.cls_head_2 = nn.Sequential(
            nn.Conv2d(in_channels=embed_dim, out_channels=128, kernel_size=1, stride=1, padding=0),
            nn.GroupNorm(group_num, 128),
            nn.SiLU(),
            nn.Conv2d(in_channels=128, out_channels=num_classes, kernel_size=1, stride=1, padding=0)
        )

        self.cls_head_3 = nn.Sequential(
            nn.Conv2d(in_channels=embed_dim, out_channels=128, kernel_size=1, stride=1, padding=0),
            nn.GroupNorm(group_num, 128),
            nn.SiLU(),
            nn.Conv2d(in_channels=128, out_channels=num_classes, kernel_size=1, stride=1, padding=0)
        )

        self.global_pool = nn.AdaptiveAvgPool2d(1)
        self.flat = nn.Flatten()

    def forward(self, x, gt=None):
        # Handle HSI classifier wrapping and standard pipeline format
        # Create a dummy ground truth map matching x's spatial dimension if not provided
        if gt is None:
            B, _, H, W = x.shape
            gt = torch.zeros((B, H, W), dtype=torch.long, device=x.device)

        interpolation = nn.UpsamplingBilinear2d(size=x.shape[2:4])

        layer1 = self.feature_1(x)
        layer2 = self.feature_2(layer1)
        layer3 = self.feature_3(layer2)

        layer1 = self.block_1(layer1)
        layer2 = self.block_2(layer2)
        layer3 = self.block_3(layer3)

        layer3 = self.path_blocks[2](layer3)
        out_3 = self.cls_head_3(layer3)

        un_3, pro_3, soft_3 = uncertainty_map(out_3)
        un_3 = interpolation(un_3)
        gt_3, mask_3 = UARB(un_3, gt)

        layer2 = self.path_blocks[1](layer3, layer2)
        out_2 = self.cls_head_2(layer2)

        un_2, pro_2, soft_2 = uncertainty_map(out_2)
        un_2 = interpolation(un_2)
        gt_2, mask_2 = UARB(un_2, gt)

        layer1 = self.path_blocks[0](layer2, layer1)

        x_out = interpolation(layer1)
        out_1 = self.cls_head_1(x_out)

        # Standard pipeline classification output wrapper:
        # pool the spatially-dense output map and flatten it to [B, num_classes]
        pooled_out = self.global_pool(out_1)
        flat_out = self.flat(pooled_out)
        return flat_out


@register_model('MambaMoE', expects_4d=True, embed_dim=64, group_num=4)
def mambamoe_model(pretrained: bool = False, **kwargs) -> MambaMoE:
    """Constructs a MambaMoE model."""
    if 'bands' in kwargs:
        kwargs['num_features'] = kwargs.pop('bands')
    kwargs.pop('patch_size', None)
    return MambaMoE(**kwargs)
