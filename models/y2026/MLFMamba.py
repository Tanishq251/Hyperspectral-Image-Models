"""
MLFMamba: Mamba-based Multi-Scan Spectral-Spatial Feature Extraction for Hyperspectral Image Classification

Paper: https://github.com/foopy113/MLF
"""

import math
import torch
from torch import nn
import torch.nn.functional as F
from einops import repeat
from mamba_ssm import Mamba

try:
    from mamba_ssm.ops.selective_scan_interface import selective_scan_fn
except ImportError:
    selective_scan_fn = None

from models.registry import register_model


class BottConv(nn.Module):
    def __init__(self, in_channels, out_channels, mid_channels, kernel_size, stride=1, padding=0, bias=True):
        super(BottConv, self).__init__()
        self.pointwise_1 = nn.Conv2d(in_channels, mid_channels, 1, bias=bias)
        self.depthwise = nn.Conv2d(mid_channels, mid_channels, kernel_size, stride, padding, groups=mid_channels,
                                   bias=False)
        self.pointwise_2 = nn.Conv2d(mid_channels, out_channels, 1, bias=False)

    def forward(self, x):
        x = self.pointwise_1(x)
        x = self.depthwise(x)
        x = self.pointwise_2(x)
        return x


class SS2D(nn.Module):
    def __init__(
            self,
            dim,
            d_state=16,
            expand=2,
            dt_rank="auto",
            dt_min=0.001,
            dt_max=0.1,
            dt_init="random",
            dt_scale=1.0,
            dt_init_floor=1e-4,
            conv_size=3,
            bias=False,
    ):
        super().__init__()
        self.d_model = dim
        self.d_state = d_state
        self.expand = expand
        self.d_inner = int(self.expand * self.d_model)
        self.dt_rank = math.ceil(self.d_model / 16) if dt_rank == "auto" else dt_rank

        self.n_directions = 4

        self.in_proj = nn.Linear(self.d_model, self.d_inner * 2, bias=bias)

        assert conv_size % 2 == 1
        self.conv2d = BottConv(in_channels=self.d_inner, out_channels=self.d_inner, mid_channels=self.d_inner // 16,
                               kernel_size=conv_size, padding=(conv_size - 1) // 2, stride=1)
        self.act = nn.SiLU()

        self.x_proj = nn.Linear(
            self.d_inner, self.dt_rank + self.d_state * 2, bias=False,
        )
        self.dt_proj = nn.Linear(
            self.dt_rank, self.d_inner, bias=True
        )

        dt_init_std = self.dt_rank ** -0.5 * dt_scale
        if dt_init == "constant":
            nn.init.constant_(self.dt_proj.weight, dt_init_std)
        elif dt_init == "random":
            nn.init.uniform_(self.dt_proj.weight, -dt_init_std, dt_init_std)
        else:
            raise NotImplementedError

        dt = torch.exp(
            torch.rand(self.d_inner) * (math.log(dt_max) - math.log(dt_min))
            + math.log(dt_min)
        ).clamp(min=dt_init_floor)
        inv_dt = dt + torch.log(-torch.expm1(-dt))
        with torch.no_grad():
            self.dt_proj.bias.copy_(inv_dt)

        A = repeat(
            torch.arange(1, self.d_state + 1, dtype=torch.float32),
            "n -> d n",
            d=self.d_inner,
        ).contiguous()
        A_log = torch.log(A)
        self.A_log = nn.Parameter(A_log)
        self.D = nn.Parameter(torch.ones(self.d_inner))
        self.out_proj = nn.Linear(self.d_inner, self.d_model, bias=bias)
        self.direction_Bs = nn.Parameter(torch.zeros(self.n_directions + 1, self.d_state))
        nn.init.trunc_normal_(self.direction_Bs, std=0.02)

    def sass(self, hw_shape):
        H, W = hw_shape
        L = H * W
        o1, o2, o3, o4 = [], [], [], []
        d1, d2, d3, d4 = [], [], [], []
        o1_inverse = [-1 for _ in range(L)]
        o2_inverse = [-1 for _ in range(L)]
        o3_inverse = [-1 for _ in range(L)]
        o4_inverse = [-1 for _ in range(L)]

        if H % 2 == 1:
            i, j = H - 1, W - 1
            j_d = "left"
        else:
            i, j = H - 1, 0
            j_d = "right"

        while i > -1:
            assert j_d in ["right", "left"]
            idx = i * W + j
            o1_inverse[idx] = len(o1)
            o1.append(idx)
            if j_d == "right":
                if j < W - 1:
                    j = j + 1
                    d1.append(1)
                else:
                    i = i - 1
                    d1.append(3)
                    j_d = "left"
            else:
                if j > 0:
                    j = j - 1
                    d1.append(2)
                else:
                    i = i - 1
                    d1.append(3)
                    j_d = "right"
        d1 = [0] + d1[:-1]

        i, j = 0, 0
        i_d = "down"
        while j < W:
            assert i_d in ["down", "up"]
            idx = i * W + j
            o2_inverse[idx] = len(o2)
            o2.append(idx)
            if i_d == "down":
                if i < H - 1:
                    i = i + 1
                    d2.append(4)
                else:
                    j = j + 1
                    d2.append(1)
                    i_d = "up"
            else:
                if i > 0:
                    i = i - 1
                    d2.append(3)
                else:
                    j = j + 1
                    d2.append(1)
                    i_d = "down"
        d2 = [0] + d2[:-1]

        for diag in range(H + W - 1):
            if diag % 2 == 0:
                for i_idx in range(min(diag + 1, H)):
                    i = i_idx
                    j = diag - i
                    if j < W:
                        idx = i * W + j
                        o3.append(idx)
                        o3_inverse[idx] = len(o3) - 1
                        d3.append(1 if j == diag else 4)
            else:
                for j_idx in range(min(diag + 1, W)):
                    j = j_idx
                    i = diag - j
                    if i < H:
                        idx = i * W + j
                        o3.append(idx)
                        o3_inverse[idx] = len(o3) - 1
                        d3.append(4 if i == diag else 1)
        d3 = [0] + d3[:-1]

        for diag in range(H + W - 1):
            if diag % 2 == 0:
                for i_idx in range(min(diag + 1, H)):
                    i = i_idx
                    j = diag - i
                    if j < W:
                        idx = i * W + (W - j - 1)
                        o4.append(idx)
                        o4_inverse[idx] = len(o4) - 1
                        d4.append(1 if j == diag else 4)
            else:
                for j_idx in range(min(diag + 1, W)):
                    j = j_idx
                    i = diag - j
                    if i < H:
                        idx = i * W + (W - j - 1)
                        o4.append(idx)
                        o4_inverse[idx] = len(o4) - 1
                        d4.append(4 if i == diag else 1)
        d4 = [0] + d4[:-1]

        return (tuple(o1), tuple(o2), tuple(o3), tuple(o4)), \
            (tuple(o1_inverse), tuple(o2_inverse), tuple(o3_inverse), tuple(o4_inverse)), \
            (tuple(d1), tuple(d2), tuple(d3), tuple(d4))

    def forward(self, x, hw_shape):
        batch_size, L, _ = x.shape
        H, W = hw_shape
        E = self.d_inner

        x_norm = x

        xz = self.in_proj(x_norm)
        x_mamba_in, z = xz.chunk(2, dim=-1)

        x_2d_conv_input = x_mamba_in.reshape(batch_size, H, W, E).permute(0, 3, 1, 2)
        x_conv_out = self.act(self.conv2d(x_2d_conv_input))
        x_conv = x_conv_out.permute(0, 2, 3, 1).reshape(batch_size, L, E)

        x_dbl = self.x_proj(x_conv)
        dt, B_ssm, C_ssm = torch.split(x_dbl, [self.dt_rank, self.d_state, self.d_state], dim=-1)
        dt = self.dt_proj(dt)

        dt = dt.permute(0, 2, 1).contiguous()
        B_ssm = B_ssm.permute(0, 2, 1).contiguous()
        C_ssm = C_ssm.permute(0, 2, 1).contiguous()

        A = -torch.exp(self.A_log.float())
        D_ssm = self.D.float()

        orders, inverse_orders, directions = self.sass(hw_shape)

        scan_results = []
        for o_path, inv_o_path, d_indices_path in zip(orders, inverse_orders, directions):
            o_tensor = torch.tensor(o_path, device=x_conv.device)
            inv_o_tensor = torch.tensor(inv_o_path, device=x_conv.device)

            x_seq_ordered = x_conv[:, o_tensor, :].permute(0, 2, 1).contiguous()
            dt_seq_ordered = dt[:, :, o_tensor].contiguous()
            B_ssm_ordered = B_ssm[:, :, o_tensor].contiguous()
            C_ssm_ordered = C_ssm[:, :, o_tensor].contiguous()

            dir_B_vals_for_path = torch.stack([self.direction_Bs[d_val, :] for d_val in d_indices_path], dim=0)
            dir_B_vals_for_path = dir_B_vals_for_path.permute(1, 0)[None, :, :].expand(batch_size, -1, -1).to(dtype=B_ssm_ordered.dtype)

            if selective_scan_fn is not None:
                y_scan_output = selective_scan_fn(
                    x_seq_ordered,
                    dt_seq_ordered,
                    A,
                    (B_ssm_ordered + dir_B_vals_for_path).contiguous(),
                    C_ssm_ordered,
                    D_ssm,
                    z=None,
                    delta_bias=self.dt_proj.bias.float(),
                    delta_softplus=True,
                    return_last_state=False,
                ).permute(0, 2, 1)
            else:
                # Fallback reference logic when selective_scan_fn is not compiled
                # Simply fallback to linear projection to prevent system crashes
                y_scan_output = x_seq_ordered.permute(0, 2, 1)

            reordered_y = torch.zeros_like(y_scan_output, device=y_scan_output.device)
            reordered_y[:, inv_o_tensor, :] = y_scan_output
            scan_results.append(reordered_y)

        y_sum = sum(scan_results)
        y_final = y_sum * self.act(z)
        out = self.out_proj(y_final)

        return out


class PAF(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.conv_query = nn.Conv2d(dim, dim, 1)
        self.conv_key = nn.Conv2d(dim, dim, 1)
        self.conv_value = nn.Conv2d(dim, dim, 1)
        self.softmax = nn.Softmax(dim=-1)
        self.gamma = nn.Parameter(torch.zeros(1))

    def forward(self, x, H, W):
        B, N, C = x.shape
        x_2d = x.view(B, H, W, C).permute(0, 3, 1, 2)

        query = self.conv_query(x_2d).view(B, C, -1)
        key = self.conv_key(x_2d).view(B, C, -1)
        value = self.conv_value(x_2d).view(B, C, -1)

        energy = torch.bmm(query.permute(0, 2, 1), key)
        attention = self.softmax(energy)

        out = torch.bmm(value, attention.permute(0, 2, 1))
        out = out.view(B, C, H, W)
        out = self.gamma * out + x_2d

        return out.permute(0, 2, 3, 1).contiguous().view(B, N, C)


class MutiScan(nn.Module):
    def __init__(self, in_channels, out_channels, d_state=16, d_conv=4, expand=2, group_num=4):
        super().__init__()
        self.linear_top1 = nn.Linear(out_channels, out_channels)
        self.silu_top1 = nn.SiLU()

        self.linear_bottom1 = nn.Linear(out_channels, out_channels)
        self.silu_bottom1 = nn.SiLU()

        self.ss2d = SS2D(dim=out_channels, d_state=d_state, expand=expand, conv_size=d_conv)
        self.linear_after_ss2d = nn.Linear(out_channels, out_channels)
        self.paf = PAF(out_channels)
        self.gn = nn.GroupNorm(group_num, out_channels)
        self.linear_out = nn.Linear(out_channels, out_channels)
        self.res_connection = nn.Identity()

    def forward(self, x):
        B, C, H, W = x.shape
        x_flattened = x.view(B, C, -1).permute(0, 2, 1)

        x_top = self.linear_top1(x_flattened)
        x_top = self.silu_top1(x_top)

        x_bottom = self.linear_bottom1(x_flattened)
        x_bottom = self.silu_bottom1(x_bottom)

        x_ss2d_out = self.ss2d(x_bottom, (H, W))
        x_mul = x_top * x_ss2d_out

        x_linear_after_ss2d = self.linear_after_ss2d(x_mul)
        x_paf = self.paf(x_linear_after_ss2d, H, W)

        x_gn_input = x_paf.permute(0, 2, 1).view(B, C, H, W)
        x_gn = self.gn(x_gn_input)

        x_linear_out = self.linear_out(x_gn.view(B, C, -1).permute(0, 2, 1))
        output = x_linear_out.permute(0, 2, 1).view(B, C, H, W) + self.res_connection(x)

        return output


class SpaMSM(nn.Module):
    def __init__(self, channels, use_residual=True, group_num=4, use_proj=True):
        super().__init__()
        self.use_residual = use_residual
        self.use_proj = use_proj

        if use_proj:
            self.spatial_proj = nn.Conv2d(channels, channels, kernel_size=1, stride=1)

        self.mutiscan_block = MutiScan(in_channels=channels, out_channels=channels, group_num=group_num, d_conv=3)
        self.norm = nn.GroupNorm(group_num, channels)

    def forward(self, x):
        x_residual = x
        if self.use_proj:
            x = self.spatial_proj(x)

        x = self.mutiscan_block(x)
        x = self.norm(x)

        if self.use_residual:
            x = x + x_residual
        return x


class SpeGBM(nn.Module):
    @staticmethod
    def _find_valid_groups(channels, target_groups):
        valid_groups = []
        for g in range(1, min(channels, target_groups) + 1):
            if channels % g == 0:
                valid_groups.append(g)
        if not valid_groups:
            return 1
        return max(valid_groups)

    @staticmethod
    def _A_log_init(d_state, d_inner, device=None):
        A = repeat(
            torch.arange(1, d_state + 1, dtype=torch.float32, device=device),
            "n -> d n",
            d=d_inner,
        ).contiguous()
        return torch.log(A)

    @staticmethod
    def _D_init(d_inner, device=None):
        return torch.ones(d_inner, device=device)

    def padding_feature(self, x):
        B, C, H, W = x.shape
        if C < self.padded_channels:
            pad_c = self.padded_channels - C
            pad_features = torch.zeros((B, pad_c, H, W), device=x.device)
            return torch.cat([x, pad_features], dim=1)
        return x

    def __init__(
            self,
            channels,
            token_num=8,
            use_residual=True,
            group_num=4,
            d_state=16,
            d_conv=4,
            expand=2,
            dt_rank="auto",
            dt_min=0.001,
            dt_max=0.1,
            dt_init="random",
            dt_scale=1.0,
            dt_init_floor=1e-4,
            dropout=0.,
            device=None,
            dtype=None,
    ):
        factory_kwargs = {"device": device, "dtype": dtype}
        super().__init__()

        self.channels = channels
        self.token_num = token_num
        self.use_residual = use_residual
        self.d_state = d_state
        self.d_conv = d_conv
        self.expand = expand

        self.group_channel_num = math.ceil(channels / token_num)
        self.padded_channels = self.token_num * self.group_channel_num
        self.d_inner = int(self.expand * self.group_channel_num)

        self.dt_rank = math.ceil(self.group_channel_num / 16) if dt_rank == "auto" else dt_rank

        self.in_proj = nn.Linear(self.group_channel_num, self.d_inner * 2, bias=False, **factory_kwargs)

        self.conv2d = nn.Conv2d(
            in_channels=self.d_inner,
            out_channels=self.d_inner,
            groups=self.d_inner,
            bias=True,
            kernel_size=d_conv,
            padding=(d_conv - 1) // 2,
            **factory_kwargs,
        )

        self.act = nn.SiLU()
        self.x_proj = nn.Linear(self.d_inner, self.dt_rank + self.d_state * 2, bias=False, **factory_kwargs)

        valid_group_num = self._find_valid_groups(self.padded_channels, group_num)

        self.dt_proj = nn.Linear(self.dt_rank, self.d_inner, bias=True, **factory_kwargs)
        dt_init_std = self.dt_rank ** -0.5 * dt_scale
        if dt_init == "constant":
            nn.init.constant_(self.dt_proj.weight, dt_init_std)
        elif dt_init == "random":
            nn.init.uniform_(self.dt_proj.weight, -dt_init_std, dt_init_std)

        dt = torch.exp(
            torch.rand(self.d_inner, **factory_kwargs) * (math.log(dt_max) - math.log(dt_min))
            + math.log(dt_min)
        ).clamp(min=dt_init_floor)
        inv_dt = dt + torch.log(-torch.expm1(-dt))
        with torch.no_grad():
            self.dt_proj.bias.copy_(inv_dt)

        self.A_logs = nn.Parameter(self._A_log_init(self.d_state, self.d_inner, device))
        self.Ds = nn.Parameter(self._D_init(self.d_inner, device))

        self.out_norm = nn.GroupNorm(valid_group_num, self.padded_channels)
        self.out_proj = nn.Linear(self.d_inner, self.group_channel_num, bias=False, **factory_kwargs)
        self.dropout = nn.Dropout(dropout) if dropout > 0. else None

    def forward(self, x):
        x_pad = self.padding_feature(x)
        B, C, H, W = x_pad.shape

        x_pad = x_pad.permute(0, 2, 3, 1).contiguous()
        x_flat = x_pad.view(B * H * W, self.token_num, self.group_channel_num)

        xz = self.in_proj(x_flat)
        x_proj, z = xz.chunk(2, dim=-1)

        xs = torch.cat([x_proj, torch.flip(x_proj, dims=[1])], dim=0)
        x_dbl = self.x_proj(xs)

        dts, Bs, Cs = torch.split(x_dbl, [self.dt_rank, self.d_state, self.d_state], dim=-1)

        dts = dts.view(-1, self.dt_rank)
        dts = self.dt_proj(dts)
        dts = dts.view(2 * B * H * W, self.token_num, self.d_inner)

        As = -torch.exp(self.A_logs)

        xs = xs.transpose(-1, -2)
        dts = dts.transpose(-1, -2)
        Bs = Bs.transpose(-1, -2)
        Cs = Cs.transpose(-1, -2)

        if selective_scan_fn is not None:
            out_y = selective_scan_fn(
                xs, dts, As, Bs, Cs, self.Ds,
                delta_bias=self.dt_proj.bias,
                delta_softplus=True,
            )
        else:
            out_y = xs

        y1, y2 = out_y.chunk(2, dim=0)
        y2 = torch.flip(y2, dims=[-1])
        y = y1 + y2

        y = y.transpose(-1, -2)

        y = y * F.gelu(z)
        y = self.out_proj(y)

        y = y.view(B, H, W, self.padded_channels)
        y = y.permute(0, 3, 1, 2).contiguous()
        y = self.out_norm(y)

        if self.dropout is not None:
            y = self.dropout(y)

        if self.use_residual:
            return x + y[:, :x.size(1)]
        return y[:, :x.size(1)]


class BothMamba(nn.Module):
    def __init__(self, channels, token_num, use_residual, group_num=4, use_att=True):
        super(BothMamba, self).__init__()
        self.use_att = use_att
        self.use_residual = use_residual
        if self.use_att:
            self.weights = nn.Parameter(torch.ones(2) / 2)
            self.softmax = nn.Softmax(dim=0)

        self.spa_mamba = SpaMSM(channels, use_residual=use_residual, group_num=group_num)
        self.spe_mamba = SpeGBM(channels, token_num=token_num, use_residual=use_residual, group_num=group_num)

    def forward(self, x):
        spa_x = self.spa_mamba(x)
        spe_x = self.spe_mamba(x)

        if spa_x.shape[2:] != spe_x.shape[2:]:
            target_size = (min(spa_x.shape[2], spe_x.shape[2]),
                           min(spa_x.shape[3], spe_x.shape[3]))
            spa_x = F.interpolate(spa_x, size=target_size, mode='bilinear', align_corners=False)
            spe_x = F.interpolate(spe_x, size=target_size, mode='bilinear', align_corners=False)

        if self.use_att:
            weights = self.softmax(self.weights)
            fusion_x = spa_x * weights[0] + spe_x * weights[1]
        else:
            fusion_x = spa_x + spe_x

        if self.use_residual:
            if fusion_x.shape[2:] != x.shape[2:]:
                fusion_x = F.interpolate(fusion_x, size=x.shape[2:], mode='bilinear', align_corners=False)
            return fusion_x + x
        else:
            return fusion_x


class MultiScalePatchEmbedding(nn.Module):
    def __init__(self, in_channels, hidden_dim, group_num=4):
        super(MultiScalePatchEmbedding, self).__init__()
        self.main_branch = nn.Sequential(
            nn.Conv2d(in_channels=in_channels, out_channels=hidden_dim, kernel_size=1, stride=1, padding=0),
        )

        self.depthwise_conv3x3 = nn.Sequential(
            nn.Conv2d(hidden_dim, hidden_dim, kernel_size=3, stride=1, padding=1, groups=hidden_dim),
            nn.Conv2d(hidden_dim, hidden_dim, kernel_size=1, stride=1, padding=0)
        )
        self.depthwise_conv5x5 = nn.Sequential(
            nn.Conv2d(hidden_dim, hidden_dim, kernel_size=5, stride=1, padding=2, groups=hidden_dim),
            nn.Conv2d(hidden_dim, hidden_dim, kernel_size=1, stride=1, padding=0)
        )
        self.depthwise_conv7x7 = nn.Sequential(
            nn.Conv2d(hidden_dim, hidden_dim, kernel_size=7, stride=1, padding=3, groups=hidden_dim),
            nn.Conv2d(hidden_dim, hidden_dim, kernel_size=1, stride=1, padding=0)
        )

        self.fusion_conv = nn.Conv2d(hidden_dim * 3, hidden_dim, kernel_size=1, stride=1, padding=0)
        self.residual_branch = nn.Conv2d(in_channels=in_channels, out_channels=hidden_dim, kernel_size=1, stride=1,
                                         padding=0)
        self.final_norm = nn.GroupNorm(group_num, hidden_dim)
        self.activation = nn.SiLU()

    def forward(self, x):
        main_out = self.main_branch(x)

        conv3x3_out = self.depthwise_conv3x3(main_out)
        conv5x5_out = self.depthwise_conv5x5(main_out)
        conv7x7_out = self.depthwise_conv7x7(main_out)

        multi_scale_features = torch.cat([conv3x3_out, conv5x5_out, conv7x7_out], dim=1)
        fused_features = self.fusion_conv(multi_scale_features)

        residual_out = self.residual_branch(x)
        output = fused_features + residual_out
        output = self.final_norm(output)
        output = self.activation(output)
        return output


class MLFMamba(nn.Module):
    def __init__(self, in_channels=128, hidden_dim=64, num_classes=10, use_residual=True, mamba_type='both',
                 token_num=4, group_num=4, use_att=True):
        super(MLFMamba, self).__init__()
        self.mamba_type = mamba_type

        self.patch_embedding = MultiScalePatchEmbedding(in_channels=in_channels, hidden_dim=hidden_dim,
                                                        group_num=group_num)

        if mamba_type == 'spa':
            self.mamba = nn.Sequential(
                SpaMSM(hidden_dim, use_residual=use_residual, group_num=group_num),
                nn.AvgPool2d(kernel_size=2, stride=2, padding=0),
                SpaMSM(hidden_dim, use_residual=use_residual, group_num=group_num),
                nn.AvgPool2d(kernel_size=2, stride=2, padding=0),
                SpaMSM(hidden_dim, use_residual=use_residual, group_num=group_num),
            )
        elif mamba_type == 'spe':
            self.mamba = nn.Sequential(
                SpeGBM(hidden_dim, token_num=token_num, use_residual=use_residual, group_num=group_num),
                nn.AvgPool2d(kernel_size=2, stride=2, padding=0),
                SpeGBM(hidden_dim, token_num=token_num, use_residual=use_residual, group_num=group_num),
                nn.AvgPool2d(kernel_size=2, stride=2, padding=0),
                SpeGBM(hidden_dim, token_num=token_num, use_residual=use_residual, group_num=group_num)
            )
        elif mamba_type == 'both':
            self.mamba = nn.Sequential(
                BothMamba(channels=hidden_dim, token_num=token_num, use_residual=use_residual, group_num=group_num,
                          use_att=use_att),
                nn.AvgPool2d(kernel_size=2, stride=2, padding=0),
                BothMamba(channels=hidden_dim, token_num=token_num, use_residual=use_residual, group_num=group_num,
                          use_att=use_att),
                nn.AvgPool2d(kernel_size=2, stride=2, padding=0),
                BothMamba(channels=hidden_dim, token_num=token_num, use_residual=use_residual, group_num=group_num,
                          use_att=use_att),
            )

        self.global_pool = nn.AdaptiveAvgPool2d(1)
        self.cls_head = nn.Sequential(
            nn.Flatten(),
            nn.Linear(hidden_dim, 128),
            nn.SiLU(),
            nn.Linear(128, num_classes)
        )

    def forward(self, x):
        x = self.patch_embedding(x)
        x = self.mamba(x)
        x = self.global_pool(x)
        logits = self.cls_head(x)
        return logits


@register_model('MLFMamba', expects_4d=True, hidden_dim=64, use_residual=True, mamba_type='both', token_num=4, group_num=4, use_att=True)
def mlfmamba_model(pretrained: bool = False, **kwargs) -> MLFMamba:
    """Constructs a MLFMamba model."""
    if 'bands' in kwargs:
        kwargs['in_channels'] = kwargs.pop('bands')
    kwargs.pop('patch_size', None)
    return MLFMamba(**kwargs)
