from models.registry import register_model
"""
HSIC_SClusterFormer: Deformable Convolution-Enhanced Hierarchical Transformer
With Spectral-Spatial Cluster Attention for Hyperspectral Image Classification

Paper: https://doi.org/10.1109/TIP.2024.3522809
GitHub: https://github.com/Fang666666/HSIC_SClusterFormer
Venue: IEEE Transactions on Image Processing (TIP)
Year: 2025
"""

import math
import torch
import torch.nn.functional as F
from einops import rearrange
from torch import nn


# ──────────────────────────────────────────────
# models/deform_conv_v3.py  (2D deformable conv)
# ──────────────────────────────────────────────

class DeformConv2d(nn.Module):
    def __init__(self, inc, outc, kernel_size=3, padding=1, stride=1, bias=None, modulation=False):
        super(DeformConv2d, self).__init__()
        self.inc = inc
        self.outc = outc
        self.kernel_size = kernel_size
        self.padding = padding
        self.stride = stride
        self.zero_padding = nn.ZeroPad2d(padding)
        self.conv = nn.Conv2d(inc, outc, kernel_size=kernel_size, stride=kernel_size, bias=bias)
        self.p_conv = nn.Conv2d(inc, 2 * kernel_size * kernel_size, kernel_size=3, padding=1, stride=stride)
        nn.init.constant_(self.p_conv.weight, 0)
        self.modulation = modulation
        if modulation:
            self.m_conv = nn.Conv2d(inc, kernel_size * kernel_size, kernel_size=3, padding=1, stride=stride)
            nn.init.constant_(self.m_conv.weight, 0)

    def forward(self, x):
        offset = self.p_conv(x)
        if self.modulation:
            m = torch.sigmoid(self.m_conv(x))
        dtype = offset.dtype
        device = offset.device
        ks = self.kernel_size
        N = offset.size(1) // 2
        if self.padding:
            x = self.zero_padding(x)
        p = self._get_p(offset, dtype, device)
        p = p.contiguous().permute(0, 2, 3, 1)
        q_lt = p.detach().floor()
        q_rb = q_lt + 1
        q_lt = torch.cat([torch.clamp(q_lt[..., :N], 0, x.size(2) - 1), torch.clamp(q_lt[..., N:], 0, x.size(3) - 1)], dim=-1).long()
        q_rb = torch.cat([torch.clamp(q_rb[..., :N], 0, x.size(2) - 1), torch.clamp(q_rb[..., N:], 0, x.size(3) - 1)], dim=-1).long()
        q_lb = torch.cat([q_lt[..., :N], q_rb[..., N:]], dim=-1)
        q_rt = torch.cat([q_rb[..., :N], q_lt[..., N:]], dim=-1)

        p = torch.cat([torch.clamp(p[..., :N], 0, x.size(2) - 1), torch.clamp(p[..., N:], 0, x.size(3) - 1)], dim=-1)

        g_lt = (1 + (q_lt[..., :N].type_as(p) - p[..., :N])) * (1 + (q_lt[..., N:].type_as(p) - p[..., N:]))
        g_rb = (1 - (q_rb[..., :N].type_as(p) - p[..., :N])) * (1 - (q_rb[..., N:].type_as(p) - p[..., N:]))
        g_lb = (1 + (q_lb[..., :N].type_as(p) - p[..., :N])) * (1 - (q_lb[..., N:].type_as(p) - p[..., N:]))
        g_rt = (1 - (q_rt[..., :N].type_as(p) - p[..., :N])) * (1 + (q_rt[..., N:].type_as(p) - p[..., N:]))

        x_q_lt = self._get_x_q(x, q_lt, N)
        x_q_rb = self._get_x_q(x, q_rb, N)
        x_q_lb = self._get_x_q(x, q_lb, N)
        x_q_rt = self._get_x_q(x, q_rt, N)

        x_offset = g_lt.unsqueeze(dim=1) * x_q_lt + \
                   g_rb.unsqueeze(dim=1) * x_q_rb + \
                   g_lb.unsqueeze(dim=1) * x_q_lb + \
                   g_rt.unsqueeze(dim=1) * x_q_rt

        if self.modulation:
            m = m.contiguous().permute(0, 2, 3, 1)
            m = m.unsqueeze(dim=1)
            m = torch.cat([m for _ in range(x_offset.size(1))], dim=1)
            x_offset *= m

        x_offset = self._reshape_x_offset(x_offset, ks)
        out = self.conv(x_offset)

        return out

    def _get_p_n(self, N, dtype, device):
        p_n_x, p_n_y = torch.meshgrid(
            torch.arange(-(self.kernel_size - 1) // 2, (self.kernel_size - 1) // 2 + 1, device=device),
            torch.arange(-(self.kernel_size - 1) // 2, (self.kernel_size - 1) // 2 + 1, device=device))

        p_n = torch.cat([torch.flatten(p_n_x), torch.flatten(p_n_y)], 0)
        p_n = p_n.view(1, 2 * N, 1, 1).to(dtype)

        return p_n

    def _get_p_0(self, h, w, N, dtype, device):
        p_0_x, p_0_y = torch.meshgrid(
            torch.arange(1, h * self.stride + 1, self.stride, device=device),
            torch.arange(1, w * self.stride + 1, self.stride, device=device))
        p_0_x = torch.flatten(p_0_x).view(1, 1, h, w).repeat(1, N, 1, 1)
        p_0_y = torch.flatten(p_0_y).view(1, 1, h, w).repeat(1, N, 1, 1)
        p_0 = torch.cat([p_0_x, p_0_y], 1).to(dtype)

        return p_0

    def _get_p(self, offset, dtype, device):
        N, h, w = offset.size(1) // 2, offset.size(2), offset.size(3)
        p_n = self._get_p_n(N, dtype, device)
        p_0 = self._get_p_0(h, w, N, dtype, device)
        p = p_0 + p_n + offset
        return p

    def _get_x_q(self, x, q, N):
        b, h, w, _ = q.size()
        padded_w = x.size(3)
        c = x.size(1)
        x = x.contiguous().view(b, c, -1)

        index = q[..., :N] * padded_w + q[..., N:]
        index = index.contiguous().unsqueeze(dim=1).expand(-1, c, -1, -1, -1).contiguous().view(b, c, -1)

        x_offset = x.gather(dim=-1, index=index).contiguous().view(b, c, h, w, N)

        return x_offset

    @staticmethod
    def _reshape_x_offset(x_offset, ks):
        b, c, h, w, N = x_offset.size()
        x_offset = torch.cat([x_offset[..., s:s + ks].contiguous().view(b, c, h, w * ks) for s in range(0, N, ks)], dim=-1)
        x_offset = x_offset.contiguous().view(b, c, h * ks, w * ks)

        return x_offset


# ──────────────────────────────────────────────
# models/Pseudo3DDeformConv.py
#
# Adaptation note: the original `spatial_conv`/`channel_compress` layers
# hardcode a spectral width of 30, tying the module to the paper's fixed
# PCA-reduction setting (pca_components=30 in their data pipeline). Here
# that literal is replaced with an explicit `pca_components` argument so
# the same layers work for whatever band count the benchmark provides.
# ──────────────────────────────────────────────

class DeformConv3d(nn.Module):
    def __init__(self, inc, outc, kernel_size=3, padding=1, bias=False, modulation=True, channel_expand=3, pca_components=30):
        super(DeformConv3d, self).__init__()
        self.inc = inc
        self.outc = outc
        self.channel_expand = channel_expand
        self.kernel_size = kernel_size
        self.padding = padding
        self.modulation = modulation
        self.pca_components = pca_components

        self.spatial_conv = DeformConv2d(
            pca_components, pca_components * channel_expand, kernel_size,
            padding=padding, stride=1,
            bias=bias, modulation=modulation
        )

        self.channel_compress = nn.Conv2d(pca_components * channel_expand, pca_components, kernel_size=1, bias=bias)

        self.spectral_conv = nn.Sequential(
            nn.Conv3d(outc, outc,
                      kernel_size=(3, 1, 1),
                      padding=(1, 0, 0), stride=1, bias=False),
            nn.BatchNorm3d(outc),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        B, T, C, H, W = x.shape
        x = x.permute(0, 2, 1, 3, 4).contiguous()
        x = x.view(B * T, C, H, W)

        out = self.spatial_conv(x)
        out = self.channel_compress(out)

        out = out.view(B, T, C, H, W)
        out = self.spectral_conv(out)

        return out


# ──────────────────────────────────────────────
# models/FS_Attention.py
# ──────────────────────────────────────────────

def get_freq_indices(method):
    assert method in ['top1', 'top2', 'top4', 'top8', 'top16', 'top32',
                       'bot1', 'bot2', 'bot4', 'bot8', 'bot16', 'bot32',
                       'low1', 'low2', 'low4', 'low8', 'low16', 'low32']
    num_freq = int(method[3:])
    if 'top' in method:
        all_top_indices_x = [0, 0, 6, 0, 0, 1, 1, 4, 5, 1, 3, 0, 0, 0, 3, 2, 4, 6, 3, 5, 5, 2, 6, 5, 5, 3, 3, 4, 2, 2, 6, 1]
        all_top_indices_y = [0, 1, 0, 5, 2, 0, 2, 0, 0, 6, 0, 4, 6, 3, 5, 2, 6, 3, 3, 3, 5, 1, 1, 2, 4, 2, 1, 1, 3, 0, 5, 3]
        mapper_x = all_top_indices_x[:num_freq]
        mapper_y = all_top_indices_y[:num_freq]
    elif 'low' in method:
        all_low_indices_x = [0, 0, 1, 1, 0, 2, 2, 1, 2, 0, 3, 4, 0, 1, 3, 0, 1, 2, 3, 4, 5, 0, 1, 2, 3, 4, 5, 6, 1, 2, 3, 4]
        all_low_indices_y = [0, 1, 0, 1, 2, 0, 1, 2, 2, 3, 0, 0, 4, 3, 1, 5, 4, 3, 2, 1, 0, 6, 5, 4, 3, 2, 1, 0, 6, 5, 4, 3]
        mapper_x = all_low_indices_x[:num_freq]
        mapper_y = all_low_indices_y[:num_freq]
    elif 'bot' in method:
        all_bot_indices_x = [6, 1, 3, 3, 2, 4, 1, 2, 4, 4, 5, 1, 4, 6, 2, 5, 6, 1, 6, 2, 2, 4, 3, 3, 5, 5, 6, 2, 5, 5, 3, 6]
        all_bot_indices_y = [6, 4, 4, 6, 6, 3, 1, 4, 4, 5, 6, 5, 2, 2, 5, 1, 4, 3, 5, 0, 3, 1, 1, 2, 4, 2, 1, 1, 5, 3, 3, 3]
        mapper_x = all_bot_indices_x[:num_freq]
        mapper_y = all_bot_indices_y[:num_freq]
    else:
        raise NotImplementedError
    return mapper_x, mapper_y


class FreqSpectralAttentionLayer(torch.nn.Module):
    def __init__(self, channel, dct_h, dct_w, reduction=16, freq_sel_method='top16'):
        super(FreqSpectralAttentionLayer, self).__init__()
        self.reduction = reduction
        self.dct_h = dct_h
        self.dct_w = dct_w

        mapper_x, mapper_y = get_freq_indices(freq_sel_method)
        self.num_split = len(mapper_x)
        mapper_x = [temp_x * (dct_h // 7) for temp_x in mapper_x]
        mapper_y = [temp_y * (dct_w // 7) for temp_y in mapper_y]

        self.dct_layer = MultiSpectralDCTLayer(dct_h, dct_w, mapper_x, mapper_y, channel)
        self.fc = nn.Sequential(
            nn.Linear(channel, channel // reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(channel // reduction, channel, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x):
        n, c, h, w = x.shape
        x_pooled = x
        if h != self.dct_h or w != self.dct_w:
            x_pooled = torch.nn.functional.adaptive_avg_pool2d(x, (self.dct_h, self.dct_w))
        y = self.dct_layer(x_pooled)

        y = self.fc(y).view(n, c, 1, 1)
        return x * y.expand_as(x)


class MultiSpectralDCTLayer(nn.Module):
    """Generate dct filters."""
    def __init__(self, height, width, mapper_x, mapper_y, channel):
        super(MultiSpectralDCTLayer, self).__init__()

        assert len(mapper_x) == len(mapper_y)
        assert channel % len(mapper_x) == 0

        self.num_freq = len(mapper_x)
        self.register_buffer('weight', self.get_dct_filter(height, width, mapper_x, mapper_y, channel))

    def forward(self, x):
        assert len(x.shape) == 4, 'x must been 4 dimensions, but got ' + str(len(x.shape))

        x = x * self.weight

        result = torch.sum(x, dim=[2, 3])
        return result

    def build_filter(self, pos, freq, POS):
        result = math.cos(math.pi * freq * (pos + 0.5) / POS) / math.sqrt(POS)
        if freq == 0:
            return result
        else:
            return result * math.sqrt(2)

    def get_dct_filter(self, tile_size_x, tile_size_y, mapper_x, mapper_y, channel):
        dct_filter = torch.zeros(channel, tile_size_x, tile_size_y)

        c_part = channel // len(mapper_x)

        for i, (u_x, v_y) in enumerate(zip(mapper_x, mapper_y)):
            for t_x in range(tile_size_x):
                for t_y in range(tile_size_y):
                    dct_filter[i * c_part: (i + 1) * c_part, t_x, t_y] = self.build_filter(t_x, u_x, tile_size_x) * self.build_filter(t_y, v_y, tile_size_y)

        return dct_filter


# ──────────────────────────────────────────────
# models/CrossAttention.py
# ──────────────────────────────────────────────

class LayerNormalize(nn.Module):
    def __init__(self, dim, fn):
        super().__init__()
        self.norm = nn.LayerNorm(dim)
        self.fn = fn

    def forward(self, x, **kwargs):
        return self.fn(self.norm(x), **kwargs)


class MLP_Block(nn.Module):
    def __init__(self, dim, hidden_dim, dropout=0.1):
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


class CTAttention(nn.Module):
    def __init__(self, dim, heads=8, dropout=0.1):
        super().__init__()
        self.heads = heads
        self.scale = dim ** -0.5

        self.to_q = nn.Linear(dim, dim, bias=True)
        self.to_k = nn.Linear(dim, dim, bias=True)
        self.to_v = nn.Linear(dim, dim, bias=True)
        self.nn1 = nn.Linear(dim, dim)
        self.do1 = nn.Dropout(dropout)

    def forward(self, x, mask=None):
        b, n, _, h = *x.shape, self.heads

        x12 = torch.chunk(x, chunks=2, dim=0)
        x1 = x12[0]
        x2 = x12[1]

        q = self.to_q(x2)
        k = self.to_k(x1)
        v = self.to_v(x1)
        qkv = []
        qkv.append(q)
        qkv.append(k)
        qkv.append(v)
        q, k, v = map(lambda t: rearrange(t, 'b n (h d) -> b h n d', h=h), qkv)

        dots = torch.einsum('bhid,bhjd->bhij', q, k) * self.scale

        attn = dots.softmax(dim=-1)

        out = torch.einsum('bhij,bhjd->bhid', attn, v)
        out = rearrange(out, 'b h n d -> b n (h d)')
        out = self.nn1(out)
        out = self.do1(out)
        return out


class CT_Transformer(nn.Module):
    def __init__(self, h_dim, depth, heads, dropout):
        super().__init__()
        self.layers = nn.ModuleList([])
        for _ in range(depth):
            self.layers.append(
                LayerNormalize(h_dim, CTAttention(h_dim, heads=heads, dropout=dropout))
            )

    def forward(self, h_tokens):
        for h_attend_lg in self.layers:
            h_tokens = h_attend_lg(h_tokens)
        return h_tokens


class FusionEncoder(nn.Module):
    """
    Adaptation note: the original `pojo` projection hardcodes an input
    sequence length of 16 tokens (= 4x4), valid only for the paper's
    specific default patch size of 13 (which downsamples to a 4x4 grid).
    Here `lower_seq_len` is passed explicitly (computed from the actual
    spatial patch size) so the same Conv1d works for any patch size.
    """
    def __init__(self, depth, h_dim, ct_attn_heads, ct_attn_depth, dropout=0.1, patchsize=13, lower_seq_len=16):
        super().__init__()
        self.pojo = nn.Conv1d(lower_seq_len, patchsize ** 2, kernel_size=1, stride=1)
        self.layers = nn.ModuleList([])
        for _ in range(depth):
            self.layers.append(
                CT_Transformer(h_dim=h_dim, depth=ct_attn_depth, heads=ct_attn_heads, dropout=dropout)
            )

    def forward(self, h_tokens, l_tokens):
        l_tokens = self.pojo(l_tokens)
        h_tokens = torch.concat((h_tokens, l_tokens), dim=0)

        for cross_attend in self.layers:
            h_tokens = cross_attend(h_tokens)

        return h_tokens


# ──────────────────────────────────────────────
# models/SClusterFormer.py
# ──────────────────────────────────────────────

class Mlp(nn.Module):
    def __init__(self, in_features, hidden_features=None, out_features=None, drop=0.):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = nn.GELU()
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop = nn.Dropout(drop)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x


def pairwise_cos_sim(x1: torch.Tensor, x2: torch.Tensor):
    x1 = F.normalize(x1, dim=-1)
    x2 = F.normalize(x2, dim=-1)

    sim = torch.matmul(x1, x2.transpose(-2, -1))
    return sim


def pairwise_euclidean_sim(x1: torch.Tensor, x2: torch.Tensor):
    x1 = F.normalize(x1, dim=-1)
    x2 = F.normalize(x2, dim=-1)

    em = torch.norm(x1.unsqueeze(-2) - x2.unsqueeze(-3), dim=-1)

    sim = torch.exp(-em)

    return sim


class Cluster3D(nn.Module):
    def __init__(self, patch_size=13, dim=256, out_dim=256, proposal_w=2, proposal_h=2, fold_w=1, fold_h=1, heads=4,
                 head_dim=24, return_center=False):
        super().__init__()
        self.patch_size = patch_size
        self.heads = heads
        self.head_dim = head_dim
        self.f = nn.Conv2d(dim, heads * head_dim, kernel_size=1)
        self.proj = nn.Conv2d(heads * head_dim, out_dim, kernel_size=1)
        self.v = nn.Conv2d(dim, heads * head_dim, kernel_size=1)
        self.sim_alpha = nn.Parameter(torch.ones(1))
        self.sim_beta = nn.Parameter(torch.zeros(1))
        self.centers_proposal = nn.AdaptiveAvgPool3d((1, proposal_w, proposal_h))
        self.fold_w = fold_w
        self.fold_h = fold_h
        self.return_center = return_center
        self.rule1 = nn.Sequential(
            nn.LeakyReLU(0.2, inplace=True),
            nn.Dropout(0.4)
        )

    def forward(self, x):  # [b, n, c]
        x = rearrange(x, "b (w h) c -> b c w h", w=self.patch_size, h=self.patch_size)
        value = self.v(x)
        x = self.f(x)
        x = rearrange(x, "b (e c) w h -> (b e) c w h", e=self.heads)
        value = rearrange(value, "b (e c) w h -> (b e) c w h", e=self.heads)
        if self.fold_w > 1 and self.fold_h > 1:
            b0, c0, w0, h0 = x.shape
            assert w0 % self.fold_w == 0 and h0 % self.fold_h == 0, \
                f"Ensure the feature map size ({w0}*{h0}) can be divided by fold {self.fold_w}*{self.fold_h}"
            x = rearrange(x, "b c (f1 w) (f2 h) -> (b f1 f2) c w h", f1=self.fold_w,
                          f2=self.fold_h)
            value = rearrange(value, "b c (f1 w) (f2 h) -> (b f1 f2) c w h", f1=self.fold_w, f2=self.fold_h)
        b, c, w, h = x.shape
        x = x.unsqueeze(2)
        value = value.unsqueeze(2)
        centers = rearrange(self.centers_proposal(x),
                            'b c d w h -> b (c d) w h')
        value_centers = rearrange(self.centers_proposal(value), 'b c d w h -> b (w h) (c d)')
        b, c, ww, hh = centers.shape
        sim = self.rule1(
            self.sim_beta +
            self.sim_alpha * pairwise_cos_sim(
                centers.reshape(b, c, -1).permute(0, 2, 1),
                x.reshape(b, c, -1).permute(0, 2, 1)
            )
        )
        sim_max, sim_max_idx = sim.max(dim=1, keepdim=True)
        mask = torch.zeros_like(sim)
        mask.scatter_(1, sim_max_idx, 1.)
        sim = sim * mask
        value2 = rearrange(value, 'b c d w h -> b (w h) (c d)')
        out = ((value2.unsqueeze(dim=1) * sim.unsqueeze(dim=-1)).sum(dim=2) + value_centers) / (mask.sum(dim=-1, keepdim=True) + 1.0)

        if self.return_center:
            out = rearrange(out, "b (w h) c -> b c w h", w=ww)
        else:
            out = (out.unsqueeze(dim=2) * sim.unsqueeze(dim=-1)).sum(dim=1)
            out = rearrange(out, "b (w h) c -> b c w h", w=w)

        if self.fold_w > 1 and self.fold_h > 1:
            out = rearrange(out, "(b f1 f2) c w h -> b c (f1 w) (f2 h)", f1=self.fold_w, f2=self.fold_h)
        out = rearrange(out, "(b e) c w h -> b (e c) w h", e=self.heads)
        out = self.proj(out)
        out = rearrange(out, "b c w h -> b (w h) c")
        return out


class Cluster2D(nn.Module):
    def __init__(self, patch_size, dim=768, out_dim=768, proposal_w=2, proposal_h=2, fold_w=2, fold_h=2, heads=4,
                 head_dim=24,
                 return_center=False):
        super().__init__()
        self.patch_size = patch_size
        self.heads = heads
        self.head_dim = head_dim
        self.f = nn.Conv2d(dim, heads * head_dim, kernel_size=1)
        self.proj = nn.Conv2d(heads * head_dim, out_dim, kernel_size=1)
        self.v = nn.Conv2d(dim, heads * head_dim, kernel_size=1)
        self.sim_alpha = nn.Parameter(torch.ones(1))
        self.sim_beta = nn.Parameter(torch.zeros(1))
        self.centers_proposal = nn.AdaptiveAvgPool2d((proposal_w, proposal_h))
        self.fold_w = fold_w
        self.fold_h = fold_h
        self.return_center = return_center
        self.rule2 = nn.Sequential(
            nn.LeakyReLU(0.2, inplace=True),
            nn.Dropout(0.4)
        )

    def forward(self, x):
        x = rearrange(x, "b (w h) c -> b c w h", w=self.patch_size, h=self.patch_size)
        value = self.v(x)
        x = self.f(x)
        x = rearrange(x, "b (e c) w h -> (b e) c w h", e=self.heads)
        value = rearrange(value, "b (e c) w h -> (b e) c w h", e=self.heads)
        if self.fold_w > 1 and self.fold_h > 1:
            b0, c0, w0, h0 = x.shape
            assert w0 % self.fold_w == 0 and h0 % self.fold_h == 0, \
                f"Ensure the feature map size ({w0}*{h0}) can be divided by fold {self.fold_w}*{self.fold_h}"
            x = rearrange(x, "b c (f1 w) (f2 h) -> (b f1 f2) c w h", f1=self.fold_w,
                          f2=self.fold_h)
            value = rearrange(value, "b c (f1 w) (f2 h) -> (b f1 f2) c w h", f1=self.fold_w, f2=self.fold_h)
        b, c, w, h = x.shape
        centers = self.centers_proposal(x)
        value_centers = rearrange(self.centers_proposal(value), 'b c w h -> b (w h) c')
        b, c, ww, hh = centers.shape
        sim = self.rule2(
            self.sim_beta +
            self.sim_alpha * pairwise_euclidean_sim(
                centers.reshape(b, c, -1).permute(0, 2, 1),
                x.reshape(b, c, -1).permute(0, 2, 1)
            )
        )
        sim_max, sim_max_idx = sim.max(dim=1, keepdim=True)
        mask = torch.zeros_like(sim)
        mask.scatter_(1, sim_max_idx, 1.)
        sim = sim * mask
        value2 = rearrange(value, 'b c w h -> b (w h) c')
        out = ((value2.unsqueeze(dim=1) * sim.unsqueeze(dim=-1)).sum(dim=2) + value_centers) / (
                mask.sum(dim=-1, keepdim=True) + 1.0)

        if self.return_center:
            out = rearrange(out, "b (w h) c -> b c w h", w=ww)
        else:
            out = (out.unsqueeze(dim=2) * sim.unsqueeze(dim=-1)).sum(dim=1)
            out = rearrange(out, "b (w h) c -> b c w h", w=w)

        if self.fold_w > 1 and self.fold_h > 1:
            out = rearrange(out, "(b f1 f2) c w h -> b c (f1 w) (f2 h)", f1=self.fold_w, f2=self.fold_h)
        out = rearrange(out, "(b e) c w h -> b (e c) w h", e=self.heads)
        out = self.proj(out)
        out = rearrange(out, "b c w h -> b (w h) c")
        return out


class GroupedPixelEmbedding(nn.Module):
    def __init__(self, in_feature_map_size=7, in_chans=3, embed_dim=128, n_groups=1):
        super().__init__()
        self.ifm_size = in_feature_map_size
        self.proj = nn.Conv2d(in_chans, embed_dim, kernel_size=3, stride=1, padding=1, groups=n_groups)
        self.batch_norm = nn.BatchNorm2d(embed_dim)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        x = self.proj(x)
        x = self.relu(self.batch_norm(x))

        x = x.flatten(2).transpose(1, 2)

        after_feature_map_size = self.ifm_size

        return x, after_feature_map_size


class PixelEmbedding(nn.Module):
    def __init__(self, in_feature_map_size=7, in_chans=3, embed_dim=128, n_groups=1, i=0):
        super().__init__()
        self.ifm_size = in_feature_map_size
        self.proj = nn.Conv2d(in_chans, embed_dim, kernel_size=3, stride=1 if i == 0 else 2,
                              padding=1 if i == 0 else (3 // 2, 3 // 2))
        self.batch_norm = nn.BatchNorm2d(embed_dim)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        x = self.proj(x)
        x = self.relu(self.batch_norm(x))

        after_feature_map_size = x.shape[2]

        x = x.flatten(2).transpose(1, 2)

        return x, after_feature_map_size


class Block(nn.Module):
    def __init__(self, patch_size, dim, num_heads, mlp_ratio=4, drop=0.):
        super().__init__()
        self.norm1 = nn.LayerNorm(dim)
        self.attn = Cluster3D(patch_size=patch_size, dim=dim, out_dim=dim, proposal_w=4, proposal_h=4, fold_w=1,
                              fold_h=1, heads=num_heads, head_dim=24)
        self.norm2 = nn.LayerNorm(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = Mlp(in_features=dim, hidden_features=mlp_hidden_dim, drop=drop)

    def forward(self, x):
        x = x + self.attn(self.norm1(x))
        x = x + self.mlp(self.norm2(x))
        return x


class Block2D(nn.Module):
    def __init__(self, patch_size, dim, num_heads, mlp_ratio=4, drop=0.):
        super().__init__()
        self.norm1 = nn.LayerNorm(dim)
        self.attn = Cluster2D(patch_size=patch_size, dim=dim, out_dim=dim, proposal_w=4, proposal_h=4, fold_w=1,
                              fold_h=1, heads=num_heads, head_dim=24)
        self.norm2 = nn.LayerNorm(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = Mlp(in_features=dim, hidden_features=mlp_hidden_dim, drop=drop)

    def forward(self, x):
        x = x + self.attn(self.norm1(x))
        x = x + self.mlp(self.norm2(x))
        return x


class MultiScaleDeformConv3D_FSA(nn.Module):
    """
    Adaptation note: `out_channels` is derived from `deform_conv.pca_components`
    (previously hardcoded to a literal 30) so the frequency-attention layer's
    channel count tracks whatever spectral width the wrapped DeformConv3d
    was actually built with.
    """
    def __init__(self, deform_conv: nn.Module, kernel_sizes=[3, 5, 7]):
        super().__init__()
        self.kernel_sizes = kernel_sizes
        self.deform_conv = deform_conv
        self.out_channels = deform_conv.outc * deform_conv.pca_components

        self.scale_convs = nn.ModuleList([
            nn.Conv3d(deform_conv.outc, deform_conv.outc, kernel_size=1, groups=deform_conv.outc)
            for _ in kernel_sizes
        ])

        self.fuse = nn.Conv3d(
            deform_conv.outc * len(kernel_sizes),
            deform_conv.outc,
            kernel_size=1
        )

        self.attention = FreqSpectralAttentionLayer(
            channel=self.out_channels,
            dct_h=kernel_sizes[-1],
            dct_w=kernel_sizes[-1],
            reduction=16,
            freq_sel_method='top2'
        )

    def forward(self, x):
        B, C, D, H, W = x.shape
        feat_base = self.deform_conv(x)
        feats = []
        for idx, k in enumerate(self.kernel_sizes):
            if k == self.kernel_sizes[0]:
                feat_scaled = feat_base
            else:
                scale = k / self.kernel_sizes[0]
                feat_scaled = F.avg_pool3d(
                    feat_base,
                    kernel_size=(1, int(scale), int(scale)),
                    stride=(1, int(scale), int(scale)),
                    ceil_mode=True
                )
                feat_scaled = F.interpolate(
                    feat_scaled, size=(D, H, W),
                    mode='trilinear', align_corners=False
                )

            feat_scaled = self.scale_convs[idx](feat_scaled)
            feats.append(feat_scaled)

        multi_scale = torch.cat(feats, dim=1)
        multi_scale = self.fuse(multi_scale)

        out = feat_base + multi_scale

        B, Ck, D, H, W = out.shape
        out_4d = out.view(B, Ck * D, H, W)
        out_attn = self.attention(out_4d)
        out_final = out_attn.view(B, Ck, D, H, W)

        return out_final


class MultiScaleDeformConv2D(nn.Module):
    def __init__(self, deform_conv: nn.Module, kernel_sizes=[3, 5, 7]):
        super().__init__()
        self.kernel_sizes = kernel_sizes
        self.deform_conv = deform_conv
        self.fuse = nn.Conv2d(deform_conv.outc * 3, deform_conv.outc, kernel_size=1)

    def forward(self, x):  # x: [B, C, H, W]
        B, C, H, W = x.shape
        feats = []

        for k in self.kernel_sizes:
            scale = k / self.kernel_sizes[0]
            if scale != 1.0:
                x_scaled = F.interpolate(x, scale_factor=scale, mode='bilinear', align_corners=False)
            else:
                x_scaled = x

            feat = self.deform_conv(x_scaled)
            feat = F.interpolate(feat, size=(H, W), mode='bilinear', align_corners=False)
            feats.append(feat)

        out = torch.cat(feats, dim=1)
        out = self.fuse(out) + feats[1]

        return out


class SClusterFormer(nn.Module):
    """
    Adaptation notes vs. the original repo:
      - `pca_components` is no longer forced to the paper's fixed value of
        30; it is derived by the registered factory below from the
        benchmark's `bands` argument (bands - emap_components, rounded down
        to an even number for the DCT frequency split in
        FreqSpectralAttentionLayer). The internal deformable-conv layers
        that used to hardcode 30 as their spectral width now take
        `pca_components` as a constructor argument (see
        MultiScaleDeformConv3D_FSA / DeformConv3d above), so the same
        upper/lower-branch architecture generalizes to any spectral width.
      - `FusionEncoder`'s cross-token projection (`pojo`) took a hardcoded
        input length of 16 tokens (only valid for the paper's default
        patch_size=13). It now takes `lower_seq_len`, computed here from
        the actual `img_size` so the fusion module works for any patch size.
    No new layers/attention mechanisms were introduced; only these
    shape-determining literals were parameterized.
    """
    def __init__(self, img_size=224, pca_components=3, emap_components=1, num_classes=1000, num_stages=3,
                 n_groups=[16, 16, 16], embed_dims=[256, 128, 64], num_heads=[8, 8, 8], mlp_ratios=[1, 1, 1],
                 depths=[2, 1, 1], patchsize=17):
        super().__init__()
        self.reducedbands = pca_components
        self.num_stages = num_stages

        new_bands = math.ceil(pca_components / n_groups[0]) * n_groups[0]
        self.pad = nn.ReplicationPad3d((0, 0, 0, 0, 0, new_bands - pca_components))

        """MDC-FSA"""
        deform_conv_shared = DeformConv3d(inc=1, outc=1, kernel_size=3, padding=1, bias=False, modulation=True,
                                           pca_components=pca_components)
        self.deform_conv_layer_pca = MultiScaleDeformConv3D_FSA(deform_conv_shared)

        deform_conv_shared_emap = DeformConv2d(inc=emap_components, outc=pca_components, kernel_size=9, padding=1, bias=False, modulation=True)
        self.deform_conv_layer_emap = MultiScaleDeformConv2D(deform_conv_shared_emap)

        """Upper Branch"""
        for i in range(num_stages):
            patch_embed = GroupedPixelEmbedding(
                in_feature_map_size=img_size,
                in_chans=new_bands if i == 0 else embed_dims[i - 1],
                embed_dim=embed_dims[i],
                n_groups=n_groups[i]
            )

            block = nn.ModuleList([Block(
                dim=embed_dims[i],
                num_heads=num_heads[i],
                mlp_ratio=mlp_ratios[i],
                drop=0.,
                patch_size=img_size) for j in range(depths[i])])

            norm2d = nn.LayerNorm(embed_dims[i])

            setattr(self, f"patch_embed{i + 1}", patch_embed)
            setattr(self, f"block{i + 1}", block)
            setattr(self, f"norm{i + 1}", norm2d)

        """Lower Branch"""
        self.embed_img = [img_size, math.ceil(img_size / 2), math.ceil(math.ceil(img_size / 2) / 2)]
        for i in range(num_stages):
            patch_embed2d = PixelEmbedding(
                in_feature_map_size=img_size if i == 0 else self.embed_img[i - 1],
                in_chans=new_bands if i == 0 else embed_dims[i - 1],
                embed_dim=embed_dims[i],
                n_groups=n_groups[i],
                i=i
            )

            block2d = nn.ModuleList([Block2D(
                dim=embed_dims[i],
                num_heads=num_heads[i],
                mlp_ratio=mlp_ratios[i],
                drop=0.,
                patch_size=self.embed_img[i]) for j in range(depths[i])])

            norm = nn.LayerNorm(embed_dims[i])

            setattr(self, f"patch_embed2d{i + 1}", patch_embed2d)
            setattr(self, f"block2d{i + 1}", block2d)
            setattr(self, f"norm2d{i + 1}", norm)

        """CFAF"""
        self.coefficients = torch.nn.Parameter(torch.Tensor([0.7]))

        self.fusion_encoder = FusionEncoder(
            depth=1,
            h_dim=embed_dims[-1],
            ct_attn_heads=4,
            ct_attn_depth=1,
            dropout=0.1,
            patchsize=patchsize,
            lower_seq_len=self.embed_img[-1] ** 2
        )

        self.head = nn.Sequential(
            nn.Linear(embed_dims[-1], num_classes),
            nn.Softmax(dim=1)
        )

    def forward_features_Upper(self, x):
        x = self.pad(x).squeeze(dim=1)
        B = x.shape[0]

        for i in range(self.num_stages):
            patch_embed = getattr(self, f"patch_embed{i + 1}")
            block = getattr(self, f"block{i + 1}")
            norm = getattr(self, f"norm{i + 1}")

            x, s = patch_embed(x)
            for blk in block:
                x = blk(x)

            x = norm(x)

            if i != self.num_stages - 1:
                x = x.reshape(B, s, s, -1).permute(0, 3, 1, 2).contiguous()

        return x

    def forward_features_Lower(self, x):
        x = self.pad(x).squeeze(dim=1)
        B = x.shape[0]

        for i in range(self.num_stages):
            patch_embed = getattr(self, f"patch_embed2d{i + 1}")
            block = getattr(self, f"block2d{i + 1}")
            norm = getattr(self, f"norm2d{i + 1}")

            x, s = patch_embed(x)
            for blk in block:
                x = blk(x)

            x = norm(x)

            if i != self.num_stages - 1:
                x = x.reshape(B, s, s, -1).permute(0, 3, 1, 2).contiguous()

        return x

    def forward(self, x):
        x1_pca = x[:, :, :self.reducedbands, :, :]
        x1_pca = self.deform_conv_layer_pca(x1_pca)
        x1_pca = x1_pca[:, :self.reducedbands, :, :]

        x_scluster = self.forward_features_Upper(x1_pca)

        x_emap = x[:, :, self.reducedbands, :, :]
        x_emap = self.deform_conv_layer_emap(x_emap)
        x_emap = torch.unsqueeze(x_emap, dim=1)

        x_emap = self.forward_features_Lower(x_emap)

        x_cfpf = self.fusion_encoder(x_scluster, x_emap)

        x_cfpf = x_cfpf.mean(dim=1)
        x_scluster = x_scluster.mean(dim=1)
        x_emap = x_emap.mean(dim=1)

        x_scluster = self.head(x_scluster)
        x_emap = self.head(x_emap)
        x_cfpf = self.head(x_cfpf)

        x = x_emap * ((1 - self.coefficients) / 2) + x_cfpf * (
                (1 - self.coefficients) / 2) + x_scluster * self.coefficients

        return x


# ──────────────────────────────────────────────
# Register model
# ──────────────────────────────────────────────

@register_model('HSIC_SClusterFormer', expects_4d=False, num_stages=3, n_groups=[16, 16, 16],
                 embed_dims=[256, 128, 64], num_heads=[8, 8, 8], mlp_ratios=[1, 1, 1], depths=[2, 1, 1])
def hsic_sclusterformer(pretrained: bool = False, **kwargs) -> SClusterFormer:
    """Constructs a HSIC_SClusterFormer model.

    The original model expects its input to already be PCA-reduced to a
    small number of spectral components (paper default: 30) with one extra
    appended EMAP (extended morphological attribute profile) channel. This
    benchmark instead feeds raw spectral bands directly, so here the last
    band of the input cube is treated as the "EMAP" channel and the
    remaining leading bands as the "PCA" channels (rounded down to an even
    count, required by the frequency-attention module's channel split).
    """
    bands = kwargs.pop('bands', 30)
    patch_size = kwargs.pop('patch_size', 11)
    if isinstance(patch_size, (list, tuple)):
        patch_size = int(patch_size[0])

    emap_components = 1
    pca_components = max(2, bands - emap_components)
    if pca_components % 2 != 0:
        pca_components -= 1

    kwargs.setdefault('img_size', patch_size)
    kwargs.setdefault('patchsize', patch_size)
    kwargs['pca_components'] = pca_components
    kwargs['emap_components'] = emap_components

    return SClusterFormer(**kwargs)
