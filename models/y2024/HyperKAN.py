from models.registry import register_model
"""
HyperKAN: Kolmogorov-Arnold Networks make Hyperspectral Image Classifiers Smarter

Paper: https://www.mdpi.com/1424-8220/24/23/7683 (arXiv: https://arxiv.org/abs/2407.05278)
GitHub: https://github.com/f-neumann77/HyperKAN
Venue: Sensors (MDPI)
Year: 2024

HyperKAN swaps Kolmogorov-Arnold (KAN) layers into several base HSI
architectures (1DCNN, 2DCNN, 3DCNN-Luo, 3DCNN-He, NM3DCNN, SSFTT). Per the
paper's own results table ("Third phase of experiments"), the SSFTT-KAN
variant is the best-performing configuration across datasets (highest
average OA/F1 of all reported KAN variants), so it is the one ported here
as the flagship "HyperKAN" model.

Architecture ("Full KAN" SSFTT): a 3D FastKAN-conv layer + 2D FastKAN-conv
layer for spectral-spatial feature extraction, followed by a learned
tokenizer, a small transformer whose attention/MLP projections are all
KAN layers (KAN_GPT), and a final KANLinear classification head.

Adapted from f-neumann77/HyperKAN — models/ssftt_full_kan.py and
models/kan_layers.py (KAN layer implementations bundled inline below,
based on https://github.com/IvanDrokin/torch-conv-kan, itself derived from
https://github.com/Blealtan/efficient-kan; no extra pip dependency needed).
The original n_bands-dependent conv2d input width is now computed
dynamically instead of assuming a hardcoded band count, so the model works
for any `bands`/`patch_size` the framework passes in.
"""
import math

import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange


# ----------------------------------------------------------------------------------------------------------------------
# KAN layer implementations (bundled from HyperKAN/models/kan_layers.py)
# ----------------------------------------------------------------------------------------------------------------------
class RadialBasisFunction(nn.Module):
    def __init__(self, grid_min: float = -2., grid_max: float = 2., num_grids: int = 4, denominator: float = None):
        super().__init__()
        grid = torch.linspace(grid_min, grid_max, num_grids)
        self.grid = nn.Parameter(grid, requires_grad=False)
        self.denominator = denominator or (grid_max - grid_min) / (num_grids - 1)

    def forward(self, x):
        return torch.exp(-((x[..., None] - self.grid) / self.denominator) ** 2)


class FastKANConvNDLayer(nn.Module):
    """Generic N-d KAN convolution: base (activation+conv) path + RBF-spline conv path."""

    def __init__(self, conv_class, norm_class, input_dim, output_dim, kernel_size,
                 groups=1, padding=0, stride=1, dilation=1,
                 ndim: int = 2, grid_size=8, base_activation=nn.SiLU, grid_range=(-2, 2), dropout=0.0, **norm_kwargs):
        super().__init__()
        self.inputdim = input_dim
        self.outdim = output_dim
        self.groups = groups
        self.ndim = ndim
        self.base_activation = base_activation()
        self.grid_range = grid_range

        self.base_conv = nn.ModuleList([conv_class(input_dim // groups, output_dim // groups, kernel_size,
                                                     stride, padding, dilation, groups=1, bias=False)
                                         for _ in range(groups)])
        self.spline_conv = nn.ModuleList([conv_class(grid_size * input_dim // groups, output_dim // groups,
                                                       kernel_size, stride, padding, dilation, groups=1, bias=False)
                                           for _ in range(groups)])
        self.layer_norm = nn.ModuleList([norm_class(input_dim // groups, **norm_kwargs) for _ in range(groups)])
        self.rbf = RadialBasisFunction(grid_range[0], grid_range[1], grid_size)

        self.dropout = None
        if dropout > 0:
            self.dropout = [nn.Dropout1d, nn.Dropout2d, nn.Dropout3d][ndim - 1](p=dropout)

        for conv_layer in self.base_conv:
            nn.init.kaiming_uniform_(conv_layer.weight, nonlinearity='linear')
        for conv_layer in self.spline_conv:
            nn.init.kaiming_uniform_(conv_layer.weight, nonlinearity='linear')

    def forward_fast_kan(self, x, group_index):
        base_output = self.base_conv[group_index](self.base_activation(x))
        if self.dropout is not None:
            x = self.dropout(x)
        spline_basis = self.rbf(self.layer_norm[group_index](x))
        spline_basis = spline_basis.moveaxis(-1, 2).flatten(1, 2)
        spline_output = self.spline_conv[group_index](spline_basis)
        return base_output + spline_output

    def forward(self, x):
        split_x = torch.split(x, self.inputdim // self.groups, dim=1)
        output = [self.forward_fast_kan(_x, i) for i, _x in enumerate(split_x)]
        return torch.cat(output, dim=1)


class FastKANConv3DLayer(FastKANConvNDLayer):
    def __init__(self, input_dim, output_dim, kernel_size, groups=1, padding=0, stride=1, dilation=1,
                 grid_size=8, base_activation=nn.SiLU, grid_range=(-2, 2), dropout=0.0,
                 norm_layer=nn.InstanceNorm3d, **norm_kwargs):
        super().__init__(nn.Conv3d, norm_layer, input_dim, output_dim, kernel_size,
                          groups=groups, padding=padding, stride=stride, dilation=dilation, ndim=3,
                          grid_size=grid_size, base_activation=base_activation, grid_range=grid_range,
                          dropout=dropout, **norm_kwargs)


class FastKANConv2DLayer(FastKANConvNDLayer):
    def __init__(self, input_dim, output_dim, kernel_size, groups=1, padding=0, stride=1, dilation=1,
                 grid_size=8, base_activation=nn.SiLU, grid_range=(-2, 2), dropout=0.0,
                 norm_layer=nn.InstanceNorm2d, **norm_kwargs):
        super().__init__(nn.Conv2d, norm_layer, input_dim, output_dim, kernel_size,
                          groups=groups, padding=padding, stride=stride, dilation=dilation, ndim=2,
                          grid_size=grid_size, base_activation=base_activation, grid_range=grid_range,
                          dropout=dropout, **norm_kwargs)


class KANLinear(nn.Module):
    """Efficient-KAN style KAN linear layer (B-spline + base activation)."""

    def __init__(self, in_features, out_features, grid_size=5, spline_order=3,
                 scale_noise=0.1, scale_base=1.0, scale_spline=1.0,
                 enable_standalone_scale_spline=True, base_activation=nn.SiLU,
                 grid_eps=0.02, grid_range=(-1, 1)):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.grid_size = grid_size
        self.spline_order = spline_order

        h = (grid_range[1] - grid_range[0]) / grid_size
        grid = (
            (torch.arange(-spline_order, grid_size + spline_order + 1) * h + grid_range[0])
            .expand(in_features, -1).contiguous()
        )
        self.register_buffer("grid", grid)

        self.base_weight = nn.Parameter(torch.Tensor(out_features, in_features))
        self.spline_weight = nn.Parameter(torch.Tensor(out_features, in_features, grid_size + spline_order))
        self.enable_standalone_scale_spline = enable_standalone_scale_spline
        if enable_standalone_scale_spline:
            self.spline_scaler = nn.Parameter(torch.Tensor(out_features, in_features))

        self.scale_noise = scale_noise
        self.scale_base = scale_base
        self.scale_spline = scale_spline
        self.base_activation = base_activation()
        self.grid_eps = grid_eps

        self.reset_parameters()

    def reset_parameters(self):
        nn.init.kaiming_uniform_(self.base_weight, a=math.sqrt(5) * self.scale_base)
        with torch.no_grad():
            noise = (
                (torch.rand(self.grid_size + 1, self.in_features, self.out_features) - 1 / 2)
                * self.scale_noise / self.grid_size
            )
            self.spline_weight.data.copy_(
                (self.scale_spline if not self.enable_standalone_scale_spline else 1.0)
                * self.curve2coeff(self.grid.T[self.spline_order:-self.spline_order], noise)
            )
            if self.enable_standalone_scale_spline:
                nn.init.kaiming_uniform_(self.spline_scaler, a=math.sqrt(5) * self.scale_spline)

    def b_splines(self, x: torch.Tensor):
        grid = self.grid
        x = x.unsqueeze(-1)
        bases = ((x >= grid[:, :-1]) & (x < grid[:, 1:])).to(x.dtype)
        for k in range(1, self.spline_order + 1):
            bases = (
                (x - grid[:, :-(k + 1)]) / (grid[:, k:-1] - grid[:, :-(k + 1)]) * bases[:, :, :-1]
            ) + (
                (grid[:, k + 1:] - x) / (grid[:, k + 1:] - grid[:, 1:(-k)]) * bases[:, :, 1:]
            )
        return bases.contiguous()

    def curve2coeff(self, x: torch.Tensor, y: torch.Tensor):
        A = self.b_splines(x).transpose(0, 1)
        B = y.transpose(0, 1)
        solution = torch.linalg.lstsq(A, B).solution
        result = solution.permute(2, 0, 1)
        return result.contiguous()

    @property
    def scaled_spline_weight(self):
        return self.spline_weight * (self.spline_scaler.unsqueeze(-1) if self.enable_standalone_scale_spline else 1.0)

    def forward(self, x: torch.Tensor):
        base_output = F.linear(self.base_activation(x), self.base_weight)
        spline_output = F.linear(self.b_splines(x).view(x.size(0), -1), self.scaled_spline_weight.view(self.out_features, -1))
        return base_output + spline_output


class KAN(nn.Module):
    def __init__(self, layers_hidden, grid_size=5, spline_order=3, scale_noise=0.1, scale_base=1.0,
                 scale_spline=1.0, base_activation=nn.SiLU, grid_eps=0.02, grid_range=(-1, 1)):
        super().__init__()
        self.layers = nn.ModuleList()
        for in_features, out_features in zip(layers_hidden, layers_hidden[1:]):
            self.layers.append(nn.BatchNorm1d(in_features))
            self.layers.append(KANLinear(in_features, out_features, grid_size=grid_size, spline_order=spline_order,
                                          scale_noise=scale_noise, scale_base=scale_base, scale_spline=scale_spline,
                                          base_activation=base_activation, grid_eps=grid_eps, grid_range=grid_range))

    def forward(self, x: torch.Tensor):
        for layer in self.layers:
            x = layer(x)
        return x


class KAN_GPT(nn.Module):
    """Sequence-shaped ([B, T, C]) wrapper around stacked KANLinear layers, used inside the transformer."""

    def __init__(self, width, grid=3, k=3, noise_scale=0.1, noise_scale_base=1.0, scale_spline=1.0,
                 base_fun=nn.SiLU, grid_eps=0.02, grid_range=(-1, 1)):
        super().__init__()
        self.layers = nn.ModuleList()
        for in_features, out_features in zip(width, width[1:]):
            self.layers.append(KANLinear(in_features, out_features, grid_size=grid, spline_order=grid,
                                          scale_noise=noise_scale, scale_base=noise_scale_base,
                                          scale_spline=scale_spline, base_activation=base_fun,
                                          grid_eps=grid_eps, grid_range=grid_range))
            self.layers.append(nn.BatchNorm1d(out_features))

    def forward(self, x: torch.Tensor):
        B, C, T = x.shape
        x = x.view(-1, T)
        for layer in self.layers:
            x = layer(x)
        U = x.shape[1]
        return x.view(B, C, U)


# ----------------------------------------------------------------------------------------------------------------------
# Transformer building blocks (from HyperKAN/models/ssftt_full_kan.py), attention/MLP projections use KAN_GPT
# ----------------------------------------------------------------------------------------------------------------------
class Residual(nn.Module):
    def __init__(self, fn):
        super().__init__()
        self.fn = fn

    def forward(self, x, **kwargs):
        return self.fn(x, **kwargs) + x


class LayerNormalize(nn.Module):
    def __init__(self, dim, fn):
        super().__init__()
        self.norm = nn.LayerNorm(dim)
        self.fn = fn

    def forward(self, x, **kwargs):
        return self.fn(self.norm(x), **kwargs)


class NewGELU(nn.Module):
    def forward(self, x):
        return 0.5 * x * (1.0 + torch.tanh(math.sqrt(2.0 / math.pi) * (x + 0.044715 * torch.pow(x, 3.0))))


class MLP_Block(nn.Module):
    def __init__(self, dim, hidden_dim, dropout=0.1):
        super().__init__()
        self.net = nn.Sequential(
            KAN_GPT(width=[dim, hidden_dim]),
            NewGELU(),
            nn.Dropout(dropout),
            KAN_GPT(width=[hidden_dim, dim]),
            nn.Dropout(dropout),
        )

    def forward(self, x):
        return self.net(x)


class Attention(nn.Module):
    def __init__(self, dim, heads=8, dropout=0.1):
        super().__init__()
        self.heads = heads
        self.scale = dim ** -0.5
        self.to_qkv = KAN_GPT(width=[dim, 3 * dim])
        self.nn1 = KAN_GPT(width=[dim, dim])
        self.do1 = nn.Dropout(dropout)

    def forward(self, x, mask=None):
        b, n, _, h = *x.shape, self.heads
        qkv = self.to_qkv(x).chunk(3, dim=-1)
        q, k, v = map(lambda t: rearrange(t, 'b n (h d) -> b h n d', h=h), qkv)

        dots = torch.einsum('bhid,bhjd->bhij', q, k) * self.scale
        if mask is not None:
            mask = F.pad(mask.flatten(1), (1, 0), value=True)
            mask = mask[:, None, :] * mask[:, :, None]
            dots.masked_fill_(~mask, float('-inf'))

        attn = dots.softmax(dim=-1)
        out = torch.einsum('bhij,bhjd->bhid', attn, v)
        out = rearrange(out, 'b h n d -> b n (h d)')
        out = self.nn1(out)
        out = self.do1(out)
        return out


class Transformer(nn.Module):
    def __init__(self, dim, depth, heads, mlp_dim, dropout):
        super().__init__()
        self.layers = nn.ModuleList([])
        for _ in range(depth):
            self.layers.append(nn.ModuleList([
                Residual(LayerNormalize(dim, Attention(dim, heads=heads, dropout=dropout))),
                Residual(LayerNormalize(dim, MLP_Block(dim, mlp_dim, dropout=dropout))),
            ]))

    def forward(self, x, mask=None):
        for attention, mlp in self.layers:
            x = attention(x, mask=mask)
            x = mlp(x)
        return x


# ----------------------------------------------------------------------------------------------------------------------
# HyperKAN (SSFTT-KAN "Full KAN" variant) — flagship/best-performing model in the HyperKAN paper
# ----------------------------------------------------------------------------------------------------------------------
class HyperKAN(nn.Module):
    """SSFTT-KAN: KAN-based spectral-spatial feature extractor + tokenizer + transformer + KAN head.

    Input: [B, C, H, W] (4D) where C = spectral bands.
    Output: [B, num_classes]
    """

    def __init__(self, num_classes, bands=30, patch_size=11, in_channels=1,
                 num_tokens=4, dim=64, depth=1, heads=8, mlp_dim=8,
                 dropout=0.1, emb_dropout=0.1):
        super().__init__()
        self.L = num_tokens
        self.cT = dim

        self.conv3d_features = nn.Sequential(
            FastKANConv3DLayer(in_channels, 8, kernel_size=(3, 3, 3), base_activation=nn.PReLU, grid_size=2),
            nn.BatchNorm3d(8),
            nn.ReLU(),
        )

        # spectral dim shrinks by (kernel-1)=2 after the 3D conv (no padding)
        conv2d_in_channels = 8 * (bands - 2)
        self.conv2d_features = nn.Sequential(
            FastKANConv2DLayer(conv2d_in_channels, 32, kernel_size=(3, 3), base_activation=nn.PReLU, grid_size=2),
            nn.BatchNorm2d(32),
            nn.ReLU(),
        )

        self.token_wA = nn.Parameter(torch.empty(1, self.L, 32), requires_grad=True)
        nn.init.xavier_normal_(self.token_wA)
        self.token_wV = nn.Parameter(torch.empty(1, 32, self.cT), requires_grad=True)
        nn.init.xavier_normal_(self.token_wV)

        self.pos_embedding = nn.Parameter(torch.empty(1, (num_tokens + 1), dim))
        nn.init.normal_(self.pos_embedding, std=.02)

        self.cls_token = nn.Parameter(torch.zeros(1, 1, dim))
        self.dropout = nn.Dropout(emb_dropout)

        self.transformer = Transformer(dim, depth, heads, mlp_dim, dropout)
        self.to_cls_token = nn.Identity()

        self.nn1 = KAN([dim, num_classes], base_activation=nn.PReLU, grid_size=2)

    def forward(self, x, mask=None):
        if x.dim() == 4:
            x = x.unsqueeze(1)  # [B, C, H, W] -> [B, 1, C, H, W]

        x = self.conv3d_features(x)
        x = rearrange(x, 'b c h w y -> b (c h) w y')
        x = self.conv2d_features(x)
        x = rearrange(x, 'b c h w -> b (h w) c')

        wa = rearrange(self.token_wA, 'b h w -> b w h')
        A = torch.einsum('bij,bjk->bik', x, wa)
        A = rearrange(A, 'b h w -> b w h')
        A = A.softmax(dim=-1)

        VV = torch.einsum('bij,bjk->bik', x, self.token_wV)
        T = torch.einsum('bij,bjk->bik', A, VV)

        cls_tokens = self.cls_token.expand(x.shape[0], -1, -1)
        x = torch.cat((cls_tokens, T), dim=1)
        x += self.pos_embedding
        x = self.dropout(x)
        x = self.transformer(x, mask)
        x = self.to_cls_token(x[:, 0])
        x = self.nn1(x)

        return x


@register_model('HyperKAN', expects_4d=True, num_tokens=4, dim=64, depth=1, heads=8, mlp_dim=8,
                 dropout=0.1, emb_dropout=0.1)
def hyperkan_model(pretrained: bool = False, **kwargs) -> HyperKAN:
    """Constructs the HyperKAN (SSFTT-KAN) model."""
    kwargs.pop('patch_size', None)
    kwargs.setdefault('in_channels', 1)
    return HyperKAN(**kwargs)
