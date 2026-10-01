from models.registry import register_model
"""
HSIConvKAN: Exploring Kolmogorov-Arnold Networks for Hyperspectral Image Classification

Paper: How to Learn More? Exploring Kolmogorov-Arnold Networks for
       Hyperspectral Image Classification, Remote Sensing 16(21):4015, 2024
       https://doi.org/10.3390/rs16214015
GitHub: https://github.com/aj1365/HSIConvKAN
Venue: Remote Sensing (MDPI)
Year: 2024

The repo's flagship model ("3D2DConvKAN" / HybridKAN, see HybridKAN.ipynb) is
a hybrid convolutional-KAN network: three 3D "1x1x1" KAN-convolution layers
(pointwise spectral mixing across bands, implemented via unfold + KANLinear,
i.e. effConvKAN3D) followed by a 2D KAN-convolution layer (ConvKAN, "Fast"
variant) that downsamples spatially, and a final FastKAN MLP head.

Adapted here for the framework's variable bands/patch_size interface:
  * effConvKAN3D originally depends on the external `unfoldNd` package; that
    unfold is reimplemented inline below with plain `torch.Tensor.unfold`
    (no extra dependency), producing identical patches.
  * The reference notebook hardcodes a maxpool kernel/final KAN input width
    (`nn.MaxPool2d(kernel_size=3, stride=3)` then `KAN([64, 32, NUM_CLASS])`)
    that only happens to work for its own patch_size and breaks for others
    (pooling a height-1 feature map with a kernel of 3). We keep the same
    layer stack (3x effConvKAN3D -> ConvKAN2D -> KAN head) but swap the
    fixed maxpool for `nn.AdaptiveMaxPool2d`, which reproduces the same kind
    of downsampling while making the classifier head's input size (and thus
    the whole model) independent of `patch_size`/`bands`.
  * KAN layer implementations (KANLinear / Fast_KANLinear / FastKAN) are
    bundled inline from HSIConvKAN/kan_linear.py and HSIConvKAN/fast_kan.py
    (both pure PyTorch, no extra dependency; fast_kan.py itself is a copy of
    https://github.com/ZiyaoLi/fast-kan bundled in the original repo).
"""
import math

import torch
import torch.nn as nn
import torch.nn.functional as F


# ----------------------------------------------------------------------------------------------------------------------
# KANLinear (efficient-kan style; from HSIConvKAN/kan_linear.py) — used inside effConvKAN3D
# ----------------------------------------------------------------------------------------------------------------------
class KANLinear(nn.Module):
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
        return solution.permute(2, 0, 1).contiguous()

    @property
    def scaled_spline_weight(self):
        return self.spline_weight * (self.spline_scaler.unsqueeze(-1) if self.enable_standalone_scale_spline else 1.0)

    def forward(self, x: torch.Tensor):
        assert x.size(-1) == self.in_features
        original_shape = x.shape
        x = x.reshape(-1, self.in_features)
        base_output = F.linear(self.base_activation(x), self.base_weight)
        spline_output = F.linear(self.b_splines(x).view(x.size(0), -1), self.scaled_spline_weight.view(self.out_features, -1))
        output = base_output + spline_output
        return output.view(*original_shape[:-1], self.out_features)


class effConvKAN3D(nn.Module):
    """3D KAN-convolution: unfold local (d,h,w) blocks and map each through a KANLinear.

    Equivalent to https://github.com/FirasBDarwish/ConvKAN3D but without the
    external `unfoldNd` dependency — unfolding is done with plain
    `torch.Tensor.unfold` over the three spatial/spectral axes.
    """

    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0, dilation=1, **kan_kwargs):
        super().__init__()
        if isinstance(kernel_size, int):
            kernel_size = (kernel_size, kernel_size, kernel_size)
        if isinstance(stride, int):
            stride = (stride, stride, stride)
        if isinstance(padding, int):
            padding = (padding, padding, padding)

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.dilation = dilation

        self.linear = KANLinear(in_channels * kernel_size[0] * kernel_size[1] * kernel_size[2],
                                 out_channels, **kan_kwargs)

    def forward(self, x):
        assert x.dim() == 5
        B, C, D, H, W = x.shape
        kD, kH, kW = self.kernel_size
        sD, sH, sW = self.stride
        pD, pH, pW = self.padding

        if pD or pH or pW:
            x = F.pad(x, (pW, pW, pH, pH, pD, pD))

        # unfold each spatial dim: -> [B, C, D', H', W', kD, kH, kW]
        x = x.unfold(2, kD, sD).unfold(3, kH, sH).unfold(4, kW, sW)
        Dp, Hp, Wp = x.shape[2], x.shape[3], x.shape[4]

        x = x.permute(0, 2, 3, 4, 1, 5, 6, 7).reshape(B * Dp * Hp * Wp, C * kD * kH * kW)
        out = self.linear(x)
        out = out.view(B, Dp, Hp, Wp, self.out_channels).permute(0, 4, 1, 2, 3).contiguous()
        return out


# ----------------------------------------------------------------------------------------------------------------------
# Fast-KAN (from HSIConvKAN/fast_kan.py, https://github.com/ZiyaoLi/fast-kan) — used for ConvKAN2D and the final head
# ----------------------------------------------------------------------------------------------------------------------
class SplineLinear(nn.Linear):
    def __init__(self, in_features: int, out_features: int, init_scale: float = 0.1, **kw):
        self.init_scale = init_scale
        super().__init__(in_features, out_features, bias=False, **kw)

    def reset_parameters(self):
        nn.init.trunc_normal_(self.weight, mean=0, std=self.init_scale)


class FastRadialBasisFunction(nn.Module):
    def __init__(self, grid_min: float = -2., grid_max: float = 2., num_grids: int = 8, denominator: float = None):
        super().__init__()
        grid = torch.linspace(grid_min, grid_max, num_grids)
        self.grid = nn.Parameter(grid, requires_grad=False)
        self.denominator = denominator or (grid_max - grid_min) / (num_grids - 1)

    def forward(self, x):
        return torch.exp(-((x[..., None] - self.grid) / self.denominator) ** 2)


class Fast_KANLinear(nn.Module):
    def __init__(self, input_dim: int, output_dim: int, grid_min: float = -2., grid_max: float = 2.,
                 num_grids: int = 8, use_base_update: bool = True, base_activation=F.silu,
                 spline_weight_init_scale: float = 0.1):
        super().__init__()
        self.layernorm = nn.LayerNorm(input_dim)
        self.rbf = FastRadialBasisFunction(grid_min, grid_max, num_grids)
        self.spline_linear = SplineLinear(input_dim * num_grids, output_dim, spline_weight_init_scale)
        self.use_base_update = use_base_update
        if use_base_update:
            self.base_activation = base_activation
            self.base_linear = nn.Linear(input_dim, output_dim)

    def forward(self, x):
        spline_basis = self.rbf(self.layernorm(x))
        ret = self.spline_linear(spline_basis.view(*spline_basis.shape[:-2], -1))
        if self.use_base_update:
            ret = ret + self.base_linear(self.base_activation(x))
        return ret


class FastKAN(nn.Module):
    def __init__(self, layers_hidden, grid_min: float = -2., grid_max: float = 2., num_grids: int = 8,
                 use_base_update: bool = True, base_activation=F.silu, spline_weight_init_scale: float = 0.1):
        super().__init__()
        self.layers = nn.ModuleList([
            Fast_KANLinear(in_dim, out_dim, grid_min=grid_min, grid_max=grid_max, num_grids=num_grids,
                            use_base_update=use_base_update, base_activation=base_activation,
                            spline_weight_init_scale=spline_weight_init_scale)
            for in_dim, out_dim in zip(layers_hidden[:-1], layers_hidden[1:])
        ])

    def forward(self, x):
        for layer in self.layers:
            x = layer(x)
        return x


class ConvKAN2D(nn.Module):
    """2D KAN-convolution (Fast-KAN variant): unfold local patches, map through Fast_KANLinear.

    Adapted from HSIConvKAN/ConvKAN.py (`version="Fast"` branch only, which
    is what the flagship HybridKAN model uses).
    """

    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0,
                 grid_size=5, scale_spline=1.0, base_activation=nn.SiLU(), grid_range=(-1, 1)):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding

        self.unfold = nn.Unfold(kernel_size, padding=padding, stride=stride)
        self.linear = Fast_KANLinear(
            input_dim=in_channels * kernel_size * kernel_size,
            output_dim=out_channels,
            num_grids=grid_size,
            spline_weight_init_scale=scale_spline,
            base_activation=base_activation,
            grid_min=grid_range[0],
            grid_max=grid_range[1],
        )

    def forward(self, x):
        assert x.dim() == 4
        batch_size, in_channels, height, width = x.size()
        assert in_channels == self.in_channels

        patches = self.unfold(x)
        patches = patches.transpose(1, 2)
        patches = patches.reshape(-1, in_channels * self.kernel_size * self.kernel_size)

        out = self.linear(patches)
        out = out.view(batch_size, -1, out.size(-1))

        out_height = (height + 2 * self.padding - self.kernel_size) // self.stride + 1
        out_width = (width + 2 * self.padding - self.kernel_size) // self.stride + 1

        out = out.transpose(1, 2)
        out = out.view(batch_size, self.out_channels, out_height, out_width)
        return out


# ----------------------------------------------------------------------------------------------------------------------
# HSIConvKAN (a.k.a. "3D2DConvKAN" / HybridKAN in the original repo) — flagship hybrid KAN model
# ----------------------------------------------------------------------------------------------------------------------
class HSIConvKAN(nn.Module):
    """Hybrid 3D+2D KAN-convolutional network for HSI classification.

    Input: [B, bands, H, W] (4D).
    Output: [B, num_classes]
    """

    def __init__(self, num_classes, bands=30, patch_size=11, pooled_size=4):
        super().__init__()
        self.bands = bands

        # Pointwise (kernel=1) 3D KAN convs: mix across the spectral/band channel dim
        self.convkan1 = effConvKAN3D(in_channels=bands, out_channels=8, kernel_size=1)
        self.convkan2 = effConvKAN3D(in_channels=8, out_channels=16, kernel_size=1)
        self.convkan3 = effConvKAN3D(in_channels=16, out_channels=32, kernel_size=1)

        # 2D KAN conv that downsamples the flattened (H*W) spatial axis
        self.convkan4 = ConvKAN2D(in_channels=32, out_channels=64, kernel_size=3, stride=2, padding=1)

        # Original repo used a fixed nn.MaxPool2d(kernel_size=3, stride=3), which only
        # works for the specific patch_size it was written for. AdaptiveMaxPool2d gives
        # the same kind of spatial downsampling while making the classifier head's input
        # size independent of patch_size/bands (framework requires variable patch_size).
        self.pool = nn.AdaptiveMaxPool2d((1, pooled_size))

        flat_dim = 64 * pooled_size
        self.head = FastKAN([flat_dim, 128, num_classes])

    def forward(self, x):
        if x.dim() == 5:
            x = x.squeeze(1)  # [B, 1, bands, H, W] -> [B, bands, H, W]

        B, C, H, W = x.shape
        x = x.unsqueeze(2)  # [B, bands, 1, H, W] : depth=1, channels=bands

        x = self.convkan1(x)
        x = self.convkan2(x)
        x = self.convkan3(x)

        # collapse the trivial depth dim into the spatial axis: [B, 32, 1, H, W] -> [B, 32, 1, H*W]
        x = x.reshape(x.shape[0], x.shape[1], x.shape[2], x.shape[3] * x.shape[4])

        x = self.pool(self.convkan4(x))
        x = torch.flatten(x, 1)
        x = self.head(x)
        return x


@register_model('HSIConvKAN', expects_4d=True, pooled_size=4)
def hsiconvkan_model(pretrained: bool = False, **kwargs) -> HSIConvKAN:
    """Constructs the HSIConvKAN (3D2DConvKAN / HybridKAN) model."""
    if 'bands' in kwargs:
        kwargs['bands'] = kwargs['bands']
    return HSIConvKAN(**kwargs)
