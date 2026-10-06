"""
PyS2CF-Mamba: A Pyramid Spatial-Spectral Competitive Fusion Mamba Network
for Hyperspectral Image Classification

Paper: https://doi.org/10.1109/LGRS.2026.3732153
GitHub: https://github.com/JiaxinLiCAS/PyS2CF-Mamba
Venue: IEEE GRSL
Year: 2026

Adaptation notes:
  - The original repo classifies full-scene tiles densely: the backbone keeps
    the input's spatial resolution, is downsampled once by 2x, and the
    classification head emits a per-location logit map (B, num_classes,
    H/2, W/2) that is later resized back to the full image for a whole-map
    prediction.
  - This benchmark instead feeds fixed-size patches and expects a single
    label per patch, so the dense logit map is reduced with the same
    AdaptiveAvgPool2d + Flatten head used by the other Mamba models in this
    repo (e.g. MambaHSI, IGroupSS-Mamba). No other part of the architecture
    (LPPS-Mamba, DGS-Mamba, channel-wise competitive fusion) is changed.
"""

import torch
from einops import rearrange
from mamba_ssm import Mamba
from torch import nn
from torch.nn import functional as F

from models.registry import register_model


def _validate_positive(name: str, value: int) -> int:
    value = int(value)
    if value <= 0:
        raise ValueError(f"{name} must be positive, got {value}.")
    return value


class PyramidChannelAttention(nn.Module):
    """Channel attention with dilated depthwise QKV projection (paper Eq. 3)."""

    def __init__(self, channels: int, num_heads: int = 4, dilation: int = 3):
        super().__init__()
        self.num_heads = _validate_positive("num_heads", num_heads)
        if channels % num_heads != 0:
            raise ValueError("channels must be divisible by num_heads.")

        dilation = _validate_positive("dilation", dilation)
        self.temperature = nn.Parameter(torch.ones(self.num_heads, 1, 1))
        self.qkv = nn.Conv2d(channels, channels * 3, kernel_size=1, bias=True)
        self.qkv_dwconv = nn.Conv2d(
            channels * 3,
            channels * 3,
            kernel_size=3,
            padding=dilation,
            dilation=dilation,
            groups=channels * 3,
            bias=True,
        )
        self.output_projection = nn.Conv2d(channels, channels, kernel_size=1, bias=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        batch, _, height, width = x.shape
        qkv = self.qkv_dwconv(self.qkv(x))

        query, key, value = qkv.chunk(3, dim=1)
        query = rearrange(query, "b (head c) h w -> b head c (h w)", head=self.num_heads)
        key = rearrange(key, "b (head c) h w -> b head c (h w)", head=self.num_heads)
        value = rearrange(value, "b (head c) h w -> b head c (h w)", head=self.num_heads)

        query = F.normalize(query, dim=-1)
        key = F.normalize(key, dim=-1)
        attention = (query @ key.transpose(-2, -1)) * self.temperature
        attention = attention.softmax(dim=-1)
        output = attention @ value
        output = rearrange(
            output,
            "b head c (h w) -> b (head c) h w",
            head=self.num_heads,
            h=height,
            w=width,
        )
        return self.output_projection(output)


class PyramidRefinedChannelAttention(nn.Module):
    """Three-scale PRCA module used by LPPS-Mamba (paper Eqs. 3-4)."""

    def __init__(
        self,
        channels: int,
        num_heads: int = 4,
        num_scales: int = 3,
        layers_per_scale: int = 3,
        dilation: int = 3,
    ):
        super().__init__()
        self.num_scales = _validate_positive("num_scales", num_scales)
        layers_per_scale = _validate_positive("layers_per_scale", layers_per_scale)
        self.scale_stacks = nn.ModuleList(
            nn.Sequential(
                *[
                    PyramidChannelAttention(channels, num_heads=num_heads, dilation=dilation)
                    for _ in range(layers_per_scale)
                ]
            )
            for _ in range(self.num_scales)
        )
        self.output_projection = nn.Conv2d(
            channels * self.num_scales,
            channels,
            kernel_size=1,
            bias=True,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        height, width = x.shape[-2:]
        refined_scales = []
        for scale_index, attention_stack in enumerate(self.scale_stacks):
            if scale_index == 0:
                scaled = x
            else:
                factor = 2**scale_index
                scaled = F.avg_pool2d(x, kernel_size=factor, stride=factor)
            refined = attention_stack(scaled)
            if scale_index > 0:
                refined = F.interpolate(
                    refined,
                    size=(height, width),
                    mode="bilinear",
                    align_corners=False,
                )
            refined_scales.append(refined)
        return self.output_projection(torch.cat(refined_scales, dim=1))


class LightweightSpatialPrior(nn.Module):
    """Lightweight Spatial Prior (LSP) module from paper Eqs. 1-2."""

    def __init__(self, channels: int, group_norm_groups: int = 4, reduction: int = 4):
        super().__init__()
        hidden_channels = max(channels // reduction, 8)
        self.depthwise_convolution = nn.Conv2d(
            channels,
            channels,
            kernel_size=3,
            padding=1,
            groups=channels,
        )
        self.spatial_gate = nn.Sequential(
            nn.Conv2d(channels, hidden_channels, kernel_size=1),
            nn.SiLU(),
            nn.Conv2d(hidden_channels, 1, kernel_size=1),
            nn.Sigmoid(),
        )
        self.feature_projection = nn.Conv2d(channels, channels, kernel_size=1)
        self.normalisation = nn.GroupNorm(group_norm_groups, channels)
        self.activation = nn.SiLU()

    def forward(self, feature_0: torch.Tensor) -> torch.Tensor:
        local_response = self.depthwise_convolution(feature_0)
        spatial_gate = self.spatial_gate(feature_0)
        local_prior = self.feature_projection(local_response * spatial_gate)
        local_prior = self.activation(self.normalisation(local_prior))
        return local_prior + feature_0


class LPPSMamba(nn.Module):
    """Local-Prior Pyramid Spatial Mamba branch (paper Eqs. 1-5)."""

    def __init__(
        self,
        channels: int,
        group_norm_groups: int = 4,
        prca_num_heads: int = 4,
        prca_num_scales: int = 3,
        prca_layers_per_scale: int = 3,
        pyramid_dilation: int = 3,
        lsp_reduction: int = 4,
        mamba_d_state: int = 16,
        mamba_d_conv: int = 4,
        mamba_expand: int = 2,
    ):
        super().__init__()
        self.lightweight_spatial_prior = LightweightSpatialPrior(
            channels,
            group_norm_groups,
            lsp_reduction,
        )
        self.pyramid_refined_channel_attention = PyramidRefinedChannelAttention(
            channels=channels,
            num_heads=prca_num_heads,
            num_scales=prca_num_scales,
            layers_per_scale=prca_layers_per_scale,
            dilation=pyramid_dilation,
        )
        self.spatial_mamba = Mamba(
            d_model=channels,
            d_state=mamba_d_state,
            d_conv=mamba_d_conv,
            expand=mamba_expand,
        )
        self.output_projection = nn.Sequential(
            nn.GroupNorm(group_norm_groups, channels),
            nn.SiLU(),
        )

    def forward(self, feature_0: torch.Tensor) -> torch.Tensor:
        feature_prior = self.lightweight_spatial_prior(feature_0)
        feature_prca = self.pyramid_refined_channel_attention(feature_prior)
        batch, channels, height, width = feature_prca.shape
        spatial_sequence = feature_prca.permute(0, 2, 3, 1).reshape(
            batch,
            height * width,
            channels,
        )
        spatial_sequence = self.spatial_mamba(spatial_sequence)
        spatial_feature = spatial_sequence.reshape(batch, height, width, channels)
        spatial_feature = spatial_feature.permute(0, 3, 1, 2).contiguous()
        spatial_feature = self.output_projection(spatial_feature)
        return spatial_feature + feature_prior


class DGSMamba(nn.Module):
    """Differential Grouped Spectral Mamba branch (paper Eqs. 6-7)."""

    def __init__(
        self,
        channels: int,
        num_spectral_groups: int = 4,
        group_norm_groups: int = 4,
        difference_alpha: float = 0.5,
        mamba_d_state: int = 16,
        mamba_d_conv: int = 4,
        mamba_expand: int = 2,
    ):
        super().__init__()
        self.num_spectral_groups = _validate_positive(
            "num_spectral_groups",
            num_spectral_groups,
        )
        if channels % num_spectral_groups != 0:
            raise ValueError("channels must be divisible by num_spectral_groups.")
        self.group_width = channels // self.num_spectral_groups
        self.difference_alpha = float(difference_alpha)
        self.grouped_spectral_mamba = Mamba(
            d_model=self.group_width,
            d_state=mamba_d_state,
            d_conv=mamba_d_conv,
            expand=mamba_expand,
        )
        self.output_projection = nn.Sequential(
            nn.GroupNorm(group_norm_groups, channels),
            nn.SiLU(),
        )

    def first_order_channel_difference(self, feature_0: torch.Tensor) -> torch.Tensor:
        difference = torch.zeros_like(feature_0)
        difference[:, :-1] = feature_0[:, 1:] - feature_0[:, :-1]
        return feature_0 + self.difference_alpha * difference

    def forward(self, feature_0: torch.Tensor) -> torch.Tensor:
        feature_diff = self.first_order_channel_difference(feature_0)

        batch, channels, height, width = feature_diff.shape
        grouped_sequence = feature_diff.permute(0, 2, 3, 1).reshape(
            batch * height * width,
            self.num_spectral_groups,
            self.group_width,
        )
        spectral_sequence = self.grouped_spectral_mamba(grouped_sequence)
        spectral_feature = spectral_sequence.reshape(batch, height, width, channels)
        spectral_feature = spectral_feature.permute(0, 3, 1, 2).contiguous()
        spectral_feature = self.output_projection(spectral_feature)
        # The residual is F0, not Fdiff, as stated in paper Eq. 7.
        return spectral_feature + feature_0


class ChannelWiseCompetitiveFusion(nn.Module):
    """Channel-wise competitive fusion from paper Eqs. 8-10."""

    def __init__(self, channels: int):
        super().__init__()
        self.spatial_response = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(1),
            nn.Linear(channels, channels, bias=False),
        )
        self.spectral_response = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(1),
            nn.Linear(channels, channels, bias=False),
        )

    def forward(
        self,
        spatial_feature: torch.Tensor,
        spectral_feature: torch.Tensor,
    ) -> torch.Tensor:
        spatial_logit = self.spatial_response(spatial_feature)
        spectral_logit = self.spectral_response(spectral_feature)
        weights = torch.softmax(
            torch.stack((spatial_logit, spectral_logit), dim=1),
            dim=1,
        )
        spatial_weight = weights[:, 0, :, None, None]
        spectral_weight = weights[:, 1, :, None, None]
        return spatial_weight * spatial_feature + spectral_weight * spectral_feature


class SpatialSpectralDualBranch(nn.Module):
    """Parallel LPPS-Mamba/DGS-Mamba backbone followed by competitive fusion."""

    def __init__(
        self,
        channels: int,
        num_spectral_groups: int = 4,
        group_norm_groups: int = 4,
        difference_alpha: float = 0.5,
        prca_num_heads: int = 4,
        prca_num_scales: int = 3,
        prca_layers_per_scale: int = 3,
        pyramid_dilation: int = 3,
        lsp_reduction: int = 4,
        mamba_d_state: int = 16,
        mamba_d_conv: int = 4,
        mamba_expand: int = 2,
    ):
        super().__init__()
        self.lpps_mamba = LPPSMamba(
            channels=channels,
            group_norm_groups=group_norm_groups,
            prca_num_heads=prca_num_heads,
            prca_num_scales=prca_num_scales,
            prca_layers_per_scale=prca_layers_per_scale,
            pyramid_dilation=pyramid_dilation,
            lsp_reduction=lsp_reduction,
            mamba_d_state=mamba_d_state,
            mamba_d_conv=mamba_d_conv,
            mamba_expand=mamba_expand,
        )
        self.dgs_mamba = DGSMamba(
            channels=channels,
            num_spectral_groups=num_spectral_groups,
            group_norm_groups=group_norm_groups,
            difference_alpha=difference_alpha,
            mamba_d_state=mamba_d_state,
            mamba_d_conv=mamba_d_conv,
            mamba_expand=mamba_expand,
        )
        self.competitive_fusion = ChannelWiseCompetitiveFusion(channels)

    def forward(self, feature_0: torch.Tensor) -> torch.Tensor:
        spatial_feature = self.lpps_mamba(feature_0)
        spectral_feature = self.dgs_mamba(feature_0)
        fused_feature = self.competitive_fusion(spatial_feature, spectral_feature)
        return fused_feature + feature_0


class PyS2CFMamba(nn.Module):
    """Pyramid Spatial-Spectral Competitive Fusion Mamba network, adapted to
    emit a single per-patch label via global average pooling (see module
    docstring)."""

    def __init__(
        self,
        in_channels: int = 30,
        num_classes: int = 18,
        hidden_dim: int = 128,
        num_spectral_groups: int = 4,
        group_norm_groups: int = 4,
        difference_alpha: float = 0.5,
        prca_num_heads: int = 4,
        prca_num_scales: int = 3,
        prca_layers_per_scale: int = 3,
        pyramid_dilation: int = 3,
        lsp_reduction: int = 4,
        mamba_d_state: int = 16,
        mamba_d_conv: int = 4,
        mamba_expand: int = 2,
        pool_size: int = 2,
        classification_head_dim: int = 128,
    ):
        super().__init__()
        for divisor_name, divisor in (
            ("num_spectral_groups", num_spectral_groups),
            ("group_norm_groups", group_norm_groups),
            ("prca_num_heads", prca_num_heads),
        ):
            if hidden_dim % divisor != 0:
                raise ValueError(f"hidden_dim must be divisible by {divisor_name}.")
        if classification_head_dim % group_norm_groups != 0:
            raise ValueError("classification_head_dim must be divisible by group_norm_groups.")

        self.input_projection = nn.Sequential(
            nn.Conv2d(in_channels, hidden_dim, kernel_size=1),
            nn.GroupNorm(group_norm_groups, hidden_dim),
            nn.SiLU(),
        )
        self.spatial_spectral_backbone = SpatialSpectralDualBranch(
            channels=hidden_dim,
            num_spectral_groups=num_spectral_groups,
            group_norm_groups=group_norm_groups,
            difference_alpha=difference_alpha,
            prca_num_heads=prca_num_heads,
            prca_num_scales=prca_num_scales,
            prca_layers_per_scale=prca_layers_per_scale,
            pyramid_dilation=pyramid_dilation,
            lsp_reduction=lsp_reduction,
            mamba_d_state=mamba_d_state,
            mamba_d_conv=mamba_d_conv,
            mamba_expand=mamba_expand,
        )
        self.downsampling = nn.AvgPool2d(kernel_size=pool_size, stride=pool_size)
        self.classification_head = nn.Sequential(
            nn.Conv2d(hidden_dim, classification_head_dim, kernel_size=1),
            nn.GroupNorm(group_norm_groups, classification_head_dim),
            nn.SiLU(),
            nn.Conv2d(classification_head_dim, num_classes, kernel_size=1),
        )
        self.global_pool = nn.AdaptiveAvgPool2d(1)
        self.flatten = nn.Flatten(1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        feature_0 = self.input_projection(x)
        fused_feature = self.spatial_spectral_backbone(feature_0)
        downsampled_feature = self.downsampling(fused_feature)
        logits_map = self.classification_head(downsampled_feature)
        logits = self.global_pool(logits_map)
        return self.flatten(logits)


@register_model('PyS2CFMamba', expects_4d=True, hidden_dim=128)
def pys2cf_mamba_model(pretrained: bool = False, **kwargs) -> PyS2CFMamba:
    """Constructs a PyS2CF-Mamba model."""
    if 'bands' in kwargs:
        kwargs['in_channels'] = kwargs.pop('bands')
    kwargs.pop('patch_size', None)  # PyS2CF-Mamba doesn't use patch_size directly
    return PyS2CFMamba(**kwargs)
