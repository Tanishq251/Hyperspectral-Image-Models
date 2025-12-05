"""
GSCViT: Hyperspectral Image Classification Using Groupwise Separable Convolutional Vision Transformer Network

Paper: https://ieeexplore.ieee.org/document/10472541
GitHub: https://github.com/flyzzie/TGRS-GSC-VIT
Venue: IEEE Transactions on Geoscience and Remote Sensing (TGRS)
Year: 2024
"""

import random
from functools import partial
import torch
from torch import nn, einsum
from einops import rearrange, repeat
from einops.layers.torch import Rearrange, Reduce
from thop import profile
from timm.models.vision_transformer import _cfg
from torchsummaryX import summary
from .registry import register_model

def cast_tuple(val, length=1):
    return val if isinstance(val, tuple) else ((val,) * length)


# 通道校准策略1
class ChannelAdjustmentLayer1(nn.Module):
    def __init__(self, target_channels=256):
        super(ChannelAdjustmentLayer1, self).__init__()
        self.target_channels = target_channels

    def forward(self, x):
        B, C, H, W = x.size()

        if C == self.target_channels:
            return x

        if C < self.target_channels:
            # 逐个通道复制，放到被复制通道的后面
            num_channels_to_copy = self.target_channels - C
            for i in range(num_channels_to_copy):
                channel_to_copy = torch.randint(0, C, (1,))
                x = torch.cat([x, x[:, channel_to_copy, :, :]], dim=1)

        else:
            # 逐个通道删除
            num_channels_to_remove = C - self.target_channels
            for i in range(num_channels_to_remove):
                rand=torch.randint(0,x.shape[1],(1,))
                x = torch.cat([x[:, :rand, :, :], x[:, rand+1:, :, :]], dim=1)
        return x


# 通道校准策略2
class ChannelAdjustmentLayer2(nn.Module):
    def __init__(self, target_channels=256):
        super(ChannelAdjustmentLayer2, self).__init__()
        self.target_channels = target_channels

    def forward(self, x):
        B, C, H, W = x.size()

        if C == self.target_channels:
            return x

        if C < self.target_channels:
            # 计算需要扩展的通道数
            num_channels_to_expand = self.target_channels - C
            # 计算每一端需要扩展的通道数
            channels_to_expand_per_side = num_channels_to_expand // 2

            # 两端均匀镜像扩展
            x = torch.cat([x[:, :channels_to_expand_per_side+1, :, :].flip(dims=(1,)),
                           x,
                           x[:, -channels_to_expand_per_side:, :, :].flip(dims=(1,))], dim=1)

        else:
            # 计算需要删除的通道数
            num_channels_to_remove = C - self.target_channels
            # 计算每一端需要删除的通道数
            channels_to_remove_per_side = num_channels_to_remove // 2

            # 两端均匀镜像删除
            x = x[:, channels_to_remove_per_side:-channels_to_remove_per_side, :, :]

        return x[:, :self.target_channels, :, :]


# 通道正则化
class ChanLayerNorm(nn.Module):
    def __init__(self, dim, eps=1e-5):
        super().__init__()
        self.eps = eps
        self.g = nn.Parameter(torch.ones(1, dim, 1, 1))
        self.b = nn.Parameter(torch.zeros(1, dim, 1, 1))

    def forward(self, x):
        var = torch.var(x, dim=1, unbiased=False, keepdim=True)
        mean = torch.mean(x, dim=1, keepdim=True)
        return (x - mean) / (var + self.eps).sqrt() * self.g + self.b


class PreNorm(nn.Module):
    def __init__(self, dim, fn):
        super().__init__()
        self.norm = ChanLayerNorm(dim)
        self.fn = fn

    def forward(self, x):
        return self.fn(self.norm(x))


# 通道校准模块
class SpectralCalibration(nn.Module):
    def __init__(self, dim_in, dim_out):
        super().__init__()
        self.conv = nn.Conv2d(dim_in, dim_out, 1)
        self.bn = nn.BatchNorm2d(dim_out)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        x = self.relu(x)
        return x


class GSC(nn.Module):
    def __init__(self, dim_in, dim_out, padding=1, num_groups=8):
        super().__init__()
        self.gpwc = nn.Conv2d(dim_in, dim_out, groups=num_groups, kernel_size=1)
        self.gc = nn.Conv2d(dim_out, dim_out, kernel_size=3, groups=num_groups, padding=padding, stride=1)
        self.bn = nn.BatchNorm2d(dim_out)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        return self.relu(self.bn(self.gc(self.gpwc(x))))


class GSSA(nn.Module):
    def __init__(
            self,
            dim,
            heads=8,
            dim_head=16,
            dropout=0.,
            group_spatial_size=3
    ):
        super().__init__()
        self.heads = heads
        self.scale = dim_head ** -0.5
        self.group_spatial_size = group_spatial_size
        inner_dim = dim_head * heads

        self.attend = nn.Sequential(
            nn.Softmax(dim=-1),
            nn.Dropout(dropout)
        )

        self.to_qkv = nn.Conv1d(dim, inner_dim * 3, 1, bias=False)

        self.group_tokens = nn.Parameter(torch.randn(dim))

        self.group_tokens_to_qk = nn.Sequential(
            nn.LayerNorm(dim_head),
            nn.GELU(),
            Rearrange('b h n c -> b (h c) n'),
            nn.Conv1d(inner_dim, inner_dim * 2, 1),
            Rearrange('b (h c) n -> b h n c', h=heads),
        )

        self.group_attend = nn.Sequential(
            nn.Softmax(dim=-1),
            nn.Dropout(dropout)
        )

        self.to_out = nn.Sequential(
            nn.Conv2d(inner_dim, dim, 1),
            nn.Dropout(dropout)
        )

    def forward(self, x):

        batch, height, width, heads, gss = x.shape[0], *x.shape[-2:], self.heads, self.group_spatial_size
        
        # Adjust group spatial size if dimensions are not divisible
        if (height % gss) != 0 or (width % gss) != 0:
            gss = min(height, width)
            while gss > 1 and ((height % gss) != 0 or (width % gss) != 0):
                gss -= 1
            self.group_spatial_size = gss
        
        num_groups = (height // gss) * (width // gss)

        x = rearrange(x, 'b c (h g1) (w g2) -> (b h w) c (g1 g2)', g1=gss, g2=gss)

        w = repeat(self.group_tokens, 'c -> b c 1', b=x.shape[0])

        x = torch.cat((w, x), dim=-1)

        q, k, v = self.to_qkv(x).chunk(3, dim=1)

        q, k, v = map(lambda t: rearrange(t, 'b (h d) ... -> b h (...) d', h=heads), (q, k, v))

        q = q * self.scale

        dots = einsum('b h i d, b h j d -> b h i j', q, k)

        attn = self.attend(dots)

        out = torch.matmul(attn, v)

        group_tokens, grouped_fmaps = out[:, :, 0], out[:, :, 1:]

        if num_groups == 1:
            fmap = rearrange(grouped_fmaps, '(b x y) h (g1 g2) d -> b (h d) (x g1) (y g2)', x=height // gss,
                             y=width // gss, g1=gss, g2=gss)
            return self.to_out(fmap)

        group_tokens = rearrange(group_tokens, '(b x y) h d -> b h (x y) d', x=height // gss, y=width // gss)

        grouped_fmaps = rearrange(grouped_fmaps, '(b x y) h n d -> b h (x y) n d', x=height // gss, y=width // gss)

        w_q, w_k = self.group_tokens_to_qk(group_tokens).chunk(2, dim=-1)

        w_q = w_q * self.scale

        w_dots = einsum('b h i d, b h j d -> b h i j', w_q, w_k)

        w_attn = self.group_attend(w_dots)

        aggregated_grouped_fmap = einsum('b h i j, b h j w d -> b h i w d', w_attn, grouped_fmaps)

        fmap = rearrange(aggregated_grouped_fmap, 'b h (x y) (g1 g2) d -> b (h d) (x g1) (y g2)', x=height // gss,
                         y=width // gss, g1=gss, g2=gss)
        return self.to_out(fmap)


class Transformer(nn.Module):
    def __init__(
            self,
            dim,
            depth,
            dim_head=16,
            heads=8,
            dropout=0.,
            norm_output=True,
            groupsize=4
    ):
        super().__init__()
        self.layers = nn.ModuleList([])

        for ind in range(depth):
            self.layers.append(
                PreNorm(dim, GSSA(dim, group_spatial_size=groupsize, heads=heads, dim_head=dim_head, dropout=dropout))
            )

        self.norm = ChanLayerNorm(dim) if norm_output else nn.Identity()

    def forward(self, x):
        for attn in self.layers:
            x = attn(x)

        return self.norm(x)


class GSCViT(nn.Module):
    """
    Group-wise Separable Convolution Vision Transformer for HSI Classification
    
    ⚠️ IMPORTANT: This model ONLY supports 8x8 patch size!
    
    The architecture uses group_spatial_size=[4,4,4] which requires
    spatial dimensions to be divisible by 4 at each stage.
    Only 8x8 patches work correctly with this design.
    
    Args:
        num_classes: Number of output classes
        depth: Depth of transformer blocks per stage
        heads: Number of attention heads per stage
        group_spatial_size: Spatial grouping size (must be [4,4,4] for 8x8)
        channels: Number of input spectral channels
        dropout: Dropout rate
        padding: Padding for convolutions
        dims: Feature dimensions per stage
        num_groups: Number of groups for group convolutions
    
    Input shape: [B, 1, C, 8, 8] where C is spectral channels
    Output shape: [B, num_classes]
    """
    def __init__(
            self,
            *,
            num_classes,
            depth,
            heads,
            group_spatial_size,
            channels=200,
            dropout=0.1,
            padding,
            dims=(256, 128, 64, 32),
            num_groups=[16,16,16]
    ):
        super().__init__()
        
        # CRITICAL: Validate group_spatial_size for 8x8 patch requirement
        assert group_spatial_size == [4, 4, 4], (
            f"GSCViT requires group_spatial_size=[4, 4, 4] for 8x8 patches! "
            f"Got: {group_spatial_size}. "
            f"This model ONLY works with 8x8 spatial dimensions. "
            f"Set patch_size=8 in your config."
        )
        
        num_stages = len(depth)

        dim_pairs = tuple(zip(dims[:-1], dims[1:]))
        hyperparams_per_stage = [heads]
        hyperparams_per_stage = list(map(partial(cast_tuple, length=num_stages), hyperparams_per_stage))
        assert all(tuple(map(lambda arr: len(arr) == num_stages, hyperparams_per_stage)))
        self.sc = SpectralCalibration(channels, 256)
        self.bn_1 = nn.BatchNorm2d(256)
        self.relu_1 = nn.ReLU(inplace=True)

        self.layers_trans = nn.ModuleList([])
        for ind, ((layer_dim_in, layer_dim), layer_depth, layer_heads, p, num_group) in enumerate(
                zip(dim_pairs, depth, *hyperparams_per_stage, padding, num_groups )):
            is_last = ind == (num_stages - 1)

            self.layers_trans.append(nn.ModuleList([
                GSC(layer_dim_in, layer_dim, p, num_group),
                Transformer(dim=int(layer_dim), depth=layer_depth, heads=layer_heads,
                            groupsize=group_spatial_size[ind], dropout=dropout, norm_output=not is_last),
                nn.BatchNorm2d(layer_dim),
                nn.ReLU(inplace=True),
                nn.Conv2d(layer_dim,layer_dim,1)
            ]))

        self.conv_last = nn.Conv2d(dims[-1], 2 * dims[-1], 3)

        self.mlp_head = nn.Sequential(
            Reduce('b d h w -> b d', 'mean'),
            nn.LayerNorm(dims[-1]),
            nn.Linear(dims[-1], num_classes)
        )

    def forward(self, x):
        x = x.squeeze(dim=1)
        
        # GSCViT only supports 8x8 patch size due to architecture constraints
        _, _, H, W = x.shape
        assert H == 8 and W == 8, (
            f"GSCViT only supports 8x8 patch size! "
            f"Got input spatial dimensions: {H}x{W}. "
            f"Please set patch_size=8 in your config when using GSCViT."
        )
        
        x = self.sc(x)
        x = self.bn_1(x)
        x = self.relu_1(x)
        for peg, transformer, bn, relu, pw in self.layers_trans:
            x = peg(x)
            y = x
            x = transformer(x)
            x = pw(x) + y
            x = bn(x)
            x = relu(x)
        return self.mlp_head(x)


@register_model('GSCViT', heads=(1, 1, 1), depth=(1, 1, 1), group_spatial_size=[4, 4, 4], dropout=0.1, padding=[1, 1, 1], dims=(256, 128, 64), num_groups=[16, 16, 16])
def gscvit(pretrained: bool = False, **kwargs) -> GSCViT:
    """
    Constructs a GSCViT model.
    
    ⚠️ WARNING: GSCViT ONLY supports 8x8 patch size!
    Set patch_size=8 in your config when using this model.
    """
    # Map standardized names to model-specific names
    if 'bands' in kwargs:
        kwargs['channels'] = kwargs.pop('bands')
    
    # Validate patch_size - GSCViT only works with 8x8
    patch_size = kwargs.pop('patch_size', 8)
    if patch_size != 8:
        raise ValueError(
            f"\n{'='*60}\n"
            f"🚫 GSCViT Patch Size Error\n"
            f"{'='*60}\n"
            f"GSCViT only supports 8x8 patch size due to its architecture.\n"
            f"You provided: patch_size={patch_size}\n\n"
            f"To fix this, either:\n"
            f"  1. Set patch_size=8 in your config\n"
            f"  2. Remove GSCViT from your model list\n"
            f"{'='*60}\n"
        )
    
    return GSCViT(**kwargs)


def gscvit_dataset(dataset):
    model = None
    if dataset == 'sa':
        model = GSCViT(
            num_classes=16,
            channels=204,
            heads=(1, 1, 1),
            depth=(1, 1, 1),
            group_spatial_size=[4, 4, 4],
            dropout=0.1,
            padding=[1, 1, 1],
            dims = (256, 128, 64),
            num_groups=[16, 16, 16]
        )
    elif dataset == 'pu':
        model = GSCViT(
            num_classes=9,
            channels=103,
            heads=(1, 1, 1),
            depth=(1, 1, 1),
            group_spatial_size=[4, 4, 4],
            dropout=0.1,
            padding=[1, 1, 1],
            dims = (256, 128, 64),
            num_groups=[16, 16, 16]
        )
    elif dataset == 'whulk':
        model = GSCViT(
            num_classes=9,
            channels=270,
            heads=(4, 4, 4),
            depth=(1, 1, 1),
            group_spatial_size=[4, 4, 4],
            dropout=0.1,
            padding=[1, 1, 1],
            dims = (256, 128, 64),
            num_groups=[16, 16, 16]
        )
    elif dataset == 'hrl':
        model = GSCViT(
            num_classes=14,
            channels=176,
            heads=(1, 1, 1),
            depth=(1, 1, 1),
            group_spatial_size=[4, 4, 4],
            dropout=0.1,
            padding=[1, 1, 1],
            dims = (256, 128, 64),
            num_groups=[16, 16, 16]
        )
    elif dataset == 'flt':
        model = GSCViT(
            num_classes=10,
            channels=80,
            heads=(1, 1, 1),
            depth=(1, 1, 1),
            group_spatial_size=[4, 4, 4],
            dropout=0.1,
            padding=[1, 1, 1],
            dims = (256, 128, 64),
            num_groups=[16, 16, 16]
        )
    elif dataset == 'ksc':
        model = GSCViT(
            num_classes=13,
            channels=176,
            heads=(1, 1, 1),
            depth=(1, 1, 1),
            group_spatial_size=[4, 4, 4],
            dropout=0.1,
            padding=[1, 1, 1],
            dims = (256, 128, 64),
            num_groups=[16, 16, 16]
        )
    elif dataset == 'ip':
        model = GSCViT(
            num_classes=16,
            channels=200,
            heads=(16, 16, 16),
            depth=(1, 1, 1),
            group_spatial_size=[4, 4,4],
            dropout=0.1,
            padding=[1, 1, 1],
            dims = (256, 128, 64),
            num_groups=[8, 8, 8]
        )
    elif dataset == 'hus':
        model = GSCViT(
            num_classes=15,
            channels=144,
            heads=(1, 1, 1),
            depth=(1, 1, 1),
            group_spatial_size=[4, 4, 4],
            dropout=0.1,
            padding=[1, 1, 1],
            dims = (256, 128, 64),
            num_groups=[16, 16, 16]
        )
    elif dataset == 'MUUFL':
        model = GSCViT(
            num_classes=11,
            channels=64,
            heads=(1, 1, 1),
            depth=(1, 1, 1),
            group_spatial_size=[4, 4, 4],
            dropout=0.1,
            padding=[1, 1, 1],
            dims = (256, 128, 64),
            num_groups=[16, 16, 16]
        )
    elif dataset == 'Trento':
        model = GSCViT(
            num_classes=6,
            channels=63,
            heads=(1, 1, 1),
            depth=(1, 1, 1),
            group_spatial_size=[4, 4, 4],
            dropout=0.1,
            padding=[1, 1, 1],
            dims = (256, 128, 64),
            num_groups=[16, 16, 16]
        )
    elif dataset == 'botswana':
        model = GSCViT(
            num_classes=14,
            channels=145,
            heads=(1, 1, 1),
            depth=(1, 1, 1),
            group_spatial_size=[4, 4, 4],
            dropout=0.1,
            padding=[1, 1, 1],
            dims = (256, 128, 64),
            num_groups=[8, 8, 8]
        )
    return model



if __name__ == '__main__':
    img = torch.randn(1, 270, 8, 8)
    print("input shape:", img.shape)
    net = gscvit_dataset(dataset='whulk')
    net.default_cfg = _cfg()
    print("output shape:", net(img).shape)
    summary(net, torch.zeros((1, 1, 270, 8, 8)))
    flops, params = profile(net, inputs=(img,))
    print('params', params)
    print('flops', flops)  ## 打印计算量
