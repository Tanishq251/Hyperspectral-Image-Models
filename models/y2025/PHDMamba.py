import torch
from torch import nn
from torch.nn import functional as F
import numbers
from einops import rearrange, repeat
import collections.abc
from thop import profile
import os
import math
from timm.layers import DropPath
from functools import partial
from typing import Callable
from mamba_ssm.ops.selective_scan_interface import selective_scan_fn
from models.registry import register_model

def _ntuple(n):

    def parse(x):
        if isinstance(x, collections.abc.Iterable):
            return x
        return tuple(repeat(x, n))

    return parse

def to_2tuple(x):
    if isinstance(x, int):
        return (x, x)
    elif hasattr(x, '__iter__'):
        return tuple(x)
    else:
        raise ValueError("Unsupported input type for to_2tuple")

to_1tuple = _ntuple(1)
# to_2tuple = _ntuple(2)
to_3tuple = _ntuple(3)
to_4tuple = _ntuple(4)
to_ntuple = _ntuple

class CrossEntropy2d(nn.Module):
    def __init__(self, size_average=True, ignore_label=255):
        super(CrossEntropy2d, self).__init__()
        self.size_average = size_average
        self.ignore_label = ignore_label

    def forward(self, predict, target, weight=None):
        """
            Args:
                predict:(n, c, h, w)
                target:(n, h, w)
                weight (Tensor, optional): a manual rescaling weight given to each class.
                                           If given, has to be a Tensor of size "nclasses"
        """
        assert not target.requires_grad
        assert predict.dim() == 4
        assert target.dim() == 3
        assert predict.size(0) == target.size(0), "{0} vs {1} ".format(predict.size(0), target.size(0))
        assert predict.size(2) == target.size(1), "{0} vs {1} ".format(predict.size(2), target.size(1))
        assert predict.size(3) == target.size(2), "{0} vs {1} ".format(predict.size(3), target.size(3))
        n, c, h, w = predict.size()
        target_mask = (target >= 0) * (target != self.ignore_label)
        target = target[target_mask]
        if not target.data.dim():
            return torch.zeros(1)
        predict = predict.transpose(1, 2).transpose(2, 3).contiguous()
        predict = predict[target_mask.view(n, h, w, 1).repeat(1, 1, 1, c)].view(-1, c)
        loss = F.cross_entropy(predict, target, weight=weight, size_average=self.size_average)
        return loss

def adjust_learning_rate(optimizer,base_lr, i_iter, max_iter, power=0.9):
    lr = base_lr * ((1 - float(i_iter) / max_iter) ** (power))
    optimizer.param_groups[0]['lr'] = lr
    if len(optimizer.param_groups) > 1:
        optimizer.param_groups[1]['lr'] = lr * 10

class PatchEmbed(nn.Module):
    r""" Image to Patch Embedding

    Args:
        img_size (int): Image size.  Default: 224.
        patch_size (int): Patch token size. Default: 4.
        in_chans (int): Number of input image channels. Default: 3.
        embed_dim (int): Number of linear projection output channels. Default: 96.
        norm_layer (nn.Module, optional): Normalization layer. Default: None
    """

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
        x = x.flatten(2).transpose(1, 2)  # b Ph*Pw c
        if self.norm is not None:
            x = self.norm(x)
        return x, (Hp, Wp)

class LayerNorm(nn.Module):
    def __init__(self, dim, LayerNorm_type):
        super(LayerNorm, self).__init__()
        if LayerNorm_type == 'BiasFree':
            self.body = BiasFree_LayerNorm(dim)
        else:
            self.body = WithBias_LayerNorm(dim)

    def forward(self, x):
        h, w = x.shape[-2:]
        return to_4d(self.body(to_3d(x)), h, w)

def to_3d(x):
    return rearrange(x, 'b c h w -> b (h w) c')

def to_4d(x, h, w):
    return rearrange(x, 'b (h w) c -> b c h w', h=h, w=w)

class BiasFree_LayerNorm(nn.Module):
    def __init__(self, normalized_shape):
        super(BiasFree_LayerNorm, self).__init__()
        if isinstance(normalized_shape, numbers.Integral):
            normalized_shape = (normalized_shape,)
        normalized_shape = torch.Size(normalized_shape)

        assert len(normalized_shape) == 1

        self.weight = nn.Parameter(torch.ones(normalized_shape))
        self.normalized_shape = normalized_shape

    def forward(self, x):
        sigma = x.var(-1, keepdim=True, unbiased=False)
        return x / torch.sqrt(sigma + 1e-5) * self.weight

class WithBias_LayerNorm(nn.Module):
    def __init__(self, normalized_shape):
        super(WithBias_LayerNorm, self).__init__()
        if isinstance(normalized_shape, numbers.Integral):
            normalized_shape = (normalized_shape,)
        normalized_shape = torch.Size(normalized_shape)

        assert len(normalized_shape) == 1

        self.weight = nn.Parameter(torch.ones(normalized_shape))
        self.bias = nn.Parameter(torch.zeros(normalized_shape))
        self.normalized_shape = normalized_shape

    def forward(self, x):
        mu = x.mean(-1, keepdim=True)
        sigma = x.var(-1, keepdim=True, unbiased=False)
        return (x - mu) / torch.sqrt(sigma + 1e-5) * self.weight + self.bias

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

class Mix(nn.Module):
    def __init__(self, m=-0.80):
        super(Mix, self).__init__()
        w = torch.nn.Parameter(torch.FloatTensor([m]), requires_grad=True)
        w = torch.nn.Parameter(w, requires_grad=True)
        self.w = w
        self.mix_block = nn.Sigmoid()

    def forward(self, fea1, fea2):
        mix_factor = self.mix_block(self.w)
        out = fea1 * mix_factor.expand_as(fea1) + fea2 * (1 - mix_factor.expand_as(fea2))
        return out

# FFN with single kernel
class FFN(nn.Module):
    def __init__(self, dim, bias,kernel_size):
        super(FFN, self).__init__()
        if kernel_size not in [3, 5, 7]:
            raise ValueError("Invalid kernel_size. Must be 3, 5, or 7.")

        self.kernel_size = kernel_size
        hidden_features = 180

        self.project_in = nn.Conv2d(dim, hidden_features, kernel_size=1, bias=bias)
        self.dwconv3x3 = nn.Conv2d(hidden_features, hidden_features, kernel_size=3, stride=1, padding=1,groups=hidden_features, bias=bias) #dwconv
        self.dwconv5x5 = nn.Conv2d(hidden_features, hidden_features, kernel_size=5, stride=1, padding=2,
                                   groups=hidden_features, bias=bias)
        self.dwconv7x7 = nn.Conv2d(hidden_features, hidden_features, kernel_size=7, stride=1, padding=3,
                                   groups=hidden_features, bias=bias)
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

class dwconv(nn.Module):
    def __init__(self, hidden_features, kernel_size=5):
        super(dwconv, self).__init__()
        self.depthwise_conv = nn.Sequential(
            nn.Conv2d(hidden_features, hidden_features, kernel_size=kernel_size, stride=1,
                      padding=(kernel_size - 1) // 2, dilation=1,
                      groups=hidden_features), nn.GELU())
        self.hidden_features = hidden_features

    def forward(self, x, x_size):
        x = x.transpose(1, 2).view(x.shape[0], self.hidden_features, x_size[0], x_size[1]).contiguous()  # b Ph*Pw c
        x = self.depthwise_conv(x)
        x = x.flatten(2).transpose(1, 2).contiguous()
        return x


class DynamicPosBias(nn.Module):
    # The implementation builds on Crossformer code https://github.com/cheerss/CrossFormer/blob/main/models/crossformer.py
    """ Dynamic Relative Position Bias.
    Args:
        dim (int): Number of input channels.
        num_heads (int): Number of attention heads.
        residual (bool):  If True, use residual strage to connect conv.
    """

    def __init__(self, dim, num_heads, residual):
        super().__init__()
        self.residual = residual
        self.num_heads = num_heads
        self.pos_dim = dim // 4
        self.pos_proj = nn.Linear(2, self.pos_dim)
        self.pos1 = nn.Sequential(
            nn.LayerNorm(self.pos_dim),
            nn.ReLU(inplace=True),
            nn.Linear(self.pos_dim, self.pos_dim),
        )
        self.pos2 = nn.Sequential(
            nn.LayerNorm(self.pos_dim),
            nn.ReLU(inplace=True),
            nn.Linear(self.pos_dim, self.pos_dim)
        )
        self.pos3 = nn.Sequential(
            nn.LayerNorm(self.pos_dim),
            nn.ReLU(inplace=True),
            nn.Linear(self.pos_dim, self.num_heads)
        )

    def forward(self, biases):
        if self.residual:
            pos = self.pos_proj(biases)  # 2Gh-1 * 2Gw-1, heads
            pos = pos + self.pos1(pos)
            pos = pos + self.pos2(pos)
            pos = self.pos3(pos)
        else:
            pos = self.pos3(self.pos2(self.pos1(self.pos_proj(biases))))
        return pos

class ChannelAttention(nn.Module):
    """Channel attention used in RCAN.
    Args:
        num_feat (int): Channel number of intermediate features.
        squeeze_factor (int): Channel squeeze factor. Default: 16.
    """

    def __init__(self, num_feat, squeeze_factor=16):
        super(ChannelAttention, self).__init__()
        self.attention = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(num_feat, num_feat // squeeze_factor, 1, padding=0),
            nn.ReLU(inplace=True),
            nn.Conv2d(num_feat // squeeze_factor, num_feat, 1, padding=0),
            nn.Sigmoid())

    def forward(self, x):
        y = self.attention(x)
        return x * y

class CAB(nn.Module):
    def __init__(self, num_feat, is_light_sr= False, compress_ratio=4,squeeze_factor=30):
        super(CAB, self).__init__()
        if is_light_sr: # we use depth-wise conv for light-SR to achieve more efficient
            self.cab = nn.Sequential(
                nn.Conv2d(num_feat, num_feat, 3, 1, 1, groups=num_feat),
                ChannelAttention(num_feat, squeeze_factor)
            )
        else: # for classic SR
            self.cab = nn.Sequential(
                nn.Conv2d(num_feat, num_feat // compress_ratio, 3, 1, 1),
                nn.GELU(),
                nn.Conv2d(num_feat // compress_ratio, num_feat, 3, 1, 1),
                ChannelAttention(num_feat, squeeze_factor)
            )

    def forward(self, x):
        return self.cab(x)

class Selective_Scan_Spa(nn.Module):
    def __init__(self, d_model, d_state=16, d_conv=3, expand=2., dt_rank="auto",
                 dt_min=0.001, dt_max=0.1, dt_init="random", dt_scale=1.0, dt_init_floor=1e-4,
                 dropout=0., conv_bias=True, bias=False, device=None, dtype=None, mode='Spa', **kwargs):
        factory_kwargs = {"device": device, "dtype": dtype}
        super().__init__()
        self.d_model = d_model
        self.d_state = d_state
        self.d_conv = d_conv
        self.expand = expand
        self.d_inner = int(self.expand * self.d_model)
        self.dt_rank = math.ceil(self.d_model / 16) if dt_rank == "auto" else dt_rank
        self.mode = mode

        self.in_proj = nn.Linear(self.d_model, self.d_inner * 2, bias=bias, **factory_kwargs)

        if self.mode == 'Spe':
            self.conv = nn.Conv1d(self.d_inner, self.d_inner, kernel_size=d_conv, padding=(d_conv - 1) // 2,
                                  groups=1, bias=conv_bias, **factory_kwargs)
        elif self.mode == 'Spa':
            self.conv = nn.Conv2d(self.d_inner, self.d_inner, kernel_size=d_conv, padding=(d_conv - 1) // 2,
                                  groups=self.d_inner, bias=conv_bias, **factory_kwargs)
        else:
            raise ValueError("mode must be one of ['Spe', 'Spa']")

        self.act = nn.SiLU()

        # projection weights
        self.x_proj = nn.ModuleList([nn.Linear(self.d_inner, self.dt_rank + self.d_state * 2, bias=False, **factory_kwargs) for _ in range(4)])
        self.x_proj_weight = nn.Parameter(torch.stack([t.weight for t in self.x_proj], dim=0))
        del self.x_proj

        self.dt_projs = nn.ModuleList([
            self.dt_init(self.dt_rank, self.d_inner, dt_scale, dt_init, dt_min, dt_max, dt_init_floor, **factory_kwargs)
            for _ in range(4)
        ])
        self.dt_projs_weight = nn.Parameter(torch.stack([t.weight for t in self.dt_projs], dim=0))
        self.dt_projs_bias = nn.Parameter(torch.stack([t.bias for t in self.dt_projs], dim=0))
        del self.dt_projs

        self.A_logs = self.A_log_init(self.d_state, self.d_inner, copies=4, merge=True)
        self.Ds = self.D_init(self.d_inner, copies=4, merge=True)

        self.selective_scan = selective_scan_fn  # You should define or import this externally

        self.out_norm = nn.LayerNorm(self.d_inner)
        self.out_proj = nn.Linear(self.d_inner, self.d_model, bias=bias, **factory_kwargs)
        self.dropout = nn.Dropout(dropout) if dropout > 0. else None

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

        dt = torch.exp(torch.rand(d_inner, **factory_kwargs) * (math.log(dt_max) - math.log(dt_min)) + math.log(dt_min)).clamp(min=dt_init_floor)
        inv_dt = dt + torch.log(-torch.expm1(-dt))
        with torch.no_grad():
            dt_proj.bias.copy_(inv_dt)
        dt_proj.bias._no_reinit = True
        return dt_proj

    @staticmethod
    def A_log_init(d_state, d_inner, copies=1, device=None, merge=True):
        A = repeat(torch.arange(1, d_state + 1, dtype=torch.float32, device=device), "n -> d n", d=d_inner).contiguous()
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

    def forward_core(self, x: torch.Tensor):
        B, C, H, W = x.shape
        L = H * W
        K = 4

        # mode-aware enhancement
        if self.mode == 'Spe':
            x_spectral = x.view(B, C, L)
            x_spectral = self.act(self.conv(x_spectral))
            x = x_spectral.view(B, C, H, W)
        elif self.mode == 'Spa':
            x = self.act(self.conv(x))

        x_hwwh = torch.stack([x.view(B, -1, L), torch.transpose(x, dim0=2, dim1=3).contiguous().view(B, -1, L)], dim=1)
        x_hwwh = x_hwwh.view(B, 2, -1, L)
        xs = torch.cat([x_hwwh, torch.flip(x_hwwh, dims=[-1])], dim=1)

        x_dbl = torch.einsum("b k d l, k c d -> b k c l", xs.view(B, K, -1, L), self.x_proj_weight)
        dts, Bs, Cs = torch.split(x_dbl, [self.dt_rank, self.d_state, self.d_state], dim=2)
        dts = torch.einsum("b k r l, k d r -> b k d l", dts.view(B, K, -1, L), self.dt_projs_weight)

        xs = xs.float().view(B, -1, L)
        dts = dts.contiguous().float().view(B, -1, L)
        Bs = Bs.float().view(B, K, -1, L)
        Cs = Cs.float().view(B, K, -1, L)
        Ds = self.Ds.float().view(-1)
        As = -torch.exp(self.A_logs.float()).view(-1, self.d_state)
        dt_projs_bias = self.dt_projs_bias.float().view(-1)

        out_y = self.selective_scan(
            xs, dts, As, Bs, Cs, Ds, z=None,
            delta_bias=dt_projs_bias,
            delta_softplus=True,
            return_last_state=False,
        ).view(B, K, -1, L)

        inv_y = torch.flip(out_y[:, 2:4], dims=[-1]).view(B, 2, -1, L)
        wh_y = torch.transpose(out_y[:, 1].view(B, -1, W, H), dim0=2, dim1=3).contiguous().view(B, -1, L)
        invwh_y = torch.transpose(inv_y[:, 1].view(B, -1, W, H), dim0=2, dim1=3).contiguous().view(B, -1, L)

        return out_y[:, 0], inv_y[:, 0], wh_y, invwh_y

    def forward(self, x: torch.Tensor, **kwargs):
        B, H, W, C = x.shape
        xz = self.in_proj(x)
        x, z = xz.chunk(2, dim=-1)

        x = x.permute(0, 3, 1, 2).contiguous()  # to (B, C, H, W)
        y1, y2, y3, y4 = self.forward_core(x)
        y = y1 + y2 + y3 + y4

        y = torch.transpose(y, dim0=1, dim1=2).contiguous().view(B, H, W, -1)
        y = self.out_norm(y)
        y = y * F.silu(z)
        out = self.out_proj(y)
        if self.dropout is not None:
            out = self.dropout(out)
        return out

class VSSBlock_Spa(nn.Module):
    def __init__(
            self,
            hidden_dim: int = 0,
            drop_path: float = 0,
            norm_layer: Callable[..., torch.nn.Module] = partial(nn.LayerNorm, eps=1e-6),
            attn_drop_rate: float = 0,
            d_state: int = 16,
            mlp_ratio: float = 2.,
            mode= 'Spa',
            **kwargs,
    ):
        super().__init__()
        self.ln_1 = norm_layer(hidden_dim)
        self.self_attention = Selective_Scan_Spa(d_model=hidden_dim, d_state=d_state,expand=mlp_ratio,dropout=attn_drop_rate,mode=mode, **kwargs)  # 2D-SSM 'Spa' or 'Spe'

        self.drop_path = DropPath(drop_path)
        self.skip_scale= nn.Parameter(torch.ones(hidden_dim))

        self.conv = nn.Conv2d(hidden_dim,hidden_dim,3,1,1)

        self.proj = nn.Sequential(
            nn.GroupNorm(4, hidden_dim),
            nn.SiLU()
        )
        self.ln_2 = nn.LayerNorm(hidden_dim)
        self.skip_scale2 = nn.Parameter(torch.ones(hidden_dim))
        self.dwconv = nn.Sequential(
            nn.Conv2d(hidden_dim, hidden_dim, kernel_size=3, stride=1, padding=1, groups=hidden_dim),
            nn.BatchNorm2d(hidden_dim),
            nn.GELU()
        )
        self.channel_interaction = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(hidden_dim, hidden_dim // 8, kernel_size=1),
            # nn.BatchNorm2d(dim // 8),
            nn.GELU(),
            nn.Conv2d(hidden_dim // 8, hidden_dim, kernel_size=1),
        )
        self.spatial_interaction = nn.Sequential(
            nn.Conv2d(hidden_dim, hidden_dim // 16, kernel_size=1),
            # nn.BatchNorm2d(dim // 16),
            nn.GELU(),
            nn.Conv2d(hidden_dim // 16, 1, kernel_size=1)
        )
        self.ca = CAB(num_feat=hidden_dim)

    def forward(self, input):
        B, C, H, W = input.shape # B,C,H,W
        L = H * W
        x = input

        # part one
        conv_x = self.ca(self.dwconv(x)) # local channel branch # B,C,H,W

        # Spa-VSSM
        x1 = x.permute(0,2,3,1).contiguous()  # B,H,W,C
        attened_x = self.self_attention(self.ln_1(x1))  # B,H,W,C spatial branch
        attened_x = attened_x.permute(0,3,1,2).contiguous() # B,C,H,W

        # S-Map (before sigmoid)
        spatial_map = self.spatial_interaction(attened_x)  # 1,1,64,64  H*W

        # C-Map (before sigmoid)
        channel_map = self.channel_interaction(conv_x).permute(0, 2, 3, 1).contiguous().view(B, 1, C)  # 1,1,180  1*C

        # C-I
        attened_x = attened_x.permute(0, 2, 3, 1).contiguous().view(B, L, C)  # B,L,C
        attened_x = attened_x * torch.sigmoid(channel_map)  # [1,4096,180] * [1,1,180] -> [1,4096,180]

        # S-I
        conv_x = torch.sigmoid(spatial_map) * conv_x  # [1,1,64,64]  * [1,180,64,64] ->  [1,180,64,64]
        conv_x = conv_x.permute(0, 2, 3, 1).contiguous().view(B, L, C)  # 1,4096,180

        x = (x1 * self.skip_scale).view(B,L,C).contiguous() + attened_x + conv_x   # B,L,C
        x = x.view(B,H,W,C).contiguous()   #B,H,W,C

        # part two
        x = x * self.skip_scale2 + self.conv(self.ln_2(x).permute(0, 3, 1, 2).contiguous()).permute(0, 2, 3,
                                                                                                    1).contiguous()  # B,H,W,C
        x = x.permute(0, 3, 1, 2).contiguous()  # B,C,H,W

        return x




class PHDMamba(nn.Module):
    def __init__(self,
                 bands=103,
                 embed_dim=64,
                 img_size=128,
                 patch_size=1,
                 norm_layer=nn.LayerNorm,
                 depth=4,
                 drop_rate=0.,
                 num_classes=9,
                 group_num=4,
                 patch_norm=True,
                 **kwargs
                 ):
        super(PHDMamba, self).__init__()

        self.patch_norm = patch_norm
        self.conv0 = nn.Conv2d(bands, embed_dim, 1)
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
            nn.Conv2d(in_channels=bands, out_channels=embed_dim, kernel_size=1, stride=1, padding=0),
            nn.GroupNorm(group_num, embed_dim),
            nn.SiLU())

        self.phdmambablock_1 = nn.Sequential(
            VSSBlock_Spa(hidden_dim=embed_dim, drop_path=0., norm_layer=nn.LayerNorm, mlp_ratio=2., d_state=16, mode='Spa'),
        )
        self.conv_1 = nn.AvgPool2d(kernel_size=2, stride=2, padding=0)
        self.phdmambablock_2 = nn.Sequential(
            VSSBlock_Spa(hidden_dim=embed_dim, drop_path=0., norm_layer=nn.LayerNorm, mlp_ratio=2., d_state=16,
                         mode='Spa'),
        )
        self.conv_2 = nn.AvgPool2d(kernel_size=2, stride=2, padding=0)
        self.phdmambablock_3 = nn.Sequential(
            VSSBlock_Spa(hidden_dim=embed_dim, drop_path=0., norm_layer=nn.LayerNorm, mlp_ratio=2., d_state=16,
                         mode='Spa'),
        )

        self.cls_head = nn.Sequential(nn.Conv2d(in_channels=embed_dim, out_channels=128, kernel_size=1, stride=1, padding=0),
                                      nn.GroupNorm(group_num,128),
                                      nn.SiLU(),
                                      nn.Conv2d(in_channels=128,out_channels=num_classes,kernel_size=1,stride=1,padding=0))

    def forward(self, x):
        interpolation = nn.UpsamplingBilinear2d(size=x.shape[2:4]) #b,c,h,w
        x = self.patch_embedding(x) # b,c,h,w
        _,_,H,W = x.size()
        layer1 = self.phdmambablock_1(x)  # 256,256
        layer2 = self.phdmambablock_2(self.conv_1(layer1)) # 128，128
        layer3 = self.phdmambablock_3(self.conv_2(layer2))  # 64，64

        x = interpolation(layer3)
        x = self.cls_head(x)
        
        # Global average pooling to convert spatial maps to class predictions
        x = torch.nn.functional.adaptive_avg_pool2d(x, 1)
        x = x.view(x.size(0), -1)

        return x


# if __name__ == '__main__':
#     os.environ["CUDA_VISIBLE_DEVICES"] = "3"
#     input = torch.rand(1,103,256,256).cuda()
#     model = PHDMamba(patch_size=1).cuda()
#     output= model(input)
#     print(output.size())

#     flops, params = profile(model, inputs=(input,))
#     print('Param:{} K' .format(params/1e3))
#     print('Flops:{} G' .format(flops/1e9))  ## 打印计算量


# Factory function for model registration
@register_model('PHDMamba', expects_4d=True, embed_dim=64, group_num=4, patch_norm=True)
def phdmamba(pretrained: bool = False, **kwargs) -> PHDMamba:
    """Constructs a PHDMamba model."""
    # Map standardized names to model-specific names
    if 'bands' in kwargs:
        kwargs['bands'] = kwargs.pop('bands')
    # num_classes is passed through as-is
    kwargs.pop('patch_size', None)  # PHDMamba doesn't use patch_size
    return PHDMamba(**kwargs)
