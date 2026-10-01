from models.registry import register_model
"""
MambaLG: Hyperspectral Image Classification with Mamba (Local-Global Mamba)

Paper: https://doi.org/10.1109/TGRS.2024.3521411
GitHub: https://github.com/danfenghong/IEEE_TGRS_MambaLG
Venue: IEEE Transactions on Geoscience and Remote Sensing (TGRS)
Year: 2024

Adapted from the authors' PyTorch implementation (mambaLG.py). Kept the
architecture (multi-scale spatial conv fusion -> local windowed Mamba ->
global Mamba over the whole spatial token sequence -> spectral-band-group
Mamba -> spatial x spectral gating -> classification head) faithful; the
authors' `Mamba` block (mamba_simple.py, a local copy of mamba_ssm's Mamba)
is replaced with `mamba_ssm.Mamba` directly (same architecture, already a
framework dependency). Hardcoded `.cuda()`/`self.device` calls in the
original `forward` were replaced with device-agnostic tensor ops so the model
runs on CPU or GPU. `spa_token` is exposed as a constructor kwarg (the
authors hardcode 12 in their `main.py`).
"""

import math
import torch
import torch.nn as nn
from einops import rearrange
from mamba_ssm import Mamba


def split_band(x, move_num, spec_num):
    """x: (B, C, H, W) -> (B, N, C, H, W), sliding windows over the channel dim."""
    b, c, h, w = x.shape
    slices = []
    for i in range(0, c, move_num):
        if i + spec_num > c:
            sl = x[:, c - spec_num:c, :, :]
        else:
            sl = x[:, i:i + spec_num, :, :]
        slices.append(sl)
    return torch.stack(slices, dim=1)


class Residual_SSMN(nn.Module):
    def __init__(self, fn):
        super().__init__()
        self.fn = fn

    def forward(self, x, **kwargs):
        return self.fn(x, **kwargs) + x


class LayerNormalize_SSMN(nn.Module):
    def __init__(self, dim, fn):
        super().__init__()
        self.norm = nn.LayerNorm(dim)
        self.fn = fn

    def forward(self, x, **kwargs):
        return self.fn(self.norm(x), **kwargs)


class mamba_block(nn.Module):
    def __init__(self, dim, depth):
        super().__init__()
        self.layers = nn.ModuleList([
            Residual_SSMN(LayerNormalize_SSMN(dim, Mamba(
                d_model=dim,
                d_state=64,
                d_conv=4,
                expand=2,
                use_fast_path=False,
            )))
            for _ in range(depth)
        ])

    def forward(self, x):
        for attention in self.layers:
            x = attention(x)
        return x


class MambaLG(nn.Module):
    def __init__(self, num_classes=16, dim=64, depth=1, dropout=0.1, in_channels=30,
                 spec_num=12, spec_rate=0.5, spa_token=4, patch_size=11, **kwargs):
        super().__init__()
        band = in_channels
        self.spec_num = spec_num
        self.spec_rate = spec_rate
        self.move_num = int(math.ceil(self.spec_num * self.spec_rate))
        self.spa_token = spa_token

        self.preprocess = nn.Sequential(
            nn.Conv2d(in_channels=band, out_channels=dim, kernel_size=(1, 1)),
            nn.BatchNorm2d(dim),
            nn.GELU(),
        )

        self.conv2d_features1 = nn.Sequential(
            nn.Conv2d(in_channels=dim, out_channels=dim, kernel_size=(3, 3), padding=(1, 1)),
            nn.BatchNorm2d(dim),
            nn.GELU(),
        )
        self.conv2d_features2 = nn.Sequential(
            nn.AvgPool2d(kernel_size=5, stride=1, padding=2),
            nn.BatchNorm2d(dim),
            nn.GELU(),
        )
        self.conv2d_channel = nn.Sequential(
            nn.Conv2d(dim, out_channels=dim, kernel_size=(1, 1)),
            nn.BatchNorm2d(dim),
            nn.GELU(),
        )
        self.conv2d_fusion = nn.Sequential(
            nn.Conv2d(4 * dim, out_channels=dim, kernel_size=(1, 1)),
            nn.GELU(),
        )
        self.dropout = nn.Dropout(dropout)

        self.SPAM = mamba_block(dim, depth)
        self.localSPAM = mamba_block(dim, depth)

        spe_dim = dim
        self.nn1 = nn.Sequential(
            nn.Linear(dim, spe_dim),
            nn.LayerNorm(spe_dim),
            nn.GELU(),
        )
        dim = spe_dim

        num_patch = math.floor((dim - (self.spec_num - self.move_num)) / self.move_num) + \
            math.ceil((((dim - (self.spec_num - self.move_num)) % self.move_num) +
                       (self.spec_num - self.move_num)) / self.move_num)

        self.spe_token1 = nn.Sequential(
            nn.Conv3d(1, 1, (1, 1, 7), stride=(1, 1, 1), padding=(0, 0, 3)),
            nn.LayerNorm(dim),
            nn.GELU(),
        )
        self.spe_token2 = nn.Sequential(
            nn.Conv3d(1, 1, (1, 1, 3), stride=(1, 1, 1), padding=(0, 0, 1)),
            nn.LayerNorm(dim),
            nn.GELU(),
        )

        self.SPEM = mamba_block(self.spec_num, depth)

        self.nn2 = nn.Sequential(
            nn.Linear(self.spec_num * num_patch, dim),
            nn.LayerNorm(dim),
            nn.GELU(),
        )
        self.outhead = nn.Sequential(
            nn.AvgPool2d(kernel_size=3, stride=1, padding=1),
            nn.Conv2d(dim, num_classes, 1, 1, 0),
        )

    def forward(self, x):
        # x: (B, C, H, W) -> (B, H, W, C) to match the authors' NHWC-style forward
        x = x.permute(0, 2, 3, 1).contiguous()
        B, H, W, C = x.shape
        x = self.preprocess(x.permute(0, 3, 1, 2)).permute(0, 2, 3, 1)

        x_spe = x
        x = x.permute(0, 3, 1, 2)
        x1 = self.conv2d_channel(x)
        x2 = self.conv2d_features1(x)
        x3 = self.conv2d_features2(x)
        x = torch.cat([x, x1, x2, x3], dim=1)
        x = self.conv2d_fusion(x)
        x = self.dropout(x)
        x_s1 = x

        eH = self.spa_token - H % self.spa_token
        eW = self.spa_token - W % self.spa_token
        pad = nn.ReflectionPad2d((0, eW, 0, eH)).to(x.device)
        x = pad(x)
        bb, cc, hh, ww = x.shape
        x = rearrange(x, 'b c (nh htoken) (nw wtoken)-> (b nh nw) (wtoken htoken) c',
                      htoken=self.spa_token, wtoken=self.spa_token)
        x = self.localSPAM(x)
        x = rearrange(x, '(b nh nw) (wtoken htoken) c-> b (nh htoken) (nw wtoken) c',
                      htoken=self.spa_token, wtoken=self.spa_token,
                      nh=hh // self.spa_token, nw=ww // self.spa_token)
        x = x[:, 0:H, 0:W, :] + x_s1.permute(0, 2, 3, 1)

        x = rearrange(x, 'b h w c -> b (h w) c')
        x = self.SPAM(x)
        x = rearrange(x, 'b (h w) c-> b h w c', h=H, w=W)

        x_spa = self.nn1(x)

        x_s = x_spe.unsqueeze(1)
        x_s = self.spe_token1(x_s) + self.spe_token2(x_s) + x_s
        x_s = self.dropout(x_s).squeeze(1).permute(0, 3, 1, 2)
        Patch_pool = nn.AvgPool2d((H, W))
        x_s = Patch_pool(x_s)
        x_s = split_band(x_s, self.move_num, self.spec_num)
        bb, nn_, cc, hh, ww = x_s.shape
        x_s = rearrange(x_s, 'b n c h w-> (b h w) n c')
        x_s = self.SPEM(x_s)
        x_s = rearrange(x_s, '(b h w) n c-> b (n c) (h w)', h=hh, w=ww).mean(-1)
        x_s = self.nn2(x_s).unsqueeze(1).unsqueeze(1)
        x = x_spa * x_s
        x = rearrange(x, 'b h w c-> b c h w')
        x = self.outhead(x)

        logits = x.mean(dim=(2, 3))
        return logits


@register_model('MambaLG', expects_4d=True, dim=64, depth=1, dropout=0.1, spec_num=12, spec_rate=0.5, spa_token=4)
def mambalg(pretrained: bool = False, **kwargs) -> MambaLG:
    """Constructs a MambaLG model."""
    if 'bands' in kwargs:
        kwargs['in_channels'] = kwargs.pop('bands')
    return MambaLG(**kwargs)
