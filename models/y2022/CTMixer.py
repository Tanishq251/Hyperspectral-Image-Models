from models.registry import register_model
"""
CTMixer: Convolution Transformer Mixer for Hyperspectral Image Classification

Paper: https://ieeexplore.ieee.org/document/9924229
GitHub: https://github.com/ZJier/CTMixer
Venue: IEEE Geoscience and Remote Sensing Letters (GRSL)
Year: 2022
"""

import math
import einops
import torch
import torch.nn as nn
import torch.nn.functional as F
from einops.layers.torch import Rearrange


# ──────────────────────────────────────────────
# embeddings.py
# ──────────────────────────────────────────────

class PatchEmbeddings(nn.Module):
    """Module that extracts patches and projects them."""
    def __init__(self, patch_size: int, patch_dim: int, emb_dim: int):
        super().__init__()
        self.patchify = Rearrange(
            "b c (h p1) (w p2) -> b (h w) c p1 p2",
            p1=patch_size, p2=patch_size)

        self.flatten = nn.Flatten(start_dim=2)
        self.proj = nn.Linear(in_features=patch_dim, out_features=emb_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.patchify(x)
        x = self.flatten(x)
        x = self.proj(x)
        return x


class CLSToken(nn.Module):
    """Prepend cls token to each embedding."""
    def __init__(self, dim: int):
        super().__init__()
        self.cls_token = nn.Parameter(torch.randn(1, 1, dim))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b = x.shape[0]
        cls_tokens = self.cls_token.repeat(b, 1, 1)
        x = torch.cat([cls_tokens, x], dim=1)
        return x


class PositionalEmbeddings(nn.Module):
    """Learned positional embeddings."""
    def __init__(self, num_pos: int, dim: int):
        super().__init__()
        self.pos = nn.Parameter(torch.randn(num_pos, dim))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.pos


# ──────────────────────────────────────────────
# transformer.py
# ──────────────────────────────────────────────

class Attention_Conv(nn.Module):

    def __init__(self, dim: int, head_dim: int, num_heads: int, num_patch: int, patch_size: int):
        super().__init__()

        self.dim = dim
        self.head_dim = head_dim
        self.num_patch = num_patch
        self.patch_size = patch_size
        self.num_heads = num_heads
        self.inner_dim = head_dim * num_heads
        self.scale = head_dim ** -0.5
        self.attn = nn.Softmax(dim=-1)
        self.act = nn.ReLU(inplace=True)
        self.bn = nn.BatchNorm2d(dim)
        self.qkv = nn.Conv2d(dim, self.inner_dim * 3, kernel_size=1, padding=0, groups=dim, bias=False)
        self.avgpool = nn.AdaptiveAvgPool1d(dim)

        self.qs = nn.Conv2d(dim, dim, kernel_size=(1, 3), padding=(0, 1), groups=dim, bias=False)
        self.ks = nn.Conv2d(dim, dim, kernel_size=(3, 1), padding=(1, 0), groups=dim, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b, n, d = x.shape
        x = x.contiguous().view(b, self.dim, self.num_patch, self.num_patch)

        qkv = self.qkv(self.act(self.bn(x)))
        qkv = qkv.contiguous().view(b, self.num_patch * self.num_patch, self.inner_dim * 3)
        qkv = qkv.chunk(3, dim=-1)
        q, k, v = map(lambda t: einops.rearrange(t, "b n (h d) -> b h n d", h=self.num_heads), qkv)

        q_1 = q[:, 0:self.num_heads // 2, :, :]
        k_1 = k[:, 0:self.num_heads // 2, :, :]
        v_1 = v[:, 0:self.num_heads // 2, :, :]
        q_2 = q[:, self.num_heads // 2:self.num_heads, :, :].reshape(b, -1, int(math.sqrt(n)), int(math.sqrt(n)))
        k_2 = k[:, self.num_heads // 2:self.num_heads, :, :].reshape(b, -1, int(math.sqrt(n)), int(math.sqrt(n)))
        v_2 = v[:, self.num_heads // 2:self.num_heads, :, :].reshape(b, -1, int(math.sqrt(n)), int(math.sqrt(n)))

        q_2 = self.qs(q_2)
        k_2 = self.ks(k_2)
        res_2 = (q_2 + k_2 + v_2).reshape(b, n, -1)

        scores = torch.einsum("b h i d, b h j d -> b h i j", q_1, k_1)
        scores = scores * self.scale
        attn = self.attn(scores)
        out = torch.einsum("b h i j, b h j d -> b h i d", attn, v_1)
        out = einops.rearrange(out, "b h n d -> b n (h d)")
        res = torch.cat([out, res_2], axis=2)
        out = self.avgpool(res)
        return out


class FeedForward_Conv(nn.Module):

    def __init__(self, dim: int, hidden_dim: int, num_patch: int, patch_size: int):
        super().__init__()
        self.dim = dim
        self.num_patch = num_patch
        self.patch_size = patch_size
        self.conv1 = nn.Sequential(
            nn.BatchNorm2d(dim), nn.GELU(),
            nn.Conv2d(dim, 64, kernel_size=1, padding=0, bias=False))

        self.conv2 = nn.Sequential(
            nn.BatchNorm2d(64), nn.GELU(),
            nn.Conv2d(64, 64, kernel_size=3, padding=1, groups=64, bias=False))

        self.conv3 = nn.Sequential(
            nn.BatchNorm2d(64), nn.GELU(),
            nn.Conv2d(64, dim, kernel_size=1, padding=0, bias=False))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b, hw, dim = x.shape
        x_reshape = x.contiguous().view(b, self.dim, self.num_patch, self.num_patch)
        out1 = self.conv1(x_reshape)
        out2 = self.conv2(out1)
        out3 = self.conv3(out2) + x_reshape
        result = out3.contiguous().view(b, self.num_patch * self.num_patch, self.dim)

        return result


class transformer(nn.Module):

    def __init__(self, dim: int, num_layers: int, num_heads: int, head_dim: int, hidden_dim: int, num_patch: int, patch_size: int):
        super().__init__()
        self.layers = nn.ModuleList()
        for _ in range(num_layers):
            layer = [
                nn.Sequential(nn.LayerNorm(dim), Attention_Conv(dim, head_dim, num_heads, num_patch, patch_size)),
                nn.Sequential(nn.LayerNorm(dim), FeedForward_Conv(dim, hidden_dim, num_patch, patch_size))
            ]
            self.layers.append(nn.ModuleList(layer))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        for attn, ff in self.layers:
            x = attn(x) + x
            x = ff(x) + x
        return x


# ──────────────────────────────────────────────
# CTMixer.py
# ──────────────────────────────────────────────

class Pooling(nn.Module):
    def __init__(self, pool: str = "mean"):
        super().__init__()
        if pool not in ["mean", "cls"]:
            raise ValueError("pool must be one of {mean, cls}")

        self.pool_fn = self.mean_pool if pool == "mean" else self.cls_pool

    def mean_pool(self, x: torch.Tensor) -> torch.Tensor:
        return x.mean(dim=1)

    def cls_pool(self, x: torch.Tensor) -> torch.Tensor:
        return x[:, 0]

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.pool_fn(x)


class Classifier(nn.Module):

    def __init__(self, dim: int, num_classes: int):
        super().__init__()
        self.model = nn.Sequential(
            nn.LayerNorm(dim),
            nn.Linear(in_features=dim, out_features=num_classes)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.model(x)


class Res2(nn.Module):
    def __init__(self, in_channels, inter_channels, kernel_size, padding=0):
        super(Res2, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, inter_channels, kernel_size=kernel_size, padding=padding)
        self.bn1 = nn.BatchNorm2d(inter_channels)
        self.conv2 = nn.Conv2d(inter_channels, in_channels, kernel_size=kernel_size, padding=padding)
        self.bn2 = nn.BatchNorm2d(in_channels)

    def forward(self, X):
        X = F.relu(self.bn1(self.conv1(X)))
        X = self.bn2(self.conv2(X))
        return X


class Res(nn.Module):
    def __init__(self, in_channels, kernel_size, padding, groups_s):
        super(Res, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, in_channels, kernel_size=kernel_size, padding=padding, groups=groups_s)
        self.bn1 = nn.BatchNorm2d(in_channels)

        self.conv2 = nn.Conv2d(in_channels, in_channels, kernel_size=kernel_size, padding=padding, groups=groups_s)
        self.bn2 = nn.BatchNorm2d(in_channels)

        self.res2 = Res2(in_channels, 32, kernel_size=kernel_size, padding=padding)

    def forward(self, X):
        Y = F.relu(self.bn1(self.conv1(X)))
        Y = self.bn2(self.conv2(Y))
        Z = self.res2(X)
        return F.relu(X + Y + Z)


class CTMixerNet(nn.Module):

    def __init__(self, channels, num_classes, image_size, datasetname='IndianPines', num_layers: int = 1, num_heads: int = 4,
                 patch_size: int = 1, emb_dim: int = 128, head_dim=64, hidden_dim: int = 64, pool: str = "mean"):
        super().__init__()
        self.emb_dim = emb_dim

        self.hidden_dim = hidden_dim
        self.channels = channels
        self.image_size = image_size
        self.num_patches = (image_size // patch_size) ** 2
        self.num_patch = int(math.sqrt(self.num_patches))
        self.act = nn.ReLU(inplace=True)

        # Conv Preprocessing Module (Ref-SPRN)
        if datasetname == 'IndianPines':
            groups = 11
            groups_width = 37
        elif datasetname == 'PaviaU':
            groups = 5
            groups_width = 64
        elif datasetname == 'Salinas':
            groups = 11
            groups_width = 37
        elif datasetname == 'Houston':
            groups = 5
            groups_width = 64
        else:
            groups = 11
            groups_width = 37
        new_bands = math.ceil(channels / groups) * groups
        patch_dim = (groups * groups_width) * patch_size ** 2
        pad_size = new_bands - channels
        self.pad = nn.ReplicationPad3d((0, 0, 0, 0, 0, pad_size))
        self.conv_1 = nn.Conv2d(new_bands, groups * groups_width, (1, 1), groups=groups)
        self.bn_1 = nn.BatchNorm2d(groups * groups_width)
        self.res0 = Res(groups * groups_width, (3, 3), (1, 1), groups_s=groups)

        # Dual Residual Block (Ref-RDACN)
        self.bn1 = nn.BatchNorm2d(emb_dim)
        self.conv1 = nn.Conv2d(emb_dim, 64, kernel_size=1, padding=0)
        self.bn2 = nn.BatchNorm2d(64)
        self.conv2 = nn.Conv2d(64, 64, kernel_size=3, padding=1, groups=64)
        self.bn3 = nn.BatchNorm2d(64)
        self.conv3 = nn.Conv2d(64, emb_dim, kernel_size=1, padding=0)

        # Vision Transformer
        self.patch_embeddings = PatchEmbeddings(patch_size=patch_size, patch_dim=patch_dim, emb_dim=emb_dim)
        self.pos_embeddings = PositionalEmbeddings(num_pos=self.num_patches, dim=emb_dim)
        self.transformer = transformer(dim=emb_dim, num_layers=num_layers, num_heads=num_heads,
                                        head_dim=head_dim, hidden_dim=hidden_dim, num_patch=self.num_patch, patch_size=patch_size)
        self.dropout = nn.Dropout(0.5)

        self.pool = Pooling(pool=pool)
        self.classifier = Classifier(dim=emb_dim, num_classes=num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.pad(x).squeeze(1)
        b, c, h, w = x.shape
        x = F.relu(self.bn_1(self.conv_1(x)))
        x = self.res0(x)

        x4 = self.patch_embeddings(x)
        x5 = self.pos_embeddings(x4)
        x6 = self.transformer(x5)

        x4_c = x4.reshape(b, -1, h, w)
        x_c1 = self.conv1(self.act(self.bn1(x4_c)))
        x_c2 = self.conv2(self.act(self.bn2(x_c1)))
        x_c3 = self.conv3(self.act(self.bn3(x_c2)))

        x7 = self.pool(self.dropout(x6 + x_c3.reshape(b, h * w, -1)))

        return self.classifier(x7)


# Register model
# Note: the original CTMixer constructor takes `patch_size` to mean the ViT
# sub-patch subdivision size (kept at its default of 1) and a separate
# `image_size` for the spatial extent of the input cube. The framework's
# `patch_size` kwarg (spatial patch, default 11) is mapped to `image_size`
# here to avoid clashing with the model's own internal `patch_size`.

@register_model('CTMixer', expects_4d=False, datasetname='IndianPines', num_layers=1, num_heads=4,
                 emb_dim=128, head_dim=64, hidden_dim=64, pool='mean')
def ctmixer(pretrained: bool = False, **kwargs) -> CTMixerNet:
    """Constructs a CTMixer model."""
    if 'bands' in kwargs:
        kwargs['channels'] = kwargs.pop('bands')
    if 'patch_size' in kwargs:
        kwargs['image_size'] = kwargs.pop('patch_size')
    return CTMixerNet(**kwargs)
