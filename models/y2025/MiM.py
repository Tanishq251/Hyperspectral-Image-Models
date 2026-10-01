"""
MiM (Mamba-in-Mamba): Centralized Mamba-Cross-Scan in Channels for HSI Classification

Paper: https://arxiv.org/abs/2405.12003
GitHub: https://github.com/zhouweilian1904/Mamba-in-Mamba
Year: 2024, IEEE TGRS

Architecture:
  1. Patch embedding with einops rearrange
  2. 4-directional snake-scan flattening (rot90 / flip augmentations)
  3. MiM blocks: T_Mamba encoders with bidirectional SSM + Gaussian decay masking
  4. Weighted Multi-directional Fusion (WMF) with learned softmax weights
  5. MLP classification head

Pure PyTorch implementation — no CUDA kernels needed.
Uses custom parallel scan (PScan) for efficient SSM computation.
"""
from models.registry import register_model

import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from torch import Tensor
from einops import rearrange
from einops.layers.torch import Rearrange


# ──────────────────────────────────────────────
# Snake-scan flattening (from multiscan_v3.py)
# ──────────────────────────────────────────────

def snake_flatten(img_tensor):
    """Flatten 2D spatial dims via snake/boustrophedon scan.

    Args:
        img_tensor: [B, C, H, W]
    Returns:
        sequence: [B, H*W, C]
    """
    batch_size, channels, height, width = img_tensor.size()
    img_tensor = img_tensor.permute(0, 2, 3, 1)  # [B, H, W, C]
    for i in range(height):
        if i % 2 != 0:
            img_tensor = img_tensor.clone()
            img_tensor[:, i] = img_tensor[:, i, :].flip(dims=[1])
    return img_tensor.reshape(batch_size, -1, channels)


# ──────────────────────────────────────────────
# Parallel Scan (PScan) — pure PyTorch
# ──────────────────────────────────────────────

class PScan(torch.autograd.Function):
    @staticmethod
    def pscan(A, X):
        B, D, L, _ = A.size()
        num_steps = int(math.log2(L))

        Aa = A
        Xa = X
        for k in range(num_steps):
            T = 2 * (Xa.size(2) // 2)
            Aa = Aa[:, :, :T].view(B, D, T // 2, 2, -1)
            Xa = Xa[:, :, :T].view(B, D, T // 2, 2, -1)
            Xa[:, :, :, 1].add_(Aa[:, :, :, 1].mul(Xa[:, :, :, 0]))
            Aa[:, :, :, 1].mul_(Aa[:, :, :, 0])
            Aa = Aa[:, :, :, 1]
            Xa = Xa[:, :, :, 1]

        for k in range(num_steps - 1, -1, -1):
            Aa = A[:, :, 2 ** k - 1: L: 2 ** k]
            Xa = X[:, :, 2 ** k - 1: L: 2 ** k]
            T = 2 * (Xa.size(2) // 2)
            if T < Xa.size(2):
                Xa[:, :, -1].add_(Aa[:, :, -1].mul(Xa[:, :, -2]))
                Aa[:, :, -1].mul_(Aa[:, :, -2])
            Aa = Aa[:, :, :T].view(B, D, T // 2, 2, -1)
            Xa = Xa[:, :, :T].view(B, D, T // 2, 2, -1)
            Xa[:, :, 1:, 0].add_(Aa[:, :, 1:, 0].mul(Xa[:, :, :-1, 1]))
            Aa[:, :, 1:, 0].mul_(Aa[:, :, :-1, 1])

    @staticmethod
    def forward(ctx, A_in, X_in):
        A = A_in.clone()
        X = X_in.clone()
        A = A.transpose(2, 1)
        X = X.transpose(2, 1)
        PScan.pscan(A, X)
        ctx.save_for_backward(A_in, X)
        return X.transpose(2, 1)

    @staticmethod
    def backward(ctx, grad_output_in):
        A_in, X = ctx.saved_tensors
        A = A_in.clone()
        A = A.transpose(2, 1)
        A = torch.cat((A[:, :, :1], A[:, :, 1:].flip(2)), dim=2)
        grad_output_b = grad_output_in.transpose(2, 1)
        grad_output_b = grad_output_b.flip(2)
        PScan.pscan(A, grad_output_b)
        grad_output_b = grad_output_b.flip(2)
        Q = torch.zeros_like(X)
        Q[:, :, 1:].add_(X[:, :, :-1] * grad_output_b[:, :, 1:])
        return Q.transpose(2, 1), grad_output_b.transpose(2, 1)


pscan = PScan.apply


def _next_power_of_2(n):
    """Return the smallest power of 2 >= n."""
    if n <= 0:
        return 1
    return 1 << (n - 1).bit_length()


def selective_scan(x, delta, A, B, C, D):
    _, L, _ = x.shape
    deltaA = torch.exp(delta.unsqueeze(-1) * A)
    deltaB = delta.unsqueeze(-1) * B.unsqueeze(2)
    BX = deltaB * x.unsqueeze(-1)

    # PScan requires power-of-2 sequence length — pad along L (dim 1) if needed
    L_pad = _next_power_of_2(L)
    if L_pad != L:
        pad_len = L_pad - L
        # For 4D tensor [B, L, D, N], pad dim=1 (L): pad tuple is
        # (N_left, N_right, D_left, D_right, L_left, L_right)
        deltaA = F.pad(deltaA, (0, 0, 0, 0, 0, pad_len), value=0.0)
        BX = F.pad(BX, (0, 0, 0, 0, 0, pad_len), value=0.0)

    hs = pscan(deltaA, BX)

    if L_pad != L:
        hs = hs[:, :L, :, :]  # trim back

    y = (hs @ C.unsqueeze(-1)).squeeze(-1)
    y = y + D * x
    return y


# ──────────────────────────────────────────────
# SSM Module
# ──────────────────────────────────────────────

class SSM(nn.Module):
    def __init__(self, in_features, dt_rank, dim_inner, d_state):
        super().__init__()
        self.dt_rank = dt_rank
        self.dim_inner = dim_inner
        self.d_state = d_state
        self.deltaBC_layer = nn.Linear(in_features, dt_rank + 2 * d_state, bias=False)
        self.dt_proj_layer = nn.Linear(dt_rank, dim_inner, bias=True)
        self.A_log = nn.Parameter(
            torch.log(
                torch.arange(1, d_state + 1, dtype=torch.float32).repeat(dim_inner, 1)
            )
        )
        self.D = nn.Parameter(torch.ones(dim_inner))

    def forward(self, x):
        A = -torch.exp(self.A_log.float())
        D = self.D.float()
        deltaBC = self.deltaBC_layer(x)
        delta, B, C = torch.split(deltaBC, [self.dt_rank, self.d_state, self.d_state], dim=-1)
        delta = F.softplus(self.dt_proj_layer(delta))
        y = selective_scan(x, delta, A, B, C, D)
        return y


# ──────────────────────────────────────────────
# Vision Encoder Mamba Block
# ──────────────────────────────────────────────

class VisionEncoderMambaBlock(nn.Module):
    def __init__(self, dim, dt_rank, dim_inner, d_state, num_tokens):
        super().__init__()
        self.dim = dim
        self.num_tokens = num_tokens
        self.forward_conv1d = nn.Conv1d(dim, dim, kernel_size=1)
        self.backward_conv1d = nn.Conv1d(dim, dim, kernel_size=1)
        self.norm = nn.LayerNorm(dim)
        self.silu = nn.SiLU()
        self.forward_ssm = SSM(dim, dt_rank, dim_inner, d_state)
        self.backward_ssm = SSM(dim, dt_rank, dim_inner, d_state)
        self.proj1 = nn.Linear(dim, dim)
        self.proj2 = nn.Linear(dim, dim)
        self.softplus = nn.Softplus()
        self.adapool = nn.AdaptiveAvgPool1d(num_tokens)

    def forward(self, x):
        skip = x
        skip = self.adapool(rearrange(skip, 'b n d -> b d n'))
        skip = rearrange(skip, 'b d n -> b n d')

        x = self.norm(x)
        z = self.proj1(x)
        x = self.proj2(x)

        x1 = self._process_direction(x, self.forward_conv1d, self.forward_ssm)
        x1 = self.adapool(rearrange(x1, 'b n d -> b d n'))
        x1 = rearrange(x1, 'b d n -> b n d')
        x1 = x1 * self._gaussian_decay_mask(x1).unsqueeze(-1)
        x1 = self.silu(x1)

        x2 = self._process_direction(torch.flip(x, dims=[1]), self.backward_conv1d, self.backward_ssm)
        x2 = torch.flip(x2, dims=[1])
        x2 = self.adapool(rearrange(x2, 'b n d -> b d n'))
        x2 = rearrange(x2, 'b d n -> b n d')
        x2 = x2 * self._gaussian_decay_mask(x2).unsqueeze(-1)
        x2 = self.silu(x2)

        z = self.adapool(rearrange(z, 'b n d -> b d n'))
        z = rearrange(z, 'b d n -> b n d')
        z = self.silu(z)

        x1 = z * x1
        x2 = z * x2
        return x1 + x2 + skip

    @staticmethod
    def _gaussian_decay_mask(sequence):
        length = sequence.shape[1]
        center_index = (length + 1) // 2
        indices = torch.arange(length, dtype=torch.float32, device=sequence.device)
        sigma = torch.abs(indices - center_index).mean()
        sigma = sigma.clamp(min=1e-6)
        weights = torch.exp(-0.5 * ((indices - center_index) ** 2) / (sigma ** 2))
        weights = weights / weights.sum()
        weights = weights.repeat(sequence.size(0), 1)
        return weights

    def _process_direction(self, x, conv1d, ssm):
        x = rearrange(x, 'b s d -> b d s')
        x = self.softplus(conv1d(x))
        x = rearrange(x, 'b d s -> b s d')
        x = ssm(x)
        return x


# ──────────────────────────────────────────────
# T_Mamba & MiM Block
# ──────────────────────────────────────────────

class T_Mamba(nn.Module):
    def __init__(self, channels, image_size, patch_size, dim, depth, emb_dropout,
                 seq_length, num_tokens):
        super().__init__()
        self.num_patches = seq_length
        self.patch_dim = channels * patch_size * patch_size
        self.to_patch_embedding = nn.Sequential(
            nn.LayerNorm(self.patch_dim),
            nn.Linear(self.patch_dim, dim),
            nn.LayerNorm(dim)
        )
        self.dim = dim
        self.pos_embedding = nn.Parameter(torch.randn(1, self.num_patches, dim))
        self.dropout = nn.Dropout(emb_dropout)
        vim = VisionEncoderMambaBlock(dim=dim, dt_rank=dim, dim_inner=dim,
                                      d_state=dim, num_tokens=num_tokens)
        self.layers = nn.ModuleList([vim for _ in range(depth)])
        self.norm = nn.LayerNorm(dim)
        self.tanh = nn.Tanh()

    def forward(self, img):
        if img.dim() == 4:
            img = snake_flatten(img)

        if img.size(2) == self.patch_dim:
            x = self.to_patch_embedding(img)
        else:
            x = img

        x = x + self.pos_embedding[:, :self.num_patches]
        x = self.dropout(x)
        for layer in self.layers:
            x = layer(x)
        x = self.norm(x)
        x = self.tanh(x)
        return x


class MiM_block(nn.Module):
    def __init__(self, channels, image_size, patch_size, dim, depth, emb_dropout,
                 seq_length, num_tokens):
        super().__init__()
        self.T_mamba1 = T_Mamba(channels, image_size, patch_size=patch_size, dim=dim,
                                depth=depth, emb_dropout=emb_dropout,
                                seq_length=seq_length, num_tokens=num_tokens)

    def forward(self, x1, x2, x3, x4):
        return (self.T_mamba1(x1), self.T_mamba1(x2),
                self.T_mamba1(x3), self.T_mamba1(x4))


# ──────────────────────────────────────────────
# Main MiM Model
# ──────────────────────────────────────────────

def pair(t):
    return t if isinstance(t, tuple) else (t, t)


class MiMModel(nn.Module):
    """Mamba-in-Mamba for Hyperspectral Image Classification.

    Input: [B, 1, C, H, W] (5D) where C = spectral bands
    Output: [B, num_classes]

    Args:
        num_classes (int): Number of output classes.
        bands (int): Number of input spectral bands.
        image_size (int): Spatial patch size (H=W). Default: 13.
        patch_size (int): Sub-patch size for tokenization. Default: 1.
        dim (int): Embedding dimension. Default: 64.
        depth (int): Number of T_Mamba layers. Default: 2.
        emb_dropout (float): Embedding dropout. Default: 0.0.
    """
    def __init__(self,
                 num_classes=9,
                 bands=30,
                 image_size=13,
                 patch_size=1,
                 dim=64,
                 depth=2,
                 emb_dropout=0.0):
        super().__init__()
        self.channels = bands
        image_height, image_width = pair(image_size)
        patch_height, patch_width = pair(patch_size)
        assert image_height % patch_height == 0 and image_width % patch_width == 0, \
            'Image dimensions must be divisible by the patch size.'
        patch_dim = bands * patch_height * patch_width
        seq_length = (image_height // patch_height) ** 2

        self.to_patch_embedding = nn.Sequential(
            Rearrange("b c (h p1) (w p2) -> b h w (p1 p2 c)",
                      p1=patch_height, p2=patch_width),
            nn.LayerNorm(patch_dim),
            nn.Linear(patch_dim, bands),
            nn.LayerNorm(bands),
            Rearrange("b h w d -> b d h w"),
        )

        self.mim_1 = MiM_block(bands, image_size, patch_size=patch_size, dim=dim,
                               depth=depth, emb_dropout=emb_dropout,
                               seq_length=seq_length, num_tokens=seq_length)

        self.mlp_head = nn.Sequential(
            nn.Linear(dim, dim),
            nn.Tanh(),
            nn.Dropout(emb_dropout),
            nn.Linear(dim, num_classes)
        )

        self.k_weights = nn.Parameter(torch.ones(4) / 4, requires_grad=True)

    def _wmf(self, *o):
        k_weights = torch.softmax(self.k_weights, dim=0)
        O = sum(w * out for w, out in zip(k_weights, o))
        if O.dim() == 3:
            O = torch.mean(O, dim=1)
        return O

    def forward(self, x):
        # x: [B, 1, C, H, W]
        x = x.squeeze(1)  # [B, C, H, W]
        x = self.to_patch_embedding(x)

        x_1 = x
        x_4 = torch.rot90(x_1, k=-1, dims=(2, 3))
        x_2 = torch.flip(x_4, dims=[3])
        x_3 = torch.rot90(x_2, k=-1, dims=(2, 3))

        x_1 = snake_flatten(x_1)
        x_2 = snake_flatten(x_2)
        x_3 = snake_flatten(x_3)
        x_4 = snake_flatten(x_4)

        tm1, tm2, tm3, tm4 = self.mim_1(x_1, x_2, x_3, x_4)
        O = self._wmf(tm1, tm2, tm3, tm4)
        return self.mlp_head(O)


# ──────────────────────────────────────────────
# Registry
# ──────────────────────────────────────────────

@register_model('MiM',
                image_size=11,
                dim=64,
                depth=2,
                emb_dropout=0.0)
def mim_factory(pretrained: bool = False, **kwargs) -> MiMModel:
    """Construct Mamba-in-Mamba (MiM) model.

    Standard kwargs: num_classes, bands, patch_size.
    MiM-specific: image_size, dim, depth, emb_dropout.
    """
    # Pipeline passes patch_size = spatial window size (e.g. 11).
    # MiM uses image_size for spatial window, patch_size for sub-patch tokenization.
    # Remap: pipeline's patch_size → image_size, sub-patch stays at 1.
    if 'patch_size' in kwargs:
        kwargs['image_size'] = kwargs.pop('patch_size')
    # MiM sub-patch tokenization is always 1
    kwargs['patch_size'] = 1
    return MiMModel(**kwargs)
