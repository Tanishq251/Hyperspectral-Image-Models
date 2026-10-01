"""
R2Mamba: Route-Reliability Mamba for Hyperspectral Image Classification
github: https://github.com/xunshang111/R2Mamba-HSI
Paper: https://doi.org/10.1109/JSTARS.2026.3728152
Venue: IEEE JSTARS
Year: 2026

Ported from the official repo: ``mamba2_ssm.py`` (pure-PyTorch Mamba2/SSD) and
``r2mamba.py`` are merged into this file; architecture is unchanged.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from functools import lru_cache

import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange, repeat

from models.registry import register_model


# ──────────────────────────────────────────────
# Mamba2 / SSD block (from mamba2_ssm.py)
# ──────────────────────────────────────────────

class RMSNorm(nn.Module):
    def __init__(self, d: int, eps: float = 1e-5):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(d))

    def forward(self, x: torch.Tensor, z: torch.Tensor | None = None) -> torch.Tensor:
        if z is not None:
            x = x * F.silu(z)
        return x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps) * self.weight


def segment_sum(x: torch.Tensor) -> torch.Tensor:
    """Lower-triangular segment sum used by SSD."""
    t = x.size(-1)
    x_rep = repeat(x, "... d -> ... d e", e=t)
    mask = torch.tril(torch.ones(t, t, dtype=torch.bool, device=x.device), diagonal=-1)
    x_rep = x_rep.masked_fill(~mask, 0)
    x_segsum = torch.cumsum(x_rep, dim=-2)
    mask = torch.tril(torch.ones(t, t, dtype=torch.bool, device=x.device), diagonal=0)
    return x_segsum.masked_fill(~mask, -torch.inf)


def ssd(
    x: torch.Tensor,
    A: torch.Tensor,
    B: torch.Tensor,
    C: torch.Tensor,
    chunk_size: int,
    initial_states: torch.Tensor | None = None,
) -> torch.Tensor:
    """
    Structured State-Space Duality computation.

    Args:
        x: [B, L, H, P]
        A: [B, L, H]
        B: [B, L, 1, N]
        C: [B, L, 1, N]
        chunk_size: SSD chunk length; L must be divisible by it.
    Returns:
        Y: [B, L, H, P]
    """
    assert x.shape[1] % chunk_size == 0, "sequence length must be divisible by chunk_size"

    x, A, B, C = [rearrange(m, "b (c l) ... -> b c l ...", l=chunk_size) for m in (x, A, B, C)]
    A = rearrange(A, "b c l h -> b h c l")
    A_cumsum = torch.cumsum(A, dim=-1)

    L = torch.exp(segment_sum(A))
    Y_diag = torch.einsum("bclhn,bcshn,bhcls,bcshp->bclhp", C, B, L, x)

    decay_states = torch.exp(A_cumsum[:, :, :, -1:] - A_cumsum)
    states = torch.einsum("bclhn,bhcl,bclhp->bchpn", B, decay_states, x)

    if initial_states is None:
        initial_states = torch.zeros_like(states[:, :1])
    states = torch.cat([initial_states, states], dim=1)

    decay_chunk = torch.exp(segment_sum(F.pad(A_cumsum[:, :, :, -1], (1, 0))))
    new_states = torch.einsum("bhzc,bchpn->bzhpn", decay_chunk, states)
    states = new_states[:, :-1]

    state_decay_out = torch.exp(A_cumsum)
    Y_off = torch.einsum("bclhn,bchpn,bhcl->bclhp", C, states, state_decay_out)
    return rearrange(Y_diag + Y_off, "b c l h p -> b (c l) h p")


class Mamba2SSM(nn.Module):
    """
    Mamba2 block for sequence tokens used inside Manhattan multi-path scan.

    Args:
        d_model: token channel dimension C.
        d_state: state size N.
        headdim: per-head hidden dimension. Must divide expand * d_model.
        chunk_size: SSD chunk size. Non-divisible sequence lengths are padded and cropped.
        expand: inner expansion ratio.
        d_conv: depthwise Conv1d kernel over projected xBC stream.
        dropout: output dropout.
    """
    def __init__(
        self,
        d_model: int,
        d_state: int = 16,
        headdim: int = 16,
        chunk_size: int = 8,
        expand: int = 2,
        d_conv: int = 4,
        dropout: float = 0.0,
    ):
        super().__init__()
        self.d_model = d_model
        self.d_inner = expand * d_model
        if self.d_inner % headdim != 0:
            raise ValueError(f"expand * d_model ({self.d_inner}) must be divisible by headdim ({headdim}).")
        self.nheads = self.d_inner // headdim
        self.d_state = d_state
        self.headdim = headdim
        self.chunk_size = chunk_size
        self.d_conv = d_conv

        d_in_proj = 2 * self.d_inner + 2 * self.d_state + self.nheads
        self.in_proj = nn.Linear(d_model, d_in_proj, bias=False)
        self.conv1d = nn.Conv1d(
            in_channels=self.d_inner + 2 * self.d_state,
            out_channels=self.d_inner + 2 * self.d_state,
            kernel_size=d_conv,
            groups=self.d_inner + 2 * self.d_state,
            padding=d_conv - 1,
        )
        self.dt_bias = nn.Parameter(torch.zeros(self.nheads))
        self.A_log = nn.Parameter(torch.randn(self.nheads) * 0.1)
        self.D = nn.Parameter(torch.ones(self.nheads))
        self.norm = RMSNorm(self.d_inner)
        self.out_proj = nn.Linear(self.d_inner, d_model, bias=False)
        self.dropout = nn.Dropout(dropout)

    def _pad_to_chunk(self, u: torch.Tensor) -> tuple[torch.Tensor, int]:
        original_len = u.shape[1]
        remainder = original_len % self.chunk_size
        if remainder == 0:
            return u, 0
        pad_len = self.chunk_size - remainder
        if original_len >= 4:
            pad_values = u[:, : min(5, original_len)].mean(dim=1, keepdim=True)
        else:
            pad_values = u[:, :1]
        pad = pad_values.repeat(1, pad_len, 1)
        return torch.cat([pad, u], dim=1), pad_len

    def forward(self, u: torch.Tensor) -> torch.Tensor:
        original_seq_len = u.shape[1]
        u, pad_len = self._pad_to_chunk(u)
        seq_len = u.shape[1]

        zxbcdt = self.in_proj(u)
        z = zxbcdt[:, :, : self.d_inner]
        xBC = zxbcdt[:, :, self.d_inner : self.d_inner + self.d_inner + 2 * self.d_state]
        dt = zxbcdt[:, :, -self.nheads :]
        dt = F.softplus(dt + self.dt_bias)

        xBC = self.conv1d(xBC.transpose(1, 2)).transpose(1, 2)[:, :seq_len, :]
        xBC = F.silu(xBC)

        x = xBC[:, :, : self.d_inner]
        B = xBC[:, :, self.d_inner : self.d_inner + self.d_state]
        C = xBC[:, :, -self.d_state :]

        x = x.view(x.shape[0], seq_len, self.nheads, self.headdim)
        A = -torch.exp(self.A_log)
        A_dt = A.view(1, 1, self.nheads) * dt
        x_dt = x * dt.unsqueeze(-1)

        y = ssd(
            x=x_dt,
            A=A_dt,
            B=B.unsqueeze(2),
            C=C.unsqueeze(2),
            chunk_size=self.chunk_size,
        )
        y = y + x * self.D.view(1, 1, self.nheads, 1)
        y = y.reshape(y.shape[0], seq_len, self.d_inner)
        y = self.norm(y, z)
        y = self.out_proj(y)

        if pad_len > 0:
            y = y[:, pad_len : pad_len + original_seq_len, :]
        return self.dropout(y)


# ──────────────────────────────────────────────
# R2Mamba architecture (from r2mamba.py)
# ──────────────────────────────────────────────

R2MAMBA_ROUTES = ("row", "col", "diag", "anti_diag", "manhattan_ring")


@lru_cache(maxsize=128)
def scan_indices(height: int, width: int, route: str) -> torch.Tensor:
    coords = [(i, j) for i in range(height) for j in range(width)]
    if route == "row":
        ordered = coords
    elif route == "col":
        ordered = [(i, j) for j in range(width) for i in range(height)]
    elif route == "diag":
        ordered = []
        for s in range(height + width - 1):
            group = [(i, s - i) for i in range(height) if 0 <= s - i < width]
            ordered.extend(group if s % 2 == 0 else list(reversed(group)))
    elif route == "anti_diag":
        ordered = []
        for s in range(height + width - 1):
            group = []
            for i in range(height):
                j = s - i
                if 0 <= j < width:
                    group.append((i, width - 1 - j))
            ordered.extend(group if s % 2 == 0 else list(reversed(group)))
    elif route == "manhattan_ring":
        center_i, center_j = (height - 1) / 2.0, (width - 1) / 2.0
        ordered = sorted(coords, key=lambda p: (abs(p[0] - center_i) + abs(p[1] - center_j), p[0], p[1]))
    else:
        raise ValueError(f"Unknown R2Mamba route: {route}")
    return torch.tensor([i * width + j for i, j in ordered], dtype=torch.long)


def gather_route(x_bhwc: torch.Tensor, route: str) -> tuple[torch.Tensor, torch.Tensor]:
    batch, height, width, channels = x_bhwc.shape
    index = scan_indices(height, width, route).to(x_bhwc.device)
    sequence = x_bhwc.reshape(batch, height * width, channels).index_select(1, index)
    return sequence, index


def scatter_route(sequence: torch.Tensor, index: torch.Tensor, height: int, width: int) -> torch.Tensor:
    batch, length, channels = sequence.shape
    output = torch.empty(batch, length, channels, device=sequence.device, dtype=sequence.dtype)
    output[:, index, :] = sequence
    return output.reshape(batch, height, width, channels)


@dataclass(frozen=True)
class R2MambaConfig:
    dim: int = 96
    depth: int = 1
    d_state: int = 16
    headdim: int = 16
    chunk_size: int = 8
    ssm_expand: int = 2
    d_conv: int = 4
    dropout: float = 0.1
    gga_alpha: float = 0.10


def channel_mlp(dim: int, dropout: float) -> nn.Sequential:
    return nn.Sequential(
        nn.LayerNorm(dim),
        nn.Linear(dim, dim * 4),
        nn.GELU(),
        nn.Dropout(dropout),
        nn.Linear(dim * 4, dim),
        nn.Dropout(dropout),
    )


class SSTE(nn.Module):
    def __init__(self, in_bands: int, dim: int):
        super().__init__()
        self.proj = nn.Sequential(
            nn.Conv2d(in_bands, dim, kernel_size=1, bias=False),
            nn.BatchNorm2d(dim),
            nn.GELU(),
            nn.Conv2d(dim, dim, kernel_size=3, padding=1, groups=dim, bias=False),
            nn.Conv2d(dim, dim, kernel_size=1, bias=False),
            nn.BatchNorm2d(dim),
            nn.GELU(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.proj(x).permute(0, 2, 3, 1).contiguous()


class R2MB(nn.Module):
    def __init__(self, cfg: R2MambaConfig):
        super().__init__()
        self.routes = R2MAMBA_ROUTES
        self.norm = nn.LayerNorm(cfg.dim)
        self.route_ssms = nn.ModuleDict(
            {
                route: Mamba2SSM(
                    d_model=cfg.dim,
                    d_state=cfg.d_state,
                    headdim=cfg.headdim,
                    chunk_size=cfg.chunk_size,
                    expand=cfg.ssm_expand,
                    d_conv=cfg.d_conv,
                    dropout=cfg.dropout,
                )
                for route in self.routes
            }
        )
        gate_hidden = max(cfg.dim // 2, 8)
        self.route_gate = nn.Sequential(
            nn.Conv2d(cfg.dim + 1, gate_hidden, kernel_size=1, bias=False),
            nn.BatchNorm2d(gate_hidden),
            nn.GELU(),
            nn.Conv2d(gate_hidden, len(self.routes), kernel_size=1),
        )
        self.local_mlp = channel_mlp(cfg.dim, cfg.dropout)
        self.gamma_route = nn.Parameter(torch.ones(cfg.dim) * 1e-3)
        self.gamma_mlp = nn.Parameter(torch.ones(cfg.dim) * 1e-3)

    @staticmethod
    def boundary_cue(x_bhwc: torch.Tensor) -> torch.Tensor:
        x_bchw = x_bhwc.permute(0, 3, 1, 2)
        gx = F.pad((x_bchw[:, :, :, 1:] - x_bchw[:, :, :, :-1]).abs().mean(1, keepdim=True), (0, 1, 0, 0))
        gy = F.pad((x_bchw[:, :, 1:, :] - x_bchw[:, :, :-1, :]).abs().mean(1, keepdim=True), (0, 0, 0, 1))
        cue = gx + gy
        return cue / (cue.amax(dim=(-2, -1), keepdim=True) + 1e-6)

    def forward(self, x_bhwc: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        batch, height, width, _ = x_bhwc.shape
        z = self.norm(x_bhwc)
        route_outputs = []
        for route in self.routes:
            sequence, index = gather_route(z, route)
            sequence = self.route_ssms[route](sequence)
            route_outputs.append(scatter_route(sequence, index, height, width))
        stacked = torch.stack(route_outputs, dim=1)
        gate_input = z.permute(0, 3, 1, 2).contiguous()
        gate_input = torch.cat([gate_input, self.boundary_cue(z)], dim=1)
        route_weights = torch.softmax(self.route_gate(gate_input), dim=1)
        mixed = (stacked * route_weights[:, :, :, :, None]).sum(dim=1)
        out = x_bhwc + self.gamma_route * mixed
        out = out + self.gamma_mlp * self.local_mlp(out)
        return out, route_weights


class GGA(nn.Module):
    def __init__(self, dim: int, num_classes: int, dropout: float = 0.1, alpha: float = 0.10):
        super().__init__()
        if not 0.0 <= alpha <= 1.0:
            raise ValueError(f"GGA alpha must be in [0, 1], got {alpha}")
        self.alpha = float(alpha)
        self.classifier = nn.Sequential(
            nn.Linear(dim, dim * 2),
            nn.BatchNorm1d(dim * 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(dim * 2, num_classes),
        )

    def pool(self, features_bhwc: torch.Tensor, route_weights_bkhw: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
        batch, height, width, _ = features_bhwc.shape
        _, num_routes, _, _ = route_weights_bkhw.shape
        entropy = -(route_weights_bkhw * torch.log(route_weights_bkhw + eps)).sum(dim=1) / math.log(num_routes)
        certainty = (1.0 - entropy.clamp(0.0, 1.0)).clamp_min(0.0)
        denom = certainty.sum(dim=(1, 2), keepdim=True)
        uniform = torch.full_like(certainty, 1.0 / float(height * width))
        certainty_attention = torch.where(denom > eps, certainty / denom.clamp_min(eps), uniform)
        attention = (1.0 - self.alpha) * uniform + self.alpha * certainty_attention
        return (features_bhwc * attention.unsqueeze(-1)).sum(dim=(1, 2))

    def forward(self, features_bhwc: torch.Tensor, route_weights_bkhw: torch.Tensor) -> torch.Tensor:
        pooled = self.pool(features_bhwc, route_weights_bkhw)
        return self.classifier(pooled)


class R2MambaHSI(nn.Module):
    def __init__(
        self,
        in_bands: int,
        num_classes: int,
        dim: int = 96,
        depth: int = 1,
        d_state: int = 16,
        headdim: int = 16,
        chunk_size: int = 8,
        ssm_expand: int = 2,
        d_conv: int = 4,
        dropout: float = 0.1,
        gga_alpha: float = 0.10,
    ):
        super().__init__()
        if depth < 1:
            raise ValueError("R2Mamba requires at least one R2MB block.")
        cfg = R2MambaConfig(
            dim=dim,
            depth=depth,
            d_state=d_state,
            headdim=headdim,
            chunk_size=chunk_size,
            ssm_expand=ssm_expand,
            d_conv=d_conv,
            dropout=dropout,
            gga_alpha=gga_alpha,
        )
        self.sste = SSTE(in_bands, dim)
        self.blocks = nn.ModuleList([R2MB(cfg) for _ in range(depth)])
        self.norm = nn.LayerNorm(dim)
        self.gga = GGA(dim, num_classes, dropout=dropout, alpha=gga_alpha)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        features = self.sste(x)
        route_weights = None
        for block in self.blocks:
            features, route_weights = block(features)
        features = self.norm(features)
        return self.gga(features, route_weights)


@register_model('R2Mamba', expects_4d=True, dim=96, depth=1, dropout=0.1, gga_alpha=0.10)
def r2mamba(num_classes, bands, patch_size=11, **kwargs):
    return R2MambaHSI(in_bands=bands, num_classes=num_classes, **kwargs)
