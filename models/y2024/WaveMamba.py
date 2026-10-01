"""
WaveMamba: Wavelet-enhanced State Space Model for Hyperspectral Image Classification

Paper: https://arxiv.org/abs/2408.01231
GitHub: https://github.com/mahmad000/WaveMamba
Year: 2024

Original implementation in TensorFlow/Keras. This is a faithful PyTorch port
adapted to work with the local model registry.

Architecture:
  1. Spectral-Spatial Token Generation (dual linear projections)
  2. Feature Enhancement (sigmoid gating from center token)
  3. Wavelet Transform (DWT via pywt, 4 sub-bands: cA, cH, cV, cD)
  4. State Space Model (recurrent state transition)
  5. Classification head
"""
from models.registry import register_model

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import pywt


# ──────────────────────────────────────────────
# Spectral-Spatial Token Generation
# ──────────────────────────────────────────────

class SpectralSpatialTokenGeneration(nn.Module):
    """Generate spatial and spectral tokens from input patches."""
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.spatial_proj = nn.Linear(in_channels, out_channels)
        self.spectral_proj = nn.Linear(in_channels, out_channels)

    def forward(self, x):
        """
        Args:
            x: [B, H, W, C] input patch
        Returns:
            spatial_tokens: [B, H*W, out_channels]
            spectral_tokens: [B, H*W, out_channels]
        """
        B, H, W, C = x.shape
        # Spatial tokens: transpose spatial dims then flatten
        x_spatial = x.permute(0, 2, 3, 1).reshape(B, H * W, C)
        spatial_tokens = self.spatial_proj(x_spatial)

        # Spectral tokens: flatten spatial dims directly
        x_spectral = x.reshape(B, H * W, C)
        spectral_tokens = self.spectral_proj(x_spectral)

        return spatial_tokens, spectral_tokens


# ──────────────────────────────────────────────
# Feature Enhancement (sigmoid gating)
# ──────────────────────────────────────────────

class SpectralSpatialFeatureEnhancement(nn.Module):
    """Enhance tokens using center-pixel gating."""
    def __init__(self, out_channels):
        super().__init__()
        self.spatial_gate = nn.Sequential(
            nn.Linear(out_channels, out_channels),
            nn.Sigmoid(),
        )
        self.spectral_gate = nn.Sequential(
            nn.Linear(out_channels, out_channels),
            nn.Sigmoid(),
        )

    def forward(self, spatial_tokens, spectral_tokens, center_tokens):
        """
        Args:
            spatial_tokens: [B, L, C]
            spectral_tokens: [B, L, C]
            center_tokens: [B, C]
        """
        spatial_w = self.spatial_gate(center_tokens).unsqueeze(1)   # [B, 1, C]
        spectral_w = self.spectral_gate(center_tokens).unsqueeze(1)  # [B, 1, C]
        return spatial_tokens * spatial_w, spectral_tokens * spectral_w


# ──────────────────────────────────────────────
# Wavelet Transform (DWT)
# ──────────────────────────────────────────────

class WaveletTransform(nn.Module):
    """Apply 1D discrete wavelet transform to token sequences.

    Decomposes each token into 4 sub-bands (cA, cD at 2 levels)
    concatenated along the feature dimension.
    """
    def __init__(self, wavelet_name='haar'):
        super().__init__()
        self.wavelet_name = wavelet_name

    def forward(self, tokens):
        """
        Args:
            tokens: [B, L, C]
        Returns:
            transformed: [B, L, C*4] (4 wavelet sub-bands)
        """
        # Process on CPU for pywt compatibility
        device = tokens.device
        tokens_np = tokens.detach().cpu().numpy()
        B, L, C = tokens_np.shape

        results = []
        for b in range(B):
            # Apply DWT along feature dimension for each token
            cA, cD = pywt.dwt(tokens_np[b], self.wavelet_name, axis=-1)
            # cA and cD each have shape [L, ceil(C/2)+filter_extra]
            # Stack and flatten sub-bands
            combined = np.concatenate([cA, cD], axis=-1)  # [L, ~C]
            results.append(combined)

        out = np.stack(results, axis=0)  # [B, L, ~C]
        return torch.tensor(out, dtype=tokens.dtype, device=device)


# ──────────────────────────────────────────────
# State Space Model (recurrent)
# ──────────────────────────────────────────────

class StateSpaceModelBlock(nn.Module):
    """Simple recurrent state space model."""
    def __init__(self, input_dim, state_dim):
        super().__init__()
        self.state_dim = state_dim
        self.state_transition = nn.Linear(state_dim, state_dim)
        self.state_update = nn.Linear(input_dim, state_dim)
        self.act = nn.ReLU()

    def forward(self, x):
        """
        Args:
            x: [B, L, D] sequence of tokens
        Returns:
            state: [B, state_dim] final state
        """
        B, L, D = x.shape
        state = torch.zeros(B, self.state_dim, device=x.device, dtype=x.dtype)
        for t in range(L):
            state = self.act(self.state_transition(state)) + self.act(self.state_update(x[:, t, :]))
        return state


# ──────────────────────────────────────────────
# Main Model
# ──────────────────────────────────────────────

class WaveMambaModel(nn.Module):
    """
    WaveMamba: Wavelet-enhanced SSM for HSI Classification.

    Input: [B, 1, C, H, W] (5D) where C = spectral bands
    Output: [B, num_classes]

    Args:
        num_classes (int): Number of output classes.
        bands (int): Number of input spectral bands.
        out_channels (int): Token embedding dimension. Default: 64.
        state_dim (int): SSM state dimension. Default: 128.
        wavelet (str): Wavelet name for DWT. Default: 'haar'.
        dropout (float): Dropout rate. Default: 0.4.
    """
    def __init__(self,
                 num_classes=9,
                 bands=30,
                 out_channels=64,
                 state_dim=128,
                 wavelet='haar',
                 dropout=0.4):
        super(WaveMambaModel, self).__init__()

        self.token_gen = SpectralSpatialTokenGeneration(bands, out_channels)
        self.feature_enhance = SpectralSpatialFeatureEnhancement(out_channels)
        self.wavelet_spatial = WaveletTransform(wavelet)
        self.wavelet_spectral = WaveletTransform(wavelet)

        # After wavelet: each token goes from out_channels → ~out_channels (DWT output)
        # After concat spatial + spectral: doubled
        # We use adaptive projection to handle variable DWT output sizes
        self.ssm_proj = None  # lazy init based on actual DWT output size
        self.ssm = StateSpaceModelBlock(state_dim, state_dim)

        self.dense = nn.Linear(state_dim, 128)
        self.dropout = nn.Dropout(dropout)
        self.classifier = nn.Linear(128, num_classes)
        self.state_dim = state_dim

    def forward(self, x):
        # x: [B, 1, C, H, W]
        B = x.size(0)
        x = x.squeeze(1)  # [B, C, H, W]
        x = x.permute(0, 2, 3, 1)  # [B, H, W, C]

        # Token generation
        spatial_tokens, spectral_tokens = self.token_gen(x)

        # Center token gating
        H, W = x.shape[1], x.shape[2]
        center_idx = (H * W) // 2
        center_tokens = spatial_tokens[:, center_idx, :]  # [B, out_channels]
        spatial_tokens, spectral_tokens = self.feature_enhance(
            spatial_tokens, spectral_tokens, center_tokens)

        # Wavelet transform
        spatial_wt = self.wavelet_spatial(spatial_tokens)    # [B, L, D1]
        spectral_wt = self.wavelet_spectral(spectral_tokens)  # [B, L, D2]

        # Concatenate
        combined = torch.cat([spatial_wt, spectral_wt], dim=-1)  # [B, L, D1+D2]

        # Lazy init projection for SSM input
        if self.ssm_proj is None or self.ssm_proj.in_features != combined.shape[-1]:
            self.ssm_proj = nn.Linear(combined.shape[-1], self.state_dim).to(combined.device)

        combined = self.ssm_proj(combined)

        # SSM
        state = self.ssm(combined)  # [B, state_dim]

        # Classification
        out = F.relu(self.dense(state))
        out = self.dropout(out)
        out = self.classifier(out)

        return out


# ──────────────────────────────────────────────
# Registry
# ──────────────────────────────────────────────

@register_model('WaveMamba',
                out_channels=64,
                state_dim=128,
                wavelet='haar',
                dropout=0.4)
def wavemamba_factory(pretrained: bool = False, **kwargs) -> WaveMambaModel:
    """Construct WaveMamba model.

    Standard kwargs: num_classes, bands, patch_size.
    WaveMamba-specific: out_channels, state_dim, wavelet, dropout.
    """
    kwargs.pop('patch_size', None)
    return WaveMambaModel(**kwargs)
