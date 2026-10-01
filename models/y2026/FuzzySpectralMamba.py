from ..registry import register_model
"""
Fuzzy Spectral Mamba (FSM) for Hyperspectral Image Classification
Combines:
- Fuzzy Gaussian Membership Learning
- Frequency-domain Global Enhancement (from FAHM)
- Spectral-Spatial Mamba Blocks (inspired by MambaHSI)
- Adaptive Fuzzy-Spectral Fusion

Date: 2025
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange, repeat
from typing import Tuple, Optional
import math

try:
    from mamba_ssm import Mamba
    MAMBA_AVAILABLE = True
except ImportError:
    MAMBA_AVAILABLE = False
    print("Warning: mamba-ssm not installed. Install with: pip install mamba-ssm")


# ==================== Fuzzy Components ====================

class FuzzyGaussianMembership(nn.Module):
    """Learnable fuzzy Gaussian membership functions for spectral uncertainty modeling."""
    def __init__(self, spectral_dim: int, membership_count: int = 9):
        super().__init__()
        self.spectral_dim = spectral_dim
        self.membership_count = membership_count
        
        # Learnable parameters for Gaussian membership functions
        self.mu = nn.Parameter(torch.randn(spectral_dim, membership_count))
        self.sigma = nn.Parameter(torch.ones(spectral_dim, membership_count) * 0.5)
        
        # Batch normalization for stability
        self.norm = nn.LayerNorm(spectral_dim)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: [B, N, C] spectral features
        Returns:
            fuzzy_x: [B, N, C] fuzzy-enhanced features
        """
        x_expanded = x.unsqueeze(-1)  # [B, N, C, 1]
        mu_expanded = self.mu.unsqueeze(0).unsqueeze(0)  # [1, 1, C, K]
        sigma_expanded = self.sigma.unsqueeze(0).unsqueeze(0)  # [1, 1, C, K]
        
        # Gaussian membership: exp(-((x - mu) / sigma)^2)
        membership = torch.exp(-((x_expanded - mu_expanded) / (sigma_expanded.abs() + 1e-6)) ** 2)
        
        # Aggregate memberships: weighted sum
        weights = F.softmax(membership, dim=-1)
        fuzzy_enhanced = (x_expanded * weights).sum(dim=-1)
        
        return self.norm(fuzzy_enhanced)
    
    
class FuzzySpectralAttention(nn.Module):
    """Fuzzy attention mechanism for spectral channels."""
    def __init__(self, channels: int, reduction: int = 16):
        super().__init__()
        self.channels = channels
        
        # Fuzzy membership for attention weights
        self.fuzzy_membership = FuzzyGaussianMembership(channels, membership_count=4)
        
        # Channel attention layers
        self.fc1 = nn.Linear(channels, channels // reduction)
        self.fc2 = nn.Linear(channels // reduction, channels)
        self.relu = nn.ReLU()
        self.sigmoid = nn.Sigmoid()
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: [B, N, C] features
        Returns:
            attended: [B, N, C]
        """
        # Global average pooling
        z = x.mean(dim=1, keepdim=True)  # [B, 1, C]
        
        # Apply fuzzy membership
        z_fuzzy = self.fuzzy_membership(z)  # [B, 1, C]
        
        # Channel attention
        attn = self.fc1(z_fuzzy)
        attn = self.relu(attn)
        attn = self.fc2(attn)
        attn = self.sigmoid(attn)
        
        return x * attn


# ==================== Frequency Components ====================

class FuzzyFrequencyGlobalEnhancement(nn.Module):
    """
    Fuzzy-enhanced frequency domain processing.
    Combines FGEM from FAHM with fuzzy membership learning.
    """
    def __init__(self, in_chans: int, embed_dim: int, n_groups: int = 4, 
                 sparsity_threshold: float = 0.01, fuzzy_memberships: int = 5):
        super().__init__()
        assert in_chans % n_groups == 0, f"in_chans {in_chans} must be divisible by n_groups {n_groups}"
        
        self.n_groups = n_groups
        self.in_chans = in_chans
        self.block_size = in_chans // n_groups
        self.sparsity_threshold = sparsity_threshold
        self.scale = 0.02
        
        # Frequency domain learnable weights (complex-valued operations)
        self.w_real = nn.Parameter(self.scale * torch.randn(n_groups, self.block_size, self.block_size))
        self.w_imag = nn.Parameter(self.scale * torch.randn(n_groups, self.block_size, self.block_size))
        self.b_real = nn.Parameter(self.scale * torch.randn(n_groups, self.block_size))
        self.b_imag = nn.Parameter(self.scale * torch.randn(n_groups, self.block_size))
        
        # Fuzzy membership learning in frequency domain
        self.fuzzy_freq = FuzzyGaussianMembership(in_chans, fuzzy_memberships)
        
        # Projection and normalization
        self.proj = nn.Linear(in_chans, embed_dim)
        self.norm = nn.LayerNorm(embed_dim)
        self.act = nn.GELU()
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: [B, C, H, W] input features
        Returns:
            out: [B, N, D] where N=H*W, D=embed_dim
        """
        B, C, H, W = x.shape
        dtype = x.dtype
        
        # 2D FFT transform to frequency domain
        x_freq = torch.fft.rfft2(x.float(), dim=(2, 3), norm="ortho")
        origin_freq = x_freq
        
        # Reshape for grouped processing
        x_freq = x_freq.reshape(B, self.n_groups, self.block_size, x_freq.shape[2], x_freq.shape[3])
        
        # Complex-valued linear transformation
        o_real_1 = torch.einsum('bkihw,kio->bkohw', x_freq.real, self.w_real)
        o_real_2 = torch.einsum('bkihw,kio->bkohw', x_freq.imag, self.w_imag)
        o_real = self.act(o_real_1 - o_real_2 + self.b_real[:, :, None, None]) + x_freq.real
        
        o_imag_1 = torch.einsum('bkihw,kio->bkohw', x_freq.imag, self.w_real)
        o_imag_2 = torch.einsum('bkihw,kio->bkohw', x_freq.real, self.w_imag)
        o_imag = self.act(o_imag_1 + o_imag_2 + self.b_imag[:, :, None, None]) + x_freq.imag
        
        # Stack and apply sparsity (fuzzy soft-thresholding)
        x_freq = torch.stack([o_real, o_imag], dim=-1)
        x_freq = F.softshrink(x_freq, lambd=self.sparsity_threshold)
        x_freq = torch.view_as_complex(x_freq)
        
        # Reshape back
        x_freq = x_freq.reshape(B, C, x_freq.shape[3], x_freq.shape[4])
        
        # Residual connection in frequency domain
        x_freq = x_freq + origin_freq
        
        # Inverse FFT back to spatial domain
        x_spatial = torch.fft.irfft2(x_freq, s=(H, W), dim=(2, 3), norm="ortho")
        x_spatial = x_spatial.type(dtype)
        
        # Flatten to sequence: [B, C, H, W] -> [B, H*W, C]
        x_seq = x_spatial.flatten(2).transpose(1, 2)  # [B, N, C]
        
        # Apply fuzzy membership learning
        x_fuzzy = self.fuzzy_freq(x_seq)  # [B, N, C]
        
        # Project to embedding dimension
        x_embed = self.proj(x_fuzzy)  # [B, N, D]
        x_embed = self.norm(self.act(x_embed))
        
        return x_embed


# ==================== Spectral Mamba Components ====================

class SpectralMambaBlock(nn.Module):
    """
    Spectral Mamba block with fuzzy enhancement.
    Processes spectral sequences with state space models.
    """
    def __init__(self, dim: int, state_dim: int = 16, n_groups: int = 4, dropout: float = 0.1):
        super().__init__()
        
        if not MAMBA_AVAILABLE:
            raise ImportError("mamba-ssm required")
        
        self.dim = dim
        self.n_groups = n_groups
        assert dim % n_groups == 0, f"dim {dim} must be divisible by n_groups {n_groups}"
        self.group_dim = dim // n_groups
        
        # Fuzzy spectral attention
        self.fuzzy_attn = FuzzySpectralAttention(dim)
        
        # Group normalization and Mamba for spectral groups
        self.norm1 = nn.LayerNorm(dim)
        self.spectral_mamba = Mamba(d_model=self.group_dim, d_state=state_dim, d_conv=4, expand=2)
        
        # FFN
        self.norm2 = nn.LayerNorm(dim)
        self.ffn = nn.Sequential(
            nn.Linear(dim, dim * 4),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(dim * 4, dim),
            nn.Dropout(dropout)
        )
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: [B, N, D] features
        Returns:
            out: [B, N, D]
        """
        B, N, D = x.shape
        
        # Apply fuzzy spectral attention
        x_fuzzy = self.fuzzy_attn(x)
        
        # Normalize
        x_norm = self.norm1(x_fuzzy)
        
        # Reshape for grouped spectral processing
        x_grouped = x_norm.reshape(B * N, self.n_groups, self.group_dim)
        
        # Process each spectral group with Mamba
        x_mamba = self.spectral_mamba(x_grouped)  # [B*N, G, D/G]
        
        # Reshape back
        x_mamba = x_mamba.reshape(B, N, D)
        
        # Residual connection
        x = x + x_mamba
        
        # FFN with residual
        x = x + self.ffn(self.norm2(x))
        
        return x


class SpatialMambaBlock(nn.Module):
    """
    Spatial Mamba block for long-range spatial dependencies.
    """
    def __init__(self, dim: int, state_dim: int = 16, dropout: float = 0.1):
        super().__init__()
        
        if not MAMBA_AVAILABLE:
            raise ImportError("mamba-ssm required")
        
        self.norm1 = nn.LayerNorm(dim)
        self.spatial_mamba = Mamba(d_model=dim, d_state=state_dim, d_conv=4, expand=2)
        
        self.norm2 = nn.LayerNorm(dim)
        self.ffn = nn.Sequential(
            nn.Linear(dim, dim * 4),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(dim * 4, dim),
            nn.Dropout(dropout)
        )
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: [B, N, D] features
        Returns:
            out: [B, N, D]
        """
        # Mamba with residual
        x = x + self.spatial_mamba(self.norm1(x))
        
        # FFN with residual
        x = x + self.ffn(self.norm2(x))
        
        return x


# ==================== Fusion Module ====================

class FuzzySpatialSpectralFusion(nn.Module):
    """
    Adaptive fusion of spatial and spectral features with fuzzy weighting.
    """
    def __init__(self, dim: int):
        super().__init__()
        self.dim = dim
        
        # Learnable fuzzy fusion weights
        self.spatial_weight = nn.Parameter(torch.ones(1))
        self.spectral_weight = nn.Parameter(torch.ones(1))
        
        # Attention-based fusion
        self.fusion_attn = nn.Sequential(
            nn.Linear(dim * 2, dim),
            nn.GELU(),
            nn.Linear(dim, 2),
            nn.Softmax(dim=-1)
        )
        
        self.norm = nn.LayerNorm(dim)
        
    def forward(self, spatial_feat: torch.Tensor, spectral_feat: torch.Tensor, 
                residual: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Args:
            spatial_feat: [B, N, D] spatial features
            spectral_feat: [B, N, D] spectral features
            residual: [B, N, D] optional residual connection
        Returns:
            fused: [B, N, D]
        """
        # Concatenate for attention computation
        concat = torch.cat([spatial_feat, spectral_feat], dim=-1)  # [B, N, 2D]
        
        # Compute adaptive fusion weights
        attn_weights = self.fusion_attn(concat.mean(dim=1))  # [B, 2]
        w_spatial = attn_weights[:, 0:1].unsqueeze(1)  # [B, 1, 1]
        w_spectral = attn_weights[:, 1:2].unsqueeze(1)  # [B, 1, 1]
        
        # Fuzzy weighted fusion
        fused = w_spatial * spatial_feat + w_spectral * spectral_feat
        
        # Add residual connection if provided
        if residual is not None:
            fused = fused + residual
        
        return self.norm(fused)


# ==================== Main Model ====================

class FuzzySpectralMamba(nn.Module):
    """
    Fuzzy Spectral Mamba (FSM) for Hyperspectral Image Classification.
    
    Architecture:
    1. Fuzzy Frequency Global Enhancement (embedding layer)
    2. Multi-stage Spatial-Spectral Mamba blocks
    3. Adaptive Fuzzy Fusion
    4. Classification head
    """
    def __init__(self,
                 in_channels: int = 48,
                 patch_size: int = 11,
                 num_classes: int = 20,
                 embed_dims: list = [256, 128, 64],
                 depths: list = [2, 2, 2],
                 n_groups: list = [32, 16, 8],
                 state_dim: int = 16,
                 dropout: float = 0.1,
                 sparsity_threshold: float = 0.01):
        super().__init__()
        
        if not MAMBA_AVAILABLE:
            raise ImportError("mamba-ssm required. Install with: pip install mamba-ssm")
        
        self.in_channels = in_channels
        self.patch_size = patch_size
        self.num_classes = num_classes
        self.num_stages = len(embed_dims)
        self.embed_dims = embed_dims
        self.depths = depths
        
        # Pad input channels if needed
        new_bands = math.ceil(in_channels / n_groups[0]) * n_groups[0]
        self.pad = nn.ReplicationPad3d((0, 0, 0, 0, 0, new_bands - in_channels))
        
        # Multi-stage fuzzy frequency embeddings
        self.embeddings = nn.ModuleList()
        for i in range(self.num_stages):
            embed = FuzzyFrequencyGlobalEnhancement(
                in_chans=new_bands if i == 0 else embed_dims[i-1],
                embed_dim=embed_dims[i],
                n_groups=n_groups[i],
                sparsity_threshold=sparsity_threshold,
                fuzzy_memberships=5
            )
            self.embeddings.append(embed)
        
        # Multi-stage Spatial-Spectral Mamba encoders
        self.spatial_blocks = nn.ModuleList()
        self.spectral_blocks = nn.ModuleList()
        self.fusion_modules = nn.ModuleList()
        
        for stage_idx in range(self.num_stages):
            spatial_stage = nn.ModuleList([
                SpatialMambaBlock(embed_dims[stage_idx], state_dim, dropout)
                for _ in range(depths[stage_idx])
            ])
            spectral_stage = nn.ModuleList([
                SpectralMambaBlock(embed_dims[stage_idx], state_dim, n_groups[stage_idx], dropout)
                for _ in range(depths[stage_idx])
            ])
            
            self.spatial_blocks.append(spatial_stage)
            self.spectral_blocks.append(spectral_stage)
            self.fusion_modules.append(FuzzySpatialSpectralFusion(embed_dims[stage_idx]))
        
        # Channel merge to final dimension
        self.channel_merges = nn.ModuleList([
            nn.Linear(embed_dims[i], embed_dims[-1])
            for i in range(self.num_stages)
        ])
        
        # Multi-scale fusion weights
        self.scale_weights = nn.Parameter(torch.ones(self.num_stages))
        
        # Classification head
        self.head = nn.Sequential(
            nn.LayerNorm(embed_dims[-1]),
            nn.Linear(embed_dims[-1], num_classes)
        )
        
        self._encoder_adjusted = False
        
    def _maybe_adjust_input(self, x: torch.Tensor):
        """Adjust model if input channels don't match."""
        if x.shape[1] != self.in_channels and not self._encoder_adjusted:
            pass
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: [B, C, H, W] or [B, 1, C, H, W] input HSI patch
        Returns:
            logits: [B, num_classes]
        """
        # Handle 5D input
        if x.dim() == 5 and x.size(1) == 1:
            x = x.squeeze(1)
        
        self._maybe_adjust_input(x)
        
        B, C, H, W = x.shape
        
        # Pad channels if necessary
        if C != self.pad.padding[5]:
            x = x.unsqueeze(1)  # [B, 1, C, H, W]
            x = self.pad(x).squeeze(1)  # [B, C', H, W]
        
        # Multi-stage processing
        multi_scale_features = []
        
        for stage_idx in range(self.num_stages):
            if stage_idx == 0:
                stage_input = x
            else:
                prev_feat = multi_scale_features[-1]  # [B, N, D]
                stage_input = prev_feat.transpose(1, 2).reshape(B, self.embed_dims[stage_idx-1], H, W)
            
            x_embed = self.embeddings[stage_idx](stage_input)  # [B, N, D]
            
            spatial_feat = x_embed
            spectral_feat = x_embed
            
            for depth_idx in range(self.depths[stage_idx]):
                spatial_feat = self.spatial_blocks[stage_idx][depth_idx](spatial_feat)
                spectral_feat = self.spectral_blocks[stage_idx][depth_idx](spectral_feat)
            
            fused_feat = self.fusion_modules[stage_idx](spatial_feat, spectral_feat, residual=x_embed)
            multi_scale_features.append(fused_feat)
        
        # Multi-scale feature fusion
        merged_features = []
        for stage_idx in range(self.num_stages):
            merged = self.channel_merges[stage_idx](multi_scale_features[stage_idx])
            merged_features.append(merged)
        
        scale_weights = F.softmax(self.scale_weights, dim=0)
        final_features = sum(w * feat for w, feat in zip(scale_weights, merged_features))
        
        # Global average pooling
        final_features = final_features.mean(dim=1)  # [B, D]
        
        # Classification
        logits = self.head(final_features)
        
        return logits


# # ==================== Model Registration ====================

# @register_model('FuzzySpectralMamba-111', expects_4d=True, num_stages=3, embed_dims=[256, 128, 64], 
#                 depths=[1,2,1], n_groups=[32, 16, 8], state_dim=16)
# def proposed(pretrained: bool = False, **kwargs) -> FuzzySpectralMamba:
#     """Constructs a Fuzzy Spectral Mamba model."""
    
#     # Map standardized names
#     if 'bands' in kwargs:
#         kwargs['in_channels'] = kwargs.pop('bands')
#     if 'img_size' in kwargs:
#         kwargs['patch_size'] = kwargs.pop('img_size')
    
#     # Remove registry metadata
#     for key in ['num_stages', 'expects_4d', 'dual_input']:
#         kwargs.pop(key, None)
    
#     defaults = dict(
#         in_channels=48,
#         patch_size=11,
#         num_classes=20,
#         embed_dims=[256, 128, 64],
#         depths=[2, 2, 2],
#         n_groups=[32, 16, 8],
#         state_dim=16,
#         dropout=0.1,
#         sparsity_threshold=0.01
#     )
    
#     config = {**defaults, **kwargs}
#     return FuzzySpectralMamba(**config)

# @register_model('FuzzySpectralMamba-121', expects_4d=True, num_stages=3, embed_dims=[256, 128, 64], 
#                 depths=[1,2,1], n_groups=[32, 16, 8], state_dim=16)
# def proposed(pretrained: bool = False, **kwargs) -> FuzzySpectralMamba:
#     """Constructs a Fuzzy Spectral Mamba model."""
    
#     # Map standardized names
#     if 'bands' in kwargs:
#         kwargs['in_channels'] = kwargs.pop('bands')
#     if 'img_size' in kwargs:
#         kwargs['patch_size'] = kwargs.pop('img_size')
    
#     # Remove registry metadata
#     for key in ['num_stages', 'expects_4d', 'dual_input']:
#         kwargs.pop(key, None)
    
#     defaults = dict(
#         in_channels=48,
#         patch_size=11,
#         num_classes=20,
#         embed_dims=[256, 128, 64],
#         depths=[2, 2, 2],
#         n_groups=[32, 16, 8],
#         state_dim=16,
#         dropout=0.1,
#         sparsity_threshold=0.01
#     )
    
#     config = {**defaults, **kwargs}
#     return FuzzySpectralMamba(**config)

# @register_model('FuzzySpectralMamba-122', expects_4d=True, num_stages=3, embed_dims=[256, 128, 64], 
#                 depths=[1,2,2], n_groups=[32, 16, 8], state_dim=16)
# def proposed(pretrained: bool = False, **kwargs) -> FuzzySpectralMamba:
#     """Constructs a Fuzzy Spectral Mamba model."""
    
#     # Map standardized names
#     if 'bands' in kwargs:
#         kwargs['in_channels'] = kwargs.pop('bands')
#     if 'img_size' in kwargs:
#         kwargs['patch_size'] = kwargs.pop('img_size')
    
#     # Remove registry metadata
#     for key in ['num_stages', 'expects_4d', 'dual_input']:
#         kwargs.pop(key, None)
    
#     defaults = dict(
#         in_channels=48,
#         patch_size=11,
#         num_classes=20,
#         embed_dims=[256, 128, 64],
#         depths=[2, 2, 2],
#         n_groups=[32, 16, 8],
#         state_dim=16,
#         dropout=0.1,
#         sparsity_threshold=0.01
#     )
    
#     config = {**defaults, **kwargs}
#     return FuzzySpectralMamba(**config)


# @register_model('FuzzySpectralMamba-211', expects_4d=True, num_stages=3, embed_dims=[256, 128, 64], 
#                 depths=[2,1,1], n_groups=[32, 16, 8], state_dim=16)
# def proposed(pretrained: bool = False, **kwargs) -> FuzzySpectralMamba:
#     """Constructs a Fuzzy Spectral Mamba model."""
    
#     # Map standardized names
#     if 'bands' in kwargs:
#         kwargs['in_channels'] = kwargs.pop('bands')
#     if 'img_size' in kwargs:
#         kwargs['patch_size'] = kwargs.pop('img_size')
    
#     # Remove registry metadata
#     for key in ['num_stages', 'expects_4d', 'dual_input']:
#         kwargs.pop(key, None)
    
#     defaults = dict(
#         in_channels=48,
#         patch_size=11,
#         num_classes=20,
#         embed_dims=[256, 128, 64],
#         depths=[2, 2, 2],
#         n_groups=[32, 16, 8],
#         state_dim=16,
#         dropout=0.1,
#         sparsity_threshold=0.01
#     )
    
#     config = {**defaults, **kwargs}
#     return FuzzySpectralMamba(**config)


# @register_model('FuzzySpectralMamba-221', expects_4d=True, num_stages=3, embed_dims=[256, 128, 64], 
#                 depths=[2,2,1], n_groups=[32, 16, 8], state_dim=16)
# def proposed(pretrained: bool = False, **kwargs) -> FuzzySpectralMamba:
#     """Constructs a Fuzzy Spectral Mamba model."""
    
#     # Map standardized names
#     if 'bands' in kwargs:
#         kwargs['in_channels'] = kwargs.pop('bands')
#     if 'img_size' in kwargs:
#         kwargs['patch_size'] = kwargs.pop('img_size')
    
#     # Remove registry metadata
#     for key in ['num_stages', 'expects_4d', 'dual_input']:
#         kwargs.pop(key, None)
    
#     defaults = dict(
#         in_channels=48,
#         patch_size=11,
#         num_classes=20,
#         embed_dims=[256, 128, 64],
#         depths=[2, 2, 2],
#         n_groups=[32, 16, 8],
#         state_dim=16,
#         dropout=0.1,
#         sparsity_threshold=0.01
#     )
    
#     config = {**defaults, **kwargs}
#     return FuzzySpectralMamba(**config)


# @register_model('FuzzySpectralMamba-222', expects_4d=True, num_stages=3, embed_dims=[256, 128, 64], 
#                 depths=[2,2,2], n_groups=[32, 16, 8], state_dim=16)
# def proposed(pretrained: bool = False, **kwargs) -> FuzzySpectralMamba:
#     """Constructs a Fuzzy Spectral Mamba model."""
    
#     # Map standardized names
#     if 'bands' in kwargs:
#         kwargs['in_channels'] = kwargs.pop('bands')
#     if 'img_size' in kwargs:
#         kwargs['patch_size'] = kwargs.pop('img_size')
    
#     # Remove registry metadata
#     for key in ['num_stages', 'expects_4d', 'dual_input']:
#         kwargs.pop(key, None)
    
#     defaults = dict(
#         in_channels=48,
#         patch_size=11,
#         num_classes=20,
#         embed_dims=[256, 128, 64],
#         depths=[2, 2, 2],
#         n_groups=[32, 16, 8],
#         state_dim=16,
#         dropout=0.1,
#         sparsity_threshold=0.01
#     )
    
#     config = {**defaults, **kwargs}
#     return FuzzySpectralMamba(**config)

# @register_model('FuzzySpectralMamba-212', expects_4d=True, num_stages=3, embed_dims=[256, 128, 64], 
#                 depths=[2,1,2], n_groups=[32, 16, 8], state_dim=16)
# def proposed(pretrained: bool = False, **kwargs) -> FuzzySpectralMamba:
#     """Constructs a Fuzzy Spectral Mamba model."""
    
#     # Map standardized names
#     if 'bands' in kwargs:
#         kwargs['in_channels'] = kwargs.pop('bands')
#     if 'img_size' in kwargs:
#         kwargs['patch_size'] = kwargs.pop('img_size')
    
#     # Remove registry metadata
#     for key in ['num_stages', 'expects_4d', 'dual_input']:
#         kwargs.pop(key, None)
    
#     defaults = dict(
#         in_channels=48,
#         patch_size=11,
#         num_classes=20,
#         embed_dims=[256, 128, 64],
#         depths=[2, 2, 2],
#         n_groups=[32, 16, 8],
#         state_dim=16,
#         dropout=0.1,
#         sparsity_threshold=0.01
#     )
    
#     config = {**defaults, **kwargs}
#     return FuzzySpectralMamba(**config)


# @register_model('FuzzySpectralMamba-112', expects_4d=True, num_stages=3, embed_dims=[256, 128, 64], 
#                 depths=[1,1,2], n_groups=[32, 16, 8], state_dim=16)
# def proposed(pretrained: bool = False, **kwargs) -> FuzzySpectralMamba:
#     """Constructs a Fuzzy Spectral Mamba model."""
    
#     # Map standardized names
#     if 'bands' in kwargs:
#         kwargs['in_channels'] = kwargs.pop('bands')
#     if 'img_size' in kwargs:
#         kwargs['patch_size'] = kwargs.pop('img_size')
    
#     # Remove registry metadata
#     for key in ['num_stages', 'expects_4d', 'dual_input']:
#         kwargs.pop(key, None)
    
#     defaults = dict(
#         in_channels=48,
#         patch_size=11,
#         num_classes=20,
#         embed_dims=[256, 128, 64],
#         depths=[2, 2, 2],
#         n_groups=[32, 16, 8],
#         state_dim=16,
#         dropout=0.1,
#         sparsity_threshold=0.01
#     )
    
#     config = {**defaults, **kwargs}
#     return FuzzySpectralMamba(**config)

@register_model('FuzzySpectralMamba', expects_4d=True, num_stages=3, embed_dims=[256, 128, 64], 
                depths=[2,2,2], n_groups=[32, 16, 8], state_dim=16)
def proposed(pretrained: bool = False, **kwargs) -> FuzzySpectralMamba:
    """Constructs a Fuzzy Spectral Mamba model."""
    
    # Map standardized names
    if 'bands' in kwargs:
        kwargs['in_channels'] = kwargs.pop('bands')
    if 'img_size' in kwargs:
        kwargs['patch_size'] = kwargs.pop('img_size')
    
    # Remove registry metadata
    for key in ['num_stages', 'expects_4d', 'dual_input']:
        kwargs.pop(key, None)
    
    defaults = dict(
        in_channels=48,
        patch_size=11,
        num_classes=20,
        embed_dims=[256, 128, 64],
        depths=[2, 2, 2],
        n_groups=[32, 16, 8],
        state_dim=16,
        dropout=0.1,
        sparsity_threshold=0.01
    )
    
    config = {**defaults, **kwargs}
    return FuzzySpectralMamba(**config)





# ==================== Testing ====================
