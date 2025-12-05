"""
Adaptive Morphological Memory Transformer (AMMT) for Hyperspectral Image Classification

Authors: Novel Architecture Design
Date: 2025

Three Core Innovations:
1. Learnable Morphological Priors - Addresses small-sample efficiency
2. Structural Memory Bank - Addresses boundary confusion  
3. Sample-Adaptive Center Weighting - Addresses spectral-spatial trade-off

Model Statistics:
- Parameters: 2.13M (trainable: 2.08M)
- MACs per sample: 321M
- Model size: 8.51 MB (FP32)
- Input: [B, 200, 11, 11] for Indian Pines dataset
- Output: [B, 16] class logits

Usage:
    model = AMMT(in_channels=200, patch_size=11, num_classes=16, 
                 embed_dim=128, depth=6, num_heads=8)

    x = torch.randn(8, 200, 11, 11)  # Batch of HSI patches
    logits = model(x)  # [8, 16]
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple
import math
from ..registry import register_model


class LearnableMorphology(nn.Module):
    """
    Learnable morphological operations with trainable structuring elements.
    Addresses: Small-sample efficiency via data-efficient structural priors.
    """
    def __init__(self, in_channels: int, kernel_size: int = 3):
        super().__init__()
        self.kernel_size = kernel_size
        self.in_channels = in_channels

        # Learnable structuring elements: initialized with cross/diamond patterns
        self.structure_erosion = nn.Parameter(torch.zeros(in_channels, 1, kernel_size, kernel_size))
        self.structure_dilation = nn.Parameter(torch.zeros(in_channels, 1, kernel_size, kernel_size))
        self._initialize_structures()

    def _initialize_structures(self):
        """Initialize with geometric patterns (cross for erosion, diamond for dilation)"""
        k = self.kernel_size
        center = k // 2

        with torch.no_grad():
            for i in range(self.in_channels):
                # Cross pattern for erosion
                self.structure_erosion.data[i, 0, center, :] = 1.0
                self.structure_erosion.data[i, 0, :, center] = 1.0

                # Diamond pattern for dilation
                for x in range(k):
                    for y in range(k):
                        if abs(x - center) + abs(y - center) <= center:
                            self.structure_dilation.data[i, 0, x, y] = 1.0

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Args: x [B, C, H, W]
        Returns: (erosion, dilation) each [B, C, H, W]
        """
        padding = self.kernel_size // 2

        # Erosion via depthwise conv
        erosion = F.conv2d(x, -self.structure_erosion, padding=padding, groups=self.in_channels)
        erosion = -erosion

        # Dilation via depthwise conv
        dilation = F.conv2d(x, self.structure_dilation, padding=padding, groups=self.in_channels)

        return erosion, dilation


class StructuralMemoryBank(nn.Module):
    """
    Maintains evolving memory of morphologically-processed features.
    Addresses: Boundary confusion via cross-sample structural regularization.
    """
    def __init__(self, memory_size: int, feature_dim: int, momentum: float = 0.9):
        super().__init__()
        self.memory_size = memory_size
        self.feature_dim = feature_dim
        self.momentum = momentum

        # Memory bank: [memory_size, feature_dim]
        self.register_buffer('memory', torch.randn(memory_size, feature_dim))
        self.register_buffer('ptr', torch.zeros(1, dtype=torch.long))

        # Memory projection
        self.memory_proj = nn.Linear(feature_dim, feature_dim)

    def forward(self, features: torch.Tensor, update: bool = True) -> torch.Tensor:
        """
        Args:
            features: [B, N, D] token features
            update: whether to update memory (True in training)
        Returns:
            memory_enhanced: [B, N, D]
        """
        B, N, D = features.shape

        # Clone memory to avoid inplace modification issues with autograd
        memory_clone = self.memory.clone()
        
        # Query memory
        memory = self.memory_proj(memory_clone)  # [M, D]

        # Attention: features attend to memory
        attn_scores = torch.matmul(features, memory.T) / math.sqrt(D)  # [B, N, M]
        attn_weights = F.softmax(attn_scores, dim=-1)
        memory_context = torch.matmul(attn_weights, memory.unsqueeze(0))  # [B, N, D]

        # Fuse with residual
        output = features + memory_context

        # Update memory (FIFO) - use no_grad to avoid inplace operation issues
        if update and self.training:
            with torch.no_grad():
                batch_mean = features.mean(dim=(0, 1)).detach()  # [D]
                ptr = int(self.ptr.item())
                new_memory = self.momentum * self.memory.data[ptr] + (1 - self.momentum) * batch_mean
                self.memory.data[ptr] = new_memory
                self.ptr.data[0] = (ptr + 1) % self.memory_size

        return output


class AdaptiveCenterWeighting(nn.Module):
    """
    Sample-adaptive Gaussian center bias.
    Addresses: Spectral-spatial trade-off via per-sample context weighting.
    """
    def __init__(self, feature_dim: int, spatial_size: int):
        super().__init__()
        self.spatial_size = spatial_size

        # MLP to predict sigma from global features
        self.sigma_predictor = nn.Sequential(
            nn.Linear(feature_dim, feature_dim // 2),
            nn.GELU(),
            nn.Linear(feature_dim // 2, 1),
            nn.Softplus()  # Ensure positive sigma
        )

        # Precompute spatial coordinates
        self.register_buffer('coords', self._get_coordinates(spatial_size))

    def _get_coordinates(self, size: int):
        """Generate 2D coordinates centered at origin"""
        center = size // 2
        y, x = torch.meshgrid(torch.arange(size), torch.arange(size), indexing='ij')
        coords = torch.stack([x - center, y - center], dim=-1).float()  # [H, W, 2]
        return coords

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        """
        Args:
            features: [B, C, H, W]
        Returns:
            weighted: [B, C, H, W]
        """
        B, C, H, W = features.shape

        # Global pooling to get sample-level representation
        global_feat = features.mean(dim=(2, 3))  # [B, C]

        # Predict sigma per sample
        sigma = self.sigma_predictor(global_feat).unsqueeze(-1).unsqueeze(-1)  # [B, 1, 1, 1]

        # Compute Gaussian weights
        dist_sq = (self.coords ** 2).sum(dim=-1)  # [H, W]
        gaussian = torch.exp(-dist_sq / (2 * sigma ** 2 + 1e-6))  # [B, 1, H, W]

        # Apply center weighting
        weighted = features * gaussian

        return weighted


class MorphologicalAttention(nn.Module):
    """
    Multi-head attention operating on morphologically-processed features.
    Decouples erosion (structural) and dilation (contextual) paths.
    """
    def __init__(self, dim: int, num_heads: int = 8, qkv_bias: bool = False):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.scale = self.head_dim ** -0.5

        # Separate projections for erosion and dilation features
        self.qkv_erosion = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.qkv_dilation = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.qkv_spectral = nn.Linear(dim, dim * 3, bias=qkv_bias)

        self.proj = nn.Linear(dim * 3, dim)
        self.fusion_gate = nn.Sequential(
            nn.Linear(dim * 3, 3),
            nn.Softmax(dim=-1)
        )

    def forward(self, x_erosion: torch.Tensor, x_dilation: torch.Tensor, 
                x_spectral: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x_erosion, x_dilation, x_spectral: [B, N, D]
        Returns:
            output: [B, N, D]
        """
        B, N, D = x_erosion.shape

        # Process each stream
        def attn_stream(x, qkv_proj):
            qkv = qkv_proj(x).reshape(B, N, 3, self.num_heads, self.head_dim).permute(2, 0, 3, 1, 4)
            q, k, v = qkv[0], qkv[1], qkv[2]

            attn = (q @ k.transpose(-2, -1)) * self.scale
            attn = F.softmax(attn, dim=-1)
            out = (attn @ v).transpose(1, 2).reshape(B, N, D)
            return out

        out_erosion = attn_stream(x_erosion, self.qkv_erosion)
        out_dilation = attn_stream(x_dilation, self.qkv_dilation)
        out_spectral = attn_stream(x_spectral, self.qkv_spectral)

        # Concatenate and fuse
        concat = torch.cat([out_erosion, out_dilation, out_spectral], dim=-1)

        # Learnable fusion gates
        gates = self.fusion_gate(concat.mean(dim=1, keepdim=True))  # [B, 1, 3]
        fused = (gates[:, :, 0:1] * out_erosion + 
                 gates[:, :, 1:2] * out_dilation + 
                 gates[:, :, 2:3] * out_spectral)

        output = self.proj(concat) + fused
        return output


class AMMTBlock(nn.Module):
    """
    Single AMMT Transformer block combining all three innovations.
    """
    def __init__(self, dim: int, num_heads: int, mlp_ratio: float = 4.0,
                 memory_size: int = 64, spatial_size: int = 11):
        super().__init__()
        self.norm1 = nn.LayerNorm(dim)
        self.norm2 = nn.LayerNorm(dim)

        self.morph_attn = MorphologicalAttention(dim, num_heads)
        self.memory = StructuralMemoryBank(memory_size, dim)

        # FFN
        mlp_hidden = int(dim * mlp_ratio)
        self.mlp = nn.Sequential(
            nn.Linear(dim, mlp_hidden),
            nn.GELU(),
            nn.Linear(mlp_hidden, dim)
        )

    def forward(self, x_erosion: torch.Tensor, x_dilation: torch.Tensor,
                x_original: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x_erosion, x_dilation, x_original: [B, N, D]
        Returns:
            output: [B, N, D]
        """
        # Morphological attention
        attn_out = self.morph_attn(
            self.norm1(x_erosion),
            self.norm1(x_dilation),
            self.norm1(x_original)
        )
        x = x_original + attn_out

        # Memory enhancement
        x = self.memory(x)

        # FFN
        x = x + self.mlp(self.norm2(x))

        return x


class AMMT(nn.Module):
    """
    Adaptive Morphological Memory Transformer for HSI Classification.

    Three core innovations:
    1. Learnable Morphological Priors - addresses small-sample efficiency
    2. Structural Memory Bank - addresses boundary confusion
    3. Sample-Adaptive Center Weighting - addresses spectral-spatial trade-off
    """
    def __init__(self, 
                 in_channels: int = 200,        # Spectral bands (e.g., Indian Pines)
                 patch_size: int = 11,          # Spatial patch size
                 num_classes: int = 16,         # Number of land cover classes
                 embed_dim: int = 128,          # Embedding dimension
                 depth: int = 6,                # Number of transformer blocks
                 num_heads: int = 8,            # Attention heads
                 mlp_ratio: float = 4.0,        # MLP expansion ratio
                 memory_size: int = 64,         # Memory bank size
                 morph_kernel: int = 3):        # Morphological kernel size
        super().__init__()

        self.patch_size = patch_size
        self.num_patches = patch_size * patch_size

        # Patch embedding
        self.patch_embed = nn.Conv2d(in_channels, embed_dim, kernel_size=1)

        # Learnable morphological operations
        self.morphology = LearnableMorphology(embed_dim, morph_kernel)

        # Positional encoding
        self.pos_embed = nn.Parameter(torch.zeros(1, self.num_patches, embed_dim))

        # AMMT blocks
        self.blocks = nn.ModuleList([
            AMMTBlock(embed_dim, num_heads, mlp_ratio, memory_size, patch_size)
            for _ in range(depth)
        ])

        # Adaptive center weighting (applied after morphology)
        self.center_weight = AdaptiveCenterWeighting(embed_dim, patch_size)

        # Classification head
        self.norm = nn.LayerNorm(embed_dim)
        self.head = nn.Linear(embed_dim, num_classes)

        self._init_weights()

    def _init_weights(self):
        nn.init.trunc_normal_(self.pos_embed, std=0.02)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: [B, C, H, W] or [B, 1, C, H, W] where C=spectral bands, H=W=patch_size
        Returns:
            logits: [B, num_classes]
        """
        # # Handle 5D input by squeezing the channel dimension
        # if x.dim() == 5:
        #     x = x.squeeze(1)  # [B, 1, C, H, W] -> [B, C, H, W]
        
        B = x.shape[0]

        # Patch embedding
        x = self.patch_embed(x)  # [B, D, H, W]

        # Apply adaptive center weighting
        x = self.center_weight(x)

        # Morphological processing
        x_erosion, x_dilation = self.morphology(x)  # Both [B, D, H, W]

        # Reshape to sequence
        def to_seq(tensor):
            return tensor.flatten(2).transpose(1, 2)  # [B, D, H, W] -> [B, N, D]

        x_ero_seq = to_seq(x_erosion) + self.pos_embed
        x_dil_seq = to_seq(x_dilation) + self.pos_embed
        x_orig_seq = to_seq(x) + self.pos_embed

        # Transformer blocks
        for block in self.blocks:
            x_orig_seq = block(x_ero_seq, x_dil_seq, x_orig_seq)
            # Update morphological features too
            x_ero_seq = x_orig_seq
            x_dil_seq = x_orig_seq

        # Global average pooling
        x = self.norm(x_orig_seq)
        x = x.mean(dim=1)  # [B, D]

        # Classification
        logits = self.head(x)  # [B, num_classes]

        return logits


# Example usage and model summary
@register_model('AMMT', expects_4d=True, embed_dim=128, depth=6, num_heads=8)
def ammt(pretrained: bool = False, **kwargs) -> AMMT:
    """Constructs an AMMT model."""
    # Map standardized names to model-specific names
    if 'bands' in kwargs:
        kwargs['in_channels'] = kwargs.pop('bands')
    return AMMT(**kwargs)


# if __name__ == "__main__":
#     print("="*100)
#     print("Adaptive Morphological Memory Transformer (AMMT) - Model Summary")
#     print("="*100)

#     # Create model
#     model = AMMT(
#         in_channels=200,
#         patch_size=11,
#         num_classes=16,
#         embed_dim=128,
#         depth=6,
#         num_heads=8
#     )

#     # Test forward pass
#     print("\nTesting forward pass...")
#     x = torch.randn(8, 200, 11, 11)
#     logits = model(x)

#     print(f"✓ Input shape: {x.shape}")
#     print(f"✓ Output shape: {logits.shape}")
#     print(f"✓ Parameters: {sum(p.numel() for p in model.parameters()):,}")

#     # Display detailed model summary using torchinfo
#     try:
#         from torchinfo import summary
#         print("\n" + "="*100)
#         print("DETAILED MODEL SUMMARY (torchinfo)")
#         print("="*100)

#         summary(
#             model,
#             input_size=(8, 200, 11, 11),
#             col_names=["input_size", "output_size", "num_params", "mult_adds"],
#             row_settings=["var_names"],
#             depth=4,
#             verbose=1
#         )

#     except ImportError:
#         print("\n⚠ torchinfo not installed. Install with: pip install torchinfo")
#         print("\nShowing manual parameter count instead:")

#         total_params = 0
#         print(f"\n{'Layer':<50} {'Parameters':<20}")
#         print("-"*70)
#         for name, param in model.named_parameters():
#             param_count = param.numel()
#             total_params += param_count
#             print(f"{name:<50} {param_count:>15,}")
#         print("-"*70)
#         print(f"{'TOTAL':<50} {total_params:>15,}")
