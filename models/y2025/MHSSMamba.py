from models.registry import register_model
"""
MHSSMamba: Multi-head Spatial-Spectral Mamba for Hyperspectral Image Classification

Paper: https://doi.org/10.1080/2150704X.2025.2461330
GitHub: https://github.com/mahmad000/MHSSMamba
Venue: Remote Sensing Letters
Year: 2025

Note: the original repo ships only a Colab notebook (MHSSMamba_Git.ipynb) with a
TensorFlow/Keras model (`SSMambaModel`, built from `SpectralSpatialTokenGeneration`,
`MultiHeadAttention`, `SpectralSpatialFeatureEnhancement` and a recurrent
`StateSpaceModel`). This file is a faithful PyTorch port of that architecture:
spectral/spatial linear tokenization -> multi-head cross attention between the
two token streams -> center-token gated feature enhancement -> a simple
recurrent state-space aggregator -> linear classifier. A couple of shape bugs
in the original notebook (e.g. indexing/broadcasting that only worked for the
specific demo tensor shapes) were fixed so the module works for arbitrary
patch sizes/band counts while preserving the intended data flow.
"""

import torch
import torch.nn as nn


class SpectralSpatialTokenGeneration(nn.Module):
    """Projects the flattened patch into a spatial-token stream and a
    spectral-token stream (mirrors the Keras `SpectralSpatialTokenGeneration`
    layer, which used two independent `Dense` projections)."""

    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.spatial_tokens = nn.Linear(in_channels, out_channels)
        self.spectral_tokens = nn.Linear(in_channels, out_channels)

    def forward(self, x):
        # x: (B, H, W, C)
        B, H, W, C = x.shape
        flat = x.reshape(B, H * W, C)
        spatial_tokens = self.spatial_tokens(flat)
        spectral_tokens = self.spectral_tokens(flat)
        return spatial_tokens, spectral_tokens


class MultiHeadAttention(nn.Module):
    def __init__(self, embed_dim, num_heads, dropout=0.1):
        super().__init__()
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads
        self.all_head_size = self.num_heads * self.head_dim
        self.query = nn.Linear(embed_dim, self.all_head_size)
        self.key = nn.Linear(embed_dim, self.all_head_size)
        self.value = nn.Linear(embed_dim, self.all_head_size)
        self.dropout = nn.Dropout(dropout)

    def forward(self, query, key, value):
        B = query.shape[0]

        def split_heads(t):
            t = t.view(B, -1, self.num_heads, self.head_dim)
            return t.permute(0, 2, 1, 3)

        q = split_heads(self.query(query))
        k = split_heads(self.key(key))
        v = split_heads(self.value(value))

        attn_scores = torch.matmul(q, k.transpose(-1, -2)) / (self.head_dim ** 0.5)
        attn_weights = torch.softmax(attn_scores, dim=-1)
        attn_output = torch.matmul(attn_weights, v)
        attn_output = attn_output.permute(0, 2, 1, 3).contiguous().view(B, -1, self.all_head_size)
        return self.dropout(attn_output)


class SpectralSpatialFeatureEnhancement(nn.Module):
    def __init__(self, embed_dim, out_channels):
        super().__init__()
        self.spatial_gate = nn.Sequential(nn.Linear(embed_dim, out_channels), nn.Sigmoid())
        self.spectral_gate = nn.Sequential(nn.Linear(embed_dim, out_channels), nn.Sigmoid())

    def forward(self, spatial_tokens, spectral_tokens, center_tokens):
        spatial_gate = self.spatial_gate(center_tokens).unsqueeze(1)   # (B, 1, out_channels)
        spectral_gate = self.spectral_gate(center_tokens).unsqueeze(1)
        spatial_enhanced = spatial_tokens * spatial_gate
        spectral_enhanced = spectral_tokens * spectral_gate
        return spatial_enhanced, spectral_enhanced


class StateSpaceModel(nn.Module):
    """Simple learned recurrent scan, matching the Keras `StateSpaceModel`
    layer (a minimal linear state-space recurrence, not the selective-scan
    Mamba kernel)."""

    def __init__(self, in_dim, state_dim):
        super().__init__()
        self.state_dim = state_dim
        self.state_transition = nn.Linear(state_dim, state_dim)
        self.state_update = nn.Linear(in_dim, state_dim)

    def forward(self, x):
        # x: (B, N, D)
        B, N, _ = x.shape
        state = x.new_zeros(B, self.state_dim)
        for t in range(N):
            state = self.state_transition(state) + self.state_update(x[:, t, :])
        return state


class MHSSMamba(nn.Module):
    def __init__(self, in_channels=30, out_channels=64, num_heads=4, state_dim=128,
                 num_classes=16, dropout=0.1, patch_size=11, **kwargs):
        super().__init__()
        self.token_generation = SpectralSpatialTokenGeneration(in_channels, out_channels)
        self.multi_head_attention = MultiHeadAttention(out_channels, num_heads, dropout)
        self.feature_enhancement = SpectralSpatialFeatureEnhancement(out_channels, out_channels)
        self.state_space_model = StateSpaceModel(out_channels, state_dim)
        self.classifier = nn.Linear(state_dim, num_classes)

    def forward(self, x):
        # x: (B, C, H, W) -> (B, H, W, C) to match the original NHWC layout
        x = x.permute(0, 2, 3, 1).contiguous()
        B, H, W, C = x.shape

        spatial_tokens, spectral_tokens = self.token_generation(x)
        center_idx = spatial_tokens.shape[1] // 2
        center_tokens = spatial_tokens[:, center_idx, :]

        spatial_enhanced, spectral_enhanced = self.feature_enhancement(
            spatial_tokens, spectral_tokens, center_tokens
        )
        attention_output = self.multi_head_attention(spatial_enhanced, spectral_enhanced, spectral_enhanced)
        state_output = self.state_space_model(attention_output)
        logits = self.classifier(state_output)
        return logits


@register_model('MHSSMamba', expects_4d=True, out_channels=64, num_heads=4, state_dim=128, dropout=0.1)
def mhssmamba(pretrained: bool = False, **kwargs) -> MHSSMamba:
    """Constructs an MHSSMamba model."""
    if 'bands' in kwargs:
        kwargs['in_channels'] = kwargs.pop('bands')
    kwargs.pop('patch_size', None)
    return MHSSMamba(**kwargs)
