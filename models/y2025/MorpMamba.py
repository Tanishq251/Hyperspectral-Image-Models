from models.registry import register_model
"""
MorpMamba: Spatial-Spectral Morphological Mamba for Hyperspectral Image Classification

Paper: https://doi.org/10.1016/j.neucom.2025.129990 (Neurocomputing)
GitHub: https://github.com/mahmad000/MorpMamba
Venue: Neurocomputing
Year: 2025

Note: the original repo ships only a Colab notebook
("Spatial-Spectral MorpMamba.ipynb") implemented in TensorFlow/Keras. This
file is a faithful PyTorch port of that architecture: learnable morphological
erosion/dilation "tokenization" (depthwise box-filter convs initialized to an
all-ones structuring element, as in the original `ErosionLayer`/
`DilationLayer`) producing spatial and spectral token maps -> multi-head
cross attention between the two streams -> center-token gated feature
enhancement -> a simple recurrent state-space aggregator -> MLP classifier.
A couple of shape bugs in the original notebook (indexing/broadcasting that
only worked for the specific demo tensor shapes) were fixed so the module
works for arbitrary patch sizes/band counts while preserving the intended
data flow.
"""

import torch
import torch.nn as nn


class MorphLayer(nn.Module):
    """Learnable depthwise structuring-element filter, initialized to an
    all-ones kernel (mirrors the Keras DepthwiseConv2D used for both erosion
    and dilation in the original notebook)."""

    def __init__(self, channels, kernel_size=5, erosion=False):
        super().__init__()
        self.conv = nn.Conv2d(channels, channels, kernel_size,
                               padding=kernel_size // 2, groups=channels, bias=False)
        nn.init.ones_(self.conv.weight)
        self.erosion = erosion

    def forward(self, x):
        out = self.conv(x)
        return -out if self.erosion else out


class SpectralSpatialTokenGeneration(nn.Module):
    """Morphological spatial/spectral token generation (mirrors the Keras
    `SpectralSpatialTokenGeneration` layer built from erosion/dilation)."""

    def __init__(self, in_channels, out_channels, kernel_size=5):
        super().__init__()
        self.erosion_spatial = MorphLayer(in_channels, kernel_size, erosion=True)
        self.dilation_spatial = MorphLayer(in_channels, kernel_size, erosion=False)
        self.erosion_spectral = MorphLayer(in_channels, kernel_size, erosion=True)
        self.dilation_spectral = MorphLayer(in_channels, kernel_size, erosion=False)
        self.conv = nn.Conv2d(2 * in_channels, out_channels, kernel_size=1)

    def forward(self, x):
        # x: (B, C, H, W)
        eroded_spatial = self.erosion_spatial(x)
        dilated_spatial = self.dilation_spatial(x)
        eroded_spectral = self.erosion_spectral(x)
        dilated_spectral = self.dilation_spectral(x)

        combined_spatial = self.conv(torch.cat([eroded_spatial, dilated_spatial], dim=1))
        combined_spectral = self.conv(torch.cat([eroded_spectral, dilated_spectral], dim=1))

        B, D, H, W = combined_spatial.shape
        spatial_tokens = combined_spatial.permute(0, 2, 3, 1).reshape(B, H * W, D)
        spectral_tokens = combined_spectral.permute(0, 2, 3, 1).reshape(B, H * W, D)
        return spatial_tokens, spectral_tokens


class MultiHeadAttention(nn.Module):
    def __init__(self, embed_dim, num_heads, dropout=0.1):
        super().__init__()
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
        spatial_gate = self.spatial_gate(center_tokens).unsqueeze(1)
        spectral_gate = self.spectral_gate(center_tokens).unsqueeze(1)
        spatial_enhanced = spatial_tokens * spatial_gate
        spectral_enhanced = spectral_tokens * spectral_gate
        return spatial_enhanced, spectral_enhanced


class StateSpaceModel(nn.Module):
    def __init__(self, in_dim, state_dim):
        super().__init__()
        self.state_dim = state_dim
        self.state_transition = nn.Linear(state_dim, state_dim)
        self.state_update = nn.Linear(in_dim, state_dim)

    def forward(self, x):
        B, N, _ = x.shape
        state = x.new_zeros(B, self.state_dim)
        for t in range(N):
            state = self.state_transition(state) + self.state_update(x[:, t, :])
        return state


class MorpMamba(nn.Module):
    def __init__(self, in_channels=30, out_channels=64, num_heads=4, state_dim=128,
                 num_classes=16, dropout=0.1, kernel_size=5, patch_size=11, **kwargs):
        super().__init__()
        self.token_generation = SpectralSpatialTokenGeneration(in_channels, out_channels, kernel_size)
        self.multi_head_attention = MultiHeadAttention(out_channels, num_heads, dropout)
        self.feature_enhancement = SpectralSpatialFeatureEnhancement(out_channels, out_channels)
        self.state_space_model = StateSpaceModel(out_channels, state_dim)
        self.dense = nn.Sequential(nn.Linear(state_dim, 128), nn.ReLU())
        self.dropout = nn.Dropout(0.4)
        self.classifier = nn.Linear(128, num_classes)

    def forward(self, x):
        # x: (B, C, H, W)
        spatial_tokens, spectral_tokens = self.token_generation(x)
        center_idx = spatial_tokens.shape[1] // 2
        center_tokens = spatial_tokens[:, center_idx, :]

        spatial_enhanced, spectral_enhanced = self.feature_enhancement(
            spatial_tokens, spectral_tokens, center_tokens
        )
        attention_output = self.multi_head_attention(spatial_enhanced, spectral_enhanced, spectral_enhanced)
        state_output = self.state_space_model(attention_output)
        dense_output = self.dropout(self.dense(state_output))
        logits = self.classifier(dense_output)
        return logits


@register_model('MorpMamba', expects_4d=True, out_channels=64, num_heads=4, state_dim=128, dropout=0.1, kernel_size=5)
def morpmamba(pretrained: bool = False, **kwargs) -> MorpMamba:
    """Constructs a MorpMamba model."""
    if 'bands' in kwargs:
        kwargs['in_channels'] = kwargs.pop('bands')
    kwargs.pop('patch_size', None)
    return MorpMamba(**kwargs)
