from models.registry import register_model
"""
GTCFN: A Graph-Based Transformer and Convolution Fusion Network for Hyperspectral Image Classification

Paper: https://ieeexplore.ieee.org/abstract/document/11195861
GitHub: https://github.com/Majunyi310321/GTCFN
Venue: IEEE Transactions on Geoscience and Remote Sensing (TGRS)
Year: 2024
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


class GCNLayer(nn.Module):
    """Graph Convolutional Layer"""
    def __init__(self, input_dim, output_dim):
        super(GCNLayer, self).__init__()
        self.BN = nn.BatchNorm1d(input_dim)
        self.Activation = nn.LeakyReLU()
        self.GCN_liner_theta = nn.Linear(input_dim, 256)
        self.GCN_liner_out = nn.Linear(input_dim, output_dim)

    def forward(self, H):
        H = self.BN(H)
        H_proj = self.GCN_liner_theta(H)
        # Self-attention based adjacency
        A = torch.sigmoid(torch.matmul(H_proj, H_proj.transpose(-2, -1)))
        A = F.softmax(A, dim=-1)
        output = self.Activation(torch.matmul(A, self.GCN_liner_out(H)))
        return output


class MSSSConv(nn.Module):
    """Multi-Scale Spatial-Spectral Convolution"""
    def __init__(self, in_ch, out_ch, ks_list=(5, 7, 9)):
        super().__init__()
        self.bn = nn.BatchNorm2d(in_ch)
        self.point = nn.Conv2d(in_ch, out_ch, 1, bias=False)
        self.branches = nn.ModuleList([
            nn.Conv2d(out_ch, out_ch, kernel_size=k, padding=k // 2, groups=out_ch, bias=False)
            for k in ks_list
        ])
        self.act = nn.LeakyReLU(inplace=True)
        self.fuse = nn.Conv2d(out_ch * len(ks_list), out_ch, 1, bias=False)

    def forward(self, x):
        x = self.bn(x)
        x = self.point(x)
        x = self.act(x)
        feats = [branch(x) for branch in self.branches]
        x = torch.cat(feats, dim=1)
        x = self.fuse(x)
        x = self.act(x)
        return x


class SimpleGraphAttention(nn.Module):
    """Simplified Graph Attention for patch tokens"""
    def __init__(self, in_dim, hidden_dim, out_dim, num_heads=4):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = hidden_dim // num_heads
        
        self.q_proj = nn.Linear(in_dim, hidden_dim)
        self.k_proj = nn.Linear(in_dim, hidden_dim)
        self.v_proj = nn.Linear(in_dim, hidden_dim)
        self.out_proj = nn.Linear(hidden_dim, out_dim)
        self.norm = nn.LayerNorm(in_dim)
        
    def forward(self, x):
        # x: [B, N, D]
        x = self.norm(x)
        B, N, D = x.shape
        
        q = self.q_proj(x).view(B, N, self.num_heads, self.head_dim).transpose(1, 2)
        k = self.k_proj(x).view(B, N, self.num_heads, self.head_dim).transpose(1, 2)
        v = self.v_proj(x).view(B, N, self.num_heads, self.head_dim).transpose(1, 2)
        
        attn = torch.matmul(q, k.transpose(-2, -1)) / (self.head_dim ** 0.5)
        attn = F.softmax(attn, dim=-1)
        
        out = torch.matmul(attn, v)
        out = out.transpose(1, 2).contiguous().view(B, N, -1)
        out = self.out_proj(out)
        
        return out


class GTCFNBase(nn.Module):
    """
    Simplified GTCFN for patch-based classification.
    Uses CNN + Graph Attention instead of superpixel-based GCN.
    """
    def __init__(self, num_features=200, num_classes=16, hidden_dim=128, 
                 denoise_layers=3, cnn_layers=2, graph_layers=2):
        super(GTCFNBase, self).__init__()
        
        self.num_classes = num_classes
        self.hidden_dim = hidden_dim
        
        # Spectral Transformation (denoising)
        self.CNN_denoise = nn.Sequential()
        for i in range(denoise_layers):
            if i == 0:
                self.CNN_denoise.add_module(f'BN_{i}', nn.BatchNorm2d(num_features))
                self.CNN_denoise.add_module(f'Conv_{i}', nn.Conv2d(num_features, hidden_dim, kernel_size=1))
                self.CNN_denoise.add_module(f'Act_{i}', nn.LeakyReLU())
            else:
                self.CNN_denoise.add_module(f'BN_{i}', nn.BatchNorm2d(hidden_dim))
                self.CNN_denoise.add_module(f'Conv_{i}', nn.Conv2d(hidden_dim, hidden_dim, kernel_size=1))
                self.CNN_denoise.add_module(f'Act_{i}', nn.LeakyReLU())
        
        # Pixel-level CNN Branch
        self.CNN_Branch = nn.Sequential(
            MSSSConv(hidden_dim, hidden_dim, ks_list=(3, 5, 7)),
            MSSSConv(hidden_dim, hidden_dim // 2, ks_list=(3, 5, 7))
        )
        
        # Graph Attention Branch (replaces superpixel GCN)
        self.graph_layers = nn.ModuleList()
        for i in range(graph_layers):
            in_d = hidden_dim if i == 0 else hidden_dim // 2
            out_d = hidden_dim // 2
            self.graph_layers.append(SimpleGraphAttention(in_d, hidden_dim, out_d, num_heads=4))
        
        # Fusion and Classification
        self.fusion = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.LeakyReLU(),
            nn.Dropout(0.3),
            nn.Linear(hidden_dim, num_classes)
        )
        
    def forward(self, x):
        B, C, H, W = x.shape
        
        # Spectral denoising
        x = self.CNN_denoise(x)  # [B, hidden_dim, H, W]
        
        # CNN Branch
        cnn_feat = self.CNN_Branch(x)  # [B, hidden_dim//2, H, W]
        
        # Graph Branch - treat spatial positions as nodes
        graph_input = x.flatten(2).transpose(1, 2)  # [B, H*W, hidden_dim]
        graph_feat = graph_input
        for layer in self.graph_layers:
            graph_feat = layer(graph_feat) + graph_feat[:, :, :graph_feat.shape[-1] // 2 * 2][:, :, :layer.out_proj.out_features] if hasattr(layer, 'out_proj') else layer(graph_feat)
        
        # Reshape graph features back
        graph_feat = graph_feat.transpose(1, 2).view(B, -1, H, W)  # [B, hidden_dim//2, H, W]
        
        # Combine features
        combined = torch.cat([cnn_feat, graph_feat], dim=1)  # [B, hidden_dim, H, W]
        
        # Global pooling and classification
        pooled = F.adaptive_avg_pool2d(combined, 1).flatten(1)  # [B, hidden_dim]
        output = self.fusion(pooled)  # [B, num_classes]
        
        return output


class GTCFN(nn.Module):
    """GTCFN wrapper for patch-based classification"""
    def __init__(self, num_features=200, num_classes=16, hidden_dim=128, **kwargs):
        super(GTCFN, self).__init__()
        self.base = GTCFNBase(num_features, num_classes, hidden_dim)
        
    def forward(self, x):
        # Handle 5D input [B, 1, C, H, W] -> [B, C, H, W]
        if x.dim() == 5:
            x = x.squeeze(1)
        return self.base(x)


# Register model


@register_model('GTCFN', hidden_dim=128)
def gtcfn_model(pretrained: bool = False, **kwargs) -> GTCFN:
    """
    Constructs a simplified GTCFN model for patch-based classification.
    
    Note: This is a simplified version that uses graph attention on patch tokens
    instead of the original superpixel-based approach.
    """
    if 'bands' in kwargs:
        kwargs['num_features'] = kwargs.pop('bands')
    kwargs.pop('patch_size', None)
    return GTCFN(**kwargs)
