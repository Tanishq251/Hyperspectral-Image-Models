from models.registry import register_model
"""
FETNet - Fuzzy Enhanced Transformer Network for Hyperspectral Image Classification

Original architecture with registry integration and single-input support.
Internally generates multi-scale dual-source representations via interpolation.

Usage:
    model = proposed(in_channels=200, patch_size=11, num_classes=16)
    x = torch.randn(8, 200, 11, 11)
    logits = model(x)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange, repeat
from typing import Tuple, Optional





def img2seq(x):
    """[B, C, H, W] -> [B, C, H*W]"""
    b, c, h, w = x.shape
    return x.reshape(b, c, h * w)


def seq2img(x):
    """[B, C, D] -> [B, C, H, W] where H=W=sqrt(D)"""
    b, c, d = x.shape
    p = int(d ** 0.5)
    return x.reshape(b, c, p, p)


class FuzzyLearn(nn.Module):
    """Learnable fuzzy membership functions."""
    def __init__(self, fuzzynum: int, channel: int):
        super().__init__()
        self.n = fuzzynum
        self.channel = channel
        self.mu = nn.Parameter(torch.randn(channel, fuzzynum))
        self.sigma = nn.Parameter(torch.ones(channel, fuzzynum))
        self.bn = nn.BatchNorm1d(channel, affine=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x_expanded = x.unsqueeze(-1)  # [B, C, N, 1]
        mu_expanded = self.mu.view(1, self.channel, 1, self.n)
        sigma_expanded = self.sigma.view(1, self.channel, 1, self.n)
        
        tmp = -((x_expanded - mu_expanded) / (sigma_expanded.abs() + 1e-6)) ** 2
        tmp = torch.logsumexp(tmp, dim=-1)
        
        return self.bn(torch.exp(tmp))


class FuzzyAttention(nn.Module):
    """Fuzzy attention module with Gaussian membership."""
    def __init__(self, in_dim: int = 1, seq_l: int = 64, dropout: float = 0.1):
        super().__init__()
        self.dropout = nn.Dropout(dropout)
        self.seq_l = seq_l

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        original_shape = x.shape
        x_flat = x.reshape(-1, 1, self.seq_l)
        fuzzy = self._fuzzy_transform(x_flat)
        out = self.dropout(x_flat + fuzzy)
        
        # Reshape back
        p = int(self.seq_l ** 0.5)
        return out.reshape(-1, 1, p, p)

    def _fuzzy_transform(self, x: torch.Tensor) -> torch.Tensor:
        mu = torch.mean(x, dim=(0, 1), keepdim=True)
        sigma = torch.std(x, dim=(0, 1), keepdim=True) + 1e-6
        
        const_part = 1 / (sigma * (2 * torch.pi) ** 0.5)
        exp_part = torch.exp(-((x - mu) ** 2) / (2 * sigma ** 2))
        
        return const_part * exp_part + x


class FSAM(nn.Module):
    """Fuzzy Spatial Attention Module for multi-scale fusion."""
    def __init__(self, kernel_size: int = 7, spatial_size: int = 8):
        super().__init__()
        assert kernel_size in (3, 7)
        padding = 3 if kernel_size == 7 else 1
        
        self.spatial_size = spatial_size
        self.conv = nn.Conv2d(2, 1, kernel_size, padding=padding, bias=False)
        self.sigmoid = nn.Sigmoid()
        self.fuzzy_attn = FuzzyAttention(seq_l=spatial_size * spatial_size)

    def forward(self, x1: torch.Tensor, x2: torch.Tensor, x3: torch.Tensor, dim: int) -> torch.Tensor:
        """
        Args:
            x1, x2, x3: Multi-scale features [B, C, N]
            dim: Output dimension
        Returns:
            Fused features [B, dim, N]
        """
        device = x1.device
        p = int(x1.shape[2] ** 0.5)
        
        # Reshape to spatial
        x1 = x1.reshape(x1.shape[0], x1.shape[1], p, p)
        x2 = x2.reshape(x2.shape[0], x2.shape[1], p, p)
        x3 = x3.reshape(x3.shape[0], x3.shape[1], p, p)
        
        num1 = x1.shape[1] // dim
        num2 = x2.shape[1] // dim
        num3 = x3.shape[1] // dim
        
        x_out = torch.empty(x1.shape[0], dim, p, p, device=device)
        
        for i in range(dim):
            # Extract channel groups
            x11_tmp = x1[:, i * num1:(i + 1) * num1, :, :]
            x22_tmp = x2[:, i * num2:(i + 1) * num2, :, :]
            x33_tmp = x3[:, i * num3:(i + 1) * num3, :, :]
            
            # Apply fuzzy attention
            x1_tmp = self.fuzzy_attn(x11_tmp) + x11_tmp
            x2_tmp = self.fuzzy_attn(x22_tmp) + x22_tmp
            x3_tmp = self.fuzzy_attn(x33_tmp) + x33_tmp
            
            # Concatenate and compute spatial attention
            x_concat = torch.cat([x1_tmp, x2_tmp, x3_tmp], dim=1)
            avgout = torch.mean(x_concat, dim=1, keepdim=True)
            maxout, _ = torch.max(x_concat, dim=1, keepdim=True)
            
            attn_input = torch.cat([avgout, maxout], dim=1)
            attn_map = self.sigmoid(self.conv(attn_input))
            
            x_out[:, i:i + 1, :, :] = attn_map
        
        return x_out.reshape(x_out.shape[0], dim, p * p)


class CNN_Encoder(nn.Module):
    """Multi-scale CNN encoder for dual-source fusion."""
    def __init__(self, l1: int, l2: int):
        super().__init__()
        
        # Initial convolutions
        self.conv1 = nn.Sequential(
            nn.Conv2d(l1, 32, 3, 1, 1),
            nn.BatchNorm2d(32),
            nn.ReLU()
        )
        self.conv2 = nn.Sequential(
            nn.Conv2d(l2, 32, 3, 1, 1),
            nn.BatchNorm2d(32),
            nn.ReLU()
        )
        
        # Scale-specific encoders with pooling
        self.conv1_1 = self._make_scale_block(32, 64)
        self.conv2_1 = self._make_scale_block(32, 64)
        self.conv1_2 = self._make_scale_block(32, 64)
        self.conv2_2 = self._make_scale_block(32, 64)
        self.conv1_3 = self._make_scale_block(32, 64)
        self.conv2_3 = self._make_scale_block(32, 64)
        
        # Learnable fusion weights
        self.xishu1 = nn.Parameter(torch.tensor(0.5))
        self.xishu2 = nn.Parameter(torch.tensor(0.5))
    
    def _make_scale_block(self, in_ch: int, out_ch: int) -> nn.Sequential:
        return nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 3, 1, 1),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(),
            nn.MaxPool2d(2)
        )

    def forward(self, x11, x21, x12, x22, x13, x23):
        # Initial encoding
        x11 = self.conv1(x11)
        x21 = self.conv2(x21)
        x12 = self.conv1(x12)
        x22 = self.conv2(x22)
        x13 = self.conv1(x13)
        x23 = self.conv2(x23)
        
        # Scale 1
        x1_1 = self.conv1_1(x11)
        x2_1 = self.conv2_1(x21)
        x_add1 = x1_1 * self.xishu1 + x2_1 * self.xishu2
        
        # Scale 2
        x1_2 = self.conv1_2(x12)
        x2_2 = self.conv2_2(x22)
        x_add2 = x1_2 * self.xishu1 + x2_2 * self.xishu2
        
        # Scale 3
        x1_3 = self.conv1_3(x13)
        x2_3 = self.conv2_3(x23)
        x_add3 = x1_3 * self.xishu1 + x2_3 * self.xishu2
        
        return x_add1, x_add2, x_add3


class CNN_Decoder(nn.Module):
    """Multi-scale CNN decoder for reconstruction."""
    def __init__(self, l1: int, l2: int):
        super().__init__()
        
        self.dconv1 = nn.Sequential(nn.Conv2d(64, l1, 3, 1, 1), nn.Sigmoid())
        self.dconv2 = nn.Sequential(nn.Conv2d(64, l2, 3, 1, 1), nn.Sigmoid())
        self.dconv3 = nn.Sequential(nn.Upsample(scale_factor=2), nn.Conv2d(64, l1, 3, 1, 1), nn.Sigmoid())
        self.dconv4 = nn.Sequential(nn.Upsample(scale_factor=2), nn.Conv2d(64, l2, 3, 1, 1), nn.Sigmoid())
        self.dconv5 = nn.Sequential(nn.Upsample(scale_factor=3), nn.Conv2d(64, l1, 3, 1, 1), nn.Sigmoid())
        self.dconv6 = nn.Sequential(nn.Upsample(scale_factor=3), nn.Conv2d(64, l2, 3, 1, 1), nn.Sigmoid())

    def forward(self, x):
        return self.dconv1(x), self.dconv2(x), self.dconv3(x), self.dconv4(x), self.dconv5(x), self.dconv6(x)


class CNN_Classifier(nn.Module):
    """CNN classification head."""
    def __init__(self, num_classes: int):
        super().__init__()
        self.conv1 = nn.Sequential(
            nn.Conv2d(64, 32, 1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d(1)
        )
        self.conv2 = nn.Conv2d(32, num_classes, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.conv1(x)
        x = self.conv2(x)
        x = x.view(x.size(0), -1)
        return F.softmax(x, dim=1)


class Residual(nn.Module):
    def __init__(self, fn):
        super().__init__()
        self.fn = fn

    def forward(self, x, **kwargs):
        return self.fn(x, **kwargs) + x


class PreNorm(nn.Module):
    def __init__(self, dim, fn):
        super().__init__()
        self.norm = nn.LayerNorm(dim)
        self.fn = fn

    def forward(self, x, **kwargs):
        return self.fn(self.norm(x), **kwargs)


class FeedForward(nn.Module):
    def __init__(self, dim: int, hidden_dim: int, dropout: float = 0.0):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, dim),
            nn.Dropout(dropout)
        )

    def forward(self, x):
        return self.net(x)


class Attention(nn.Module):
    def __init__(self, dim: int, heads: int, dim_head: int, dropout: float):
        super().__init__()
        inner_dim = dim_head * heads
        self.heads = heads
        self.scale = dim_head ** -0.5
        
        self.to_qkv = nn.Linear(dim, inner_dim * 3, bias=False)
        self.to_out = nn.Sequential(
            nn.Linear(inner_dim, dim),
            nn.Dropout(dropout)
        )

    def forward(self, x, mask=None):
        b, n, _, h = *x.shape, self.heads
        
        qkv = self.to_qkv(x).chunk(3, dim=-1)
        q, k, v = map(lambda t: rearrange(t, 'b n (h d) -> b h n d', h=h), qkv)
        
        dots = torch.einsum('bhid,bhjd->bhij', q, k) * self.scale
        
        if mask is not None:
            mask = F.pad(mask.flatten(1), (1, 0), value=True)
            assert mask.shape[-1] == dots.shape[-1]
            mask = mask[:, None, :] * mask[:, :, None]
            dots.masked_fill_(~mask, -torch.finfo(dots.dtype).max)
        
        attn = dots.softmax(dim=-1)
        out = torch.einsum('bhij,bhjd->bhid', attn, v)
        out = rearrange(out, 'b h n d -> b n (h d)')
        return self.to_out(out)


class Transformer(nn.Module):
    def __init__(self, dim: int, depth: int, heads: int, dim_head: int, mlp_dim: int, 
                 dropout: float, num_channel: int):
        super().__init__()
        
        self.layers = nn.ModuleList([])
        for _ in range(depth):
            self.layers.append(nn.ModuleList([
                Residual(PreNorm(dim, Attention(dim, heads, dim_head, dropout))),
                Residual(PreNorm(dim, FeedForward(dim, mlp_dim, dropout)))
            ]))
        
        self.fuzzy_layer = FuzzyLearn(fuzzynum=9, channel=num_channel + 1)

    def forward(self, x, mask=None):
        x = self.fuzzy_layer(x)
        for attn, ff in self.layers:
            x = attn(x, mask=mask)
            x = ff(x)
        return x


class FETNet(nn.Module):
    """
    Fuzzy Enhanced Transformer Network for HSI Classification.
    
    Accepts single input [B, C, H, W] and internally generates multi-scale representations.
    """
    def __init__(self,
                 in_channels: int = 200,
                 patch_size: int = 11,
                 num_classes: int = 16,
                 encoder_embed_dim: int = 64,
                 decoder_embed_dim: int = 64,
                 en_depth: int = 4,
                 de_depth: int = 2,
                 en_heads: int = 4,
                 de_heads: int = 4,
                 mlp_dim: int = 256,
                 dim_head: int = 16,
                 dropout: float = 0.1,
                 emb_dropout: float = 0.1):
        super().__init__()
        
        self.in_channels = in_channels
        self.patch_size = patch_size
        self.encoder_embed_dim = encoder_embed_dim
        
        # Calculate split for dual-source simulation
        self.l1 = in_channels * 6 // 10  # 60%
        self.l2 = in_channels - self.l1   # 40%
        
        # Internal patch size for transformer (after pooling)
        self.internal_patch = max(4, patch_size // 2)
        num_patches = self.internal_patch ** 2
        
        # CNN encoder/decoder
        self.cnn_encoder = CNN_Encoder(self.l1, self.l2)
        self.cnn_decoder = CNN_Decoder(self.l1, self.l2)
        self.cnn_classifier = CNN_Classifier(num_classes)
        
        # Classification fusion weights
        self.coefficient1 = nn.Parameter(torch.tensor(0.5))
        self.coefficient2 = nn.Parameter(torch.tensor(0.5))
        
        # FSAM module
        self.fsam = FSAM(spatial_size=self.internal_patch)
        
        # Reconstruction loss
        self.loss_fn = nn.MSELoss()
        
        # Embeddings
        self.encoder_pos_embed = nn.Parameter(torch.randn(1, self.internal_patch ** 2 + 1, encoder_embed_dim))
        self.decoder_pos_embed = nn.Parameter(torch.randn(1, self.internal_patch ** 2 + 1, decoder_embed_dim))
        
        # Scale projections (dynamic based on pooled sizes)
        # After MaxPool2d(2), sizes become: scale1=2x2=4, scale2=4x4=16, scale3=6x6=36
        self.encoder_embedding1 = nn.Linear(4, self.internal_patch ** 2)
        self.encoder_embedding2 = nn.Linear(16, self.internal_patch ** 2)
        self.encoder_embedding3 = nn.Linear(36, self.internal_patch ** 2)
        
        self.decoder_embedding = nn.Linear(encoder_embed_dim, decoder_embed_dim, bias=True)
        
        # CLS token
        self.cls_token = nn.Parameter(torch.randn(1, 1, encoder_embed_dim))
        self.dropout = nn.Dropout(emb_dropout)
        
        # Transformers
        self.en_transformer = Transformer(encoder_embed_dim, en_depth, en_heads, dim_head, mlp_dim, dropout, num_patches)
        self.de_transformer = Transformer(decoder_embed_dim, de_depth, de_heads, dim_head, mlp_dim, dropout, num_patches)
        
        # Decoder prediction
        self.decoder_pred = nn.Linear(decoder_embed_dim, 64, bias=True)
        
        # Classification head
        self.to_latent = nn.Identity()
        self.mlp_head = nn.Sequential(
            nn.LayerNorm(encoder_embed_dim),
            nn.Linear(encoder_embed_dim, num_classes)
        )
        
        # Fuzzy layer for CNN path - must match encoder_embed_dim
        self.fuzzy_layer = FuzzyLearn(fuzzynum=9, channel=encoder_embed_dim)
        
        self._encoder_adjusted = False

    def _prepare_multiscale_inputs(self, x: torch.Tensor):
        """
        Generate multi-scale dual-source inputs from single input.
        
        Args:
            x: [B, C, H, W] single input
        Returns:
            6 tensors for dual-source multi-scale processing
        """
        B, C, H, W = x.shape
        
        # Split channels into two sources
        source1 = x[:, :self.l1, :, :]
        source2 = x[:, self.l1:self.l1 + self.l2, :, :]
        
        # Handle case where input channels don't match expected
        if source1.shape[1] != self.l1 or source2.shape[1] != self.l2:
            actual_l1 = C * 6 // 10
            actual_l2 = C - actual_l1
            source1 = x[:, :actual_l1, :, :]
            source2 = x[:, actual_l1:, :, :]
            
            # Adjust encoder if needed
            if not self._encoder_adjusted:
                device = x.device
                self.cnn_encoder = CNN_Encoder(actual_l1, actual_l2).to(device)
                self.cnn_decoder = CNN_Decoder(actual_l1, actual_l2).to(device)
                self.l1 = actual_l1
                self.l2 = actual_l2
                self._encoder_adjusted = True
        
        # Generate multi-scale (sizes: 4x4, 8x8, 12x12 before pooling)
        # Scale 1: smallest
        s1_h, s1_w = max(4, H // 2), max(4, W // 2)
        x11 = F.interpolate(source1, size=(s1_h, s1_w), mode='bilinear', align_corners=False)
        x21 = F.interpolate(source2, size=(s1_h, s1_w), mode='bilinear', align_corners=False)
        
        # Scale 2: medium
        s2_h, s2_w = max(8, H), max(8, W)
        x12 = F.interpolate(source1, size=(s2_h, s2_w), mode='bilinear', align_corners=False)
        x22 = F.interpolate(source2, size=(s2_h, s2_w), mode='bilinear', align_corners=False)
        
        # Scale 3: largest
        s3_h, s3_w = max(12, int(H * 1.5)), max(12, int(W * 1.5))
        x13 = F.interpolate(source1, size=(s3_h, s3_w), mode='bilinear', align_corners=False)
        x23 = F.interpolate(source2, size=(s3_h, s3_w), mode='bilinear', align_corners=False)
        
        return x11, x21, x12, x22, x13, x23

    def _encoder(self, x11, x21, x12, x22, x13, x23):
        """Encoder forward pass."""
        # CNN encoding
        x_fuse1, x_fuse2, x_fuse3 = self.cnn_encoder(x11, x21, x12, x22, x13, x23)
        
        # Flatten and project
        x_flat1 = x_fuse1.flatten(2)
        x_flat2 = x_fuse2.flatten(2)
        x_flat3 = x_fuse3.flatten(2)
        
        # Adjust projection layers if sizes don't match
        if x_flat1.shape[2] != 4:
            device = x_flat1.device
            self.encoder_embedding1 = nn.Linear(x_flat1.shape[2], self.internal_patch ** 2).to(device)
        if x_flat2.shape[2] != 16:
            device = x_flat2.device
            self.encoder_embedding2 = nn.Linear(x_flat2.shape[2], self.internal_patch ** 2).to(device)
        if x_flat3.shape[2] != 36:
            device = x_flat3.device
            self.encoder_embedding3 = nn.Linear(x_flat3.shape[2], self.internal_patch ** 2).to(device)
        
        x_1 = self.encoder_embedding1(x_flat1)
        x_2 = self.encoder_embedding2(x_flat2)
        x_3 = self.encoder_embedding3(x_flat3)
        
        # FSAM fusion
        x_cnn = self.fsam(x_1, x_2, x_3, self.encoder_embed_dim)
        x_cnn = torch.einsum('nld->ndl', x_cnn)
        
        b, n, _ = x_cnn.shape
        
        # Handle variable sequence length
        if n != self.encoder_pos_embed.shape[1] - 1:
            pos_embed = F.interpolate(
                self.encoder_pos_embed[:, 1:, :].transpose(1, 2),
                size=n,
                mode='linear',
                align_corners=False
            ).transpose(1, 2)
            cls_pos = self.encoder_pos_embed[:, :1, :]
        else:
            pos_embed = self.encoder_pos_embed[:, 1:, :]
            cls_pos = self.encoder_pos_embed[:, :1, :]
        
        # Add position embedding
        x = x_cnn + pos_embed
        
        # Append CLS token
        cls_tokens = repeat(self.cls_token, '() n d -> b n d', b=b)
        x = torch.cat([cls_tokens, x], dim=1)
        x = x + torch.cat([cls_pos, torch.zeros_like(pos_embed)], dim=1)
        x = self.dropout(x)
        
        # Transformer encoding
        x = self.en_transformer(x, mask=None)
        
        return x, x_cnn

    def _classifier(self, x, x_cnn):
        """Classification using dual path."""
        # Transformer path - x[:, 0] is CLS token
        x_cls1 = self.mlp_head(self.to_latent(x[:, 0]))
        
        # CNN path
        # x_cnn is [B, N, D] where N=spatial tokens, D=encoder_embed_dim
        # FuzzyLearn expects [B, C, N] so we need to transpose
        x_cnn_transposed = x_cnn.transpose(1, 2)  # [B, D, N] = [B, encoder_embed_dim, N]
        x_cnn_fuzzy = self.fuzzy_layer(x_cnn_transposed)  # [B, D, N]
        
        # seq2img expects [B, C, N] and outputs [B, C, H, W]
        # x_cnn_fuzzy is already [B, D, N] which is correct for seq2img
        x_cnn_spatial = seq2img(x_cnn_fuzzy)  # [B, D, H, W]
        x_cls2 = self.cnn_classifier(x_cnn_spatial)
        
        # Fused classification
        return x_cls1 * self.coefficient1 + x_cls2 * self.coefficient2

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.
        
        Args:
            x: [B, C, H, W] input HSI patch
        Returns:
            logits: [B, num_classes]
        """
        # Handle 5D input
        if x.dim() == 5 and x.size(1) == 1:
            x = x.squeeze(1)
        
        # Generate multi-scale inputs
        x11, x21, x12, x22, x13, x23 = self._prepare_multiscale_inputs(x)
        
        # Encode
        x_vit, x_cnn = self._encoder(x11, x21, x12, x22, x13, x23)
        
        # Classify
        logits = self._classifier(x_vit, x_cnn)
        
        return logits

    def forward_with_reconstruction(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass with reconstruction loss.
        
        Args:
            x: [B, C, H, W] input HSI patch
        Returns:
            logits: [B, num_classes]
            recon_loss: scalar reconstruction loss
        """
        if x.dim() == 5 and x.size(1) == 1:
            x = x.squeeze(1)
        
        x11, x21, x12, x22, x13, x23 = self._prepare_multiscale_inputs(x)
        x_vit, x_cnn = self._encoder(x11, x21, x12, x22, x13, x23)
        logits = self._classifier(x_vit, x_cnn)
        
        # Decoder for reconstruction
        x_dec = self.decoder_embedding(x_vit)
        
        # Handle variable sequence length for decoder
        if x_dec.shape[1] != self.decoder_pos_embed.shape[1]:
            dec_pos = F.interpolate(
                self.decoder_pos_embed.transpose(1, 2),
                size=x_dec.shape[1],
                mode='linear',
                align_corners=False
            ).transpose(1, 2)
        else:
            dec_pos = self.decoder_pos_embed
        
        x_dec = x_dec + dec_pos
        x_dec = self.de_transformer(x_dec, mask=None)
        
        x_pred = self.decoder_pred(x_dec)
        x_con = x_pred[:, 1:, :]  # Remove CLS
        x_con = torch.einsum('nld->ndl', x_con)
        
        # Reshape for CNN decoder
        p = int(x_con.shape[2] ** 0.5)
        x_con = x_con.reshape(x_con.shape[0], x_con.shape[1], p, p)
        
        # Decode
        r1, r2, r3, r4, r5, r6 = self.cnn_decoder(x_con)
        
        # Compute reconstruction loss
        loss1 = 0.5 * self.loss_fn(r1, F.interpolate(x11, size=r1.shape[2:])) + \
                0.5 * self.loss_fn(r2, F.interpolate(x21, size=r2.shape[2:]))
        loss2 = 0.5 * self.loss_fn(r3, F.interpolate(x12, size=r3.shape[2:])) + \
                0.5 * self.loss_fn(r4, F.interpolate(x22, size=r4.shape[2:]))
        loss3 = 0.5 * self.loss_fn(r5, F.interpolate(x13, size=r5.shape[2:])) + \
                0.5 * self.loss_fn(r6, F.interpolate(x23, size=r6.shape[2:]))
        
        recon_loss = (loss1 + loss2 + loss3) / 3.0
        
        return logits, recon_loss


@register_model('FETNet', expects_4d=True, feature_dim=64, encoder_depth=4)
def proposed(pretrained: bool = False, **kwargs) -> FETNet:
    """Constructs a FETNet model for hyperspectral classification."""
    
    # Remove registry metadata
    for key in ['expects_4d', 'dual_input', 'feature_dim', 'encoder_depth']:
        kwargs.pop(key, None)
    
    # Map standardized parameter names
    if 'bands' in kwargs:
        kwargs['in_channels'] = kwargs.pop('bands')
    if 'patch_size' in kwargs:
        pass  # Keep as is
    
    defaults = dict(
        in_channels=200,
        patch_size=11,
        num_classes=16,
        encoder_embed_dim=64,
        decoder_embed_dim=64,
        en_depth=4,
        de_depth=2,
        en_heads=4,
        de_heads=4,
        mlp_dim=256,
        dim_head=16,
        dropout=0.1,
        emb_dropout=0.1
    )
    
    config = {**defaults, **kwargs}
    return FETNet(**config)


# if __name__ == "__main__":
#     device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
#     print("=" * 60)
#     print("Testing FETNet")
#     print("=" * 60)
    
#     test_configs = [
#         (8, 200, 11, 16),
#         (4, 48, 9, 20),
#         (2, 145, 11, 14),
#         (4, 30, 7, 10),
#     ]
    
#     for batch, channels, spatial, classes in test_configs:
#         print(f"\nTest: B={batch}, C={channels}, S={spatial}, Classes={classes}")
        
#         try:
#             model = proposed(
#                 in_channels=channels,
#                 patch_size=spatial,
#                 num_classes=classes
#             ).to(device)
            
#             x = torch.randn(batch, channels, spatial, spatial).to(device)
            
#             with torch.no_grad():
#                 logits = model(x)
#                 logits_r, recon_loss = model.forward_with_reconstruction(x)
            
#             print(f"  Input: {x.shape}")
#             print(f"  Output: {logits.shape}")
#             print(f"  Recon Loss: {recon_loss.item():.4f}")
#             print(f"  Params: {sum(p.numel() for p in model.parameters()):,}")
            
#         except Exception as e:
#             print(f"  Failed: {e}")
#             import traceback
#             traceback.print_exc()
    
#     print("\n" + "=" * 60)
#     print("All tests completed!")
#     print("=" * 60)
