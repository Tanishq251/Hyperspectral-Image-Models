"""ENL-FCN: Efficient Non-local Features for Hyperspectral Image Classification

Paper: Efficient Deep Learning of Non-local Features for Hyperspectral Image Classification
Year: 2020
Venue: IEEE TGRS

This implementation adapts the ENL-FCN model to the unified API.
The original model uses Criss-Cross Attention for efficient non-local feature extraction.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from models.registry import register_model


class CrissCrossAttention(nn.Module):
    """Criss-Cross Attention Module
    
    Simplified implementation without CUDA extensions.
    Uses standard PyTorch operations for cross-shaped attention.
    """
    def __init__(self, in_dim):
        super(CrissCrossAttention, self).__init__()
        self.chanel_in = in_dim
        self.query_conv = nn.Conv2d(in_channels=in_dim, out_channels=in_dim, kernel_size=1)
        self.key_conv = nn.Conv2d(in_channels=in_dim, out_channels=in_dim, kernel_size=1)
        self.value_conv = nn.Conv2d(in_channels=in_dim, out_channels=in_dim, kernel_size=1)
        self.gamma = 0.6
    
    def forward(self, x):
        """Forward pass with simplified criss-cross attention
        
        Args:
            x: Input tensor of shape (B, C, H, W)
            
        Returns:
            Output tensor of shape (B, C, H, W)
        """
        B, C, H, W = x.size()
        
        proj_query = self.query_conv(x)  # (B, C, H, W)
        proj_key = self.key_conv(x)      # (B, C, H, W)
        proj_value = self.value_conv(x)  # (B, C, H, W)
        
        # Simplified attention: use global average pooling along spatial dimensions
        # to approximate the criss-cross pattern
        
        # Horizontal attention (aggregate along width)
        query_h = proj_query.mean(dim=3, keepdim=True)  # (B, C, H, 1)
        key_h = proj_key.mean(dim=3, keepdim=True)      # (B, C, H, 1)
        value_h = proj_value.mean(dim=3, keepdim=True)  # (B, C, H, 1)
        
        # Vertical attention (aggregate along height)
        query_w = proj_query.mean(dim=2, keepdim=True)  # (B, C, 1, W)
        key_w = proj_key.mean(dim=2, keepdim=True)      # (B, C, 1, W)
        value_w = proj_value.mean(dim=2, keepdim=True)  # (B, C, 1, W)
        
        # Compute attention weights
        energy_h = (query_h * key_h).sum(dim=1, keepdim=True)  # (B, 1, H, 1)
        energy_w = (query_w * key_w).sum(dim=1, keepdim=True)  # (B, 1, 1, W)
        
        # Broadcast and combine
        attention_h = F.softmax(energy_h, dim=2)  # (B, 1, H, 1)
        attention_w = F.softmax(energy_w, dim=3)  # (B, 1, 1, W)
        
        # Apply attention
        out_h = value_h * attention_h  # (B, C, H, 1)
        out_w = value_w * attention_w  # (B, C, 1, W)
        
        # Broadcast and combine
        out = out_h.expand_as(x) + out_w.expand_as(x)
        out = self.gamma * out + x
        
        return out


class ENL_FCN(nn.Module):
    """ENL-FCN: Efficient Non-local Fully Convolutional Network
    
    Architecture:
    - Spectral-spatial feature extraction with depthwise separable convolutions
    - Criss-cross attention for efficient non-local feature aggregation
    - Dense connections for feature reuse
    """
    
    def __init__(self, num_classes, bands, patch_size=11, chanel=150):
        super(ENL_FCN, self).__init__()
        self.bands = bands
        self.chanel = chanel
        kernel = 5
        CCChannel = 25
        
        # First block: spectral feature extraction
        self.b1 = nn.BatchNorm2d(self.bands)
        self.con1 = nn.Conv2d(self.bands, chanel, 1, padding=0, bias=True)
        self.s1 = nn.Sigmoid()
        self.cond1 = nn.Conv2d(chanel, chanel, kernel, padding=2, groups=chanel, bias=True)
        self.sd1 = nn.Sigmoid()
        
        # Second block: spectral-spatial feature extraction
        self.b2 = nn.BatchNorm2d(self.bands + chanel)
        self.con2 = nn.Conv2d(self.bands + chanel, chanel, 1, padding=0, bias=True)
        self.s2 = nn.Sigmoid()
        self.cond2 = nn.Conv2d(chanel, CCChannel, kernel, padding=2, groups=25, bias=True)
        self.sd2 = nn.Sigmoid()
        
        # Non-local attention blocks
        self.b4 = nn.BatchNorm2d(CCChannel)
        self.nlcon2 = CrissCrossAttention(CCChannel)
        self.nlcon3 = CrissCrossAttention(CCChannel)
        
        # Feature fusion after attention
        self.bcat = nn.BatchNorm2d(CCChannel + CCChannel)
        self.con4 = nn.Conv2d(CCChannel + CCChannel, chanel, 1, padding=0, bias=True)
        self.s4 = nn.Sigmoid()
        self.cond4 = nn.Conv2d(chanel, chanel, kernel, padding=2, groups=chanel, bias=True)
        self.sd4 = nn.Sigmoid()
        
        # Fifth block: feature refinement
        self.b5 = nn.BatchNorm2d(CCChannel + chanel)
        self.con5 = nn.Conv2d(CCChannel + chanel, chanel, 1, padding=0, bias=True)
        self.s5 = nn.Sigmoid()
        self.cond5 = nn.Conv2d(chanel, chanel, kernel, padding=2, groups=chanel, bias=True)
        self.sd5 = nn.Sigmoid()
        
        # Classification head
        self.con6 = nn.Conv2d(chanel + CCChannel, num_classes, 1, padding=0, bias=True)
        
        # Global average pooling for final classification
        self.gap = nn.AdaptiveAvgPool2d(1)
    
    def forward(self, x):
        """Forward pass
        
        Args:
            x: Input tensor of shape (B, bands, H, W) or (B, 1, bands, H, W)
            
        Returns:
            Output logits of shape (B, num_classes)
        """
        # Handle 4D input (B, 1, bands, H, W) -> (B, bands, H, W)
        if x.dim() == 5:
            x = x.squeeze(1)
        
        # First block
        out1 = self.b1(x)
        out1 = self.con1(out1)
        out1 = self.s1(out1)
        out1 = self.cond1(out1)
        out1 = self.sd1(out1)
        
        # Second block with dense connection
        out2 = torch.cat((out1, x), 1)
        out2 = self.b2(out2)
        out2 = self.con2(out2)
        out2 = self.s2(out2)
        out2 = self.cond2(out2)
        out2 = self.sd2(out2)
        
        # Non-local attention blocks
        xx = self.b4(out2)
        nl2 = self.nlcon2(xx)
        nl2 = self.nlcon2(nl2)  # Apply twice for recurrent attention
        nl3 = self.nlcon3(xx)
        nl3 = self.nlcon3(nl3)  # Apply twice for recurrent attention
        nl2 = (nl2 + nl3) * 0.7 + xx  # Combine two attention branches
        
        # Fourth block: feature fusion
        out4 = torch.cat((xx, nl2), 1)
        out4 = self.bcat(out4)
        out4 = self.con4(out4)
        out4 = self.s4(out4)
        out4 = self.cond4(out4)
        out4 = self.sd4(out4)
        
        # Fifth block: feature refinement
        out5 = torch.cat((out4, out2), 1)
        out5 = self.b5(out5)
        out5 = self.con5(out5)
        out5 = self.s5(out5)
        out5 = self.cond5(out5)
        out5 = self.sd5(out5)
        
        # Classification
        out6 = torch.cat((out5, out2), 1)
        out6 = self.con6(out6)
        
        # Global average pooling to get (B, num_classes, 1, 1)
        out6 = self.gap(out6)
        
        # Flatten to (B, num_classes)
        out6 = out6.view(out6.size(0), -1)
        
        return out6


@register_model('ENL_FCN', expects_4d=True)
def enl_fcn_model(num_classes, bands, patch_size=11, chanel=150, **kwargs):
    """Factory function for ENL-FCN model
    
    Args:
        num_classes: Number of classification classes
        bands: Number of spectral bands
        patch_size: Spatial size of input patches (not used in architecture)
        chanel: Number of channels in intermediate layers (default: 150)
        
    Returns:
        ENL_FCN model instance
    """
    return ENL_FCN(num_classes=num_classes, bands=bands, patch_size=patch_size, chanel=chanel)
