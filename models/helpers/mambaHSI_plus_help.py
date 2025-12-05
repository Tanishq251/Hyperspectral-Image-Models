import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from mamba_ssm import Mamba
import math

def padding_feature(self, x):
    """Ensure the input feature map has the required number of channels."""
    B, C, H, W = x.shape
    if C < self.channel_num:
        pad_c = self.channel_num - C
        pad_features = torch.zeros((B, pad_c, H, W), device=x.device)
        return torch.cat([x, pad_features], dim=1)
    return x


class ECALayer(nn.Module):
    """
    ECA: Efficient Channel Attention with Mamba
    """
    def __init__(self, channel):
        super().__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.mamba = Mamba(d_model=channel, d_state=16, d_conv=4, expand=2)  # 用 Mamba 代替 Conv1d

    def forward(self, x):
        # x: [B, C, H, W]
        # x = self.conv(x)
        y = self.avg_pool(x)  # [B, C, 1, 1]
        y = y.squeeze(-1).permute(0, 2, 1)  # [B, C, 1] -> [B, 1, C]

        y = self.mamba(y)  # 经过 Mamba 注意力计算 [B, 1, C]

        y = y.permute(0, 2, 1).unsqueeze(-1)  # [B, 1, C] -> [B, C, 1] -> [B, C, 1, 1]
        return x * torch.sigmoid(y)

class SpaMambaProcessor(nn.Module):
    def __init__(self, channels):
        super(SpaMambaProcessor, self).__init__()
        self.mamba = Mamba(d_model=channels, d_state=16, d_conv=4, expand=2)

    def forward(self, x):
        """
        Applies Mamba processing along one spatial dimension.
        x: B H W C
        """
        B, H, W, C = x.shape
        x_flat = x.view(B, -1, x.shape[-1])  # Flatten to [B, H*W, C]
        x_proc = self.mamba(x_flat)

        x_flipped = torch.flip(x_flat, dims=[1])
        x_proc_flipped = torch.flip(self.mamba(x_flipped), dims=[1])
        return x_proc.view(*x.shape) + x_proc_flipped.view(*x.shape)


class SpeMambaProcessor(nn.Module):
    """A module for processing input features using Mamba in both forward and reversed directions."""
    def __init__(self, group_channel_num):
        super(SpeMambaProcessor, self).__init__()
        self.mamba = Mamba(d_model=group_channel_num, d_state=16, d_conv=4, expand=2)
        self.group_channel_num = group_channel_num

    def forward(self, x):
        B, H, W, C = x.shape
        x_flat = x.view(B * H * W, -1, self.group_channel_num)  # Flatten to [1, B*H*W, C]
        # x_flat = x.view(B * H * W, -1, 128)  # Flatten to [1, B*H*W, C]
        x_flipped = torch.flip(x_flat, dims=[1])
        x_proc = self.mamba(x_flat)
        x_proc_flipped = torch.flip(self.mamba(x_flipped), dims=[1])
        return x_proc.view(*x.shape) + x_proc_flipped.view(*x.shape)

class CustomAttention(nn.Module):
    def __init__(self, embed_dim=32, num_heads=4):
        super().__init__()
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads

        assert self.head_dim * num_heads == embed_dim, "embed_dim must be divisible by num_heads"

        # 多头线性投影层
        self.qkv_proj = nn.Linear(embed_dim, 3 * embed_dim)
        self.out_proj = nn.Linear(embed_dim, embed_dim)

        # 层归一化和Dropout（可选）
        self.layer_norm = nn.LayerNorm(embed_dim)
        self.dropout = nn.Dropout(0.1)

    def forward(self, x):
        """
        x: 输入张量，形状为 (B, L, D)
        返回: 输出张量，形状为 (B, L, D)
        """
        B, L, D = x.shape
        residual = x  # 残差连接

        # 1. 层归一化
        x = self.layer_norm(x)

        # 2. 生成Q、K、V（多头版本）
        qkv = self.qkv_proj(x).chunk(3, dim=-1)  # 拆分为Q、K、V
        q, k, v = [t.view(B, L, self.num_heads, self.head_dim).transpose(1, 2) 
                   for t in qkv]  # (B, H, L, C/H)

        # 3. 计算 Q * softmax(K^T V)
        k_t = k.transpose(-2, -1)  # (B, H, C/H, L)
        kt_v = torch.matmul(k_t, v)  # (B, H, C/H, C/H)
        softmax_kt_v = F.softmax(kt_v, dim=-1)  # 对最后一个维度做softmax
        output = torch.matmul(q, softmax_kt_v)  # (B, H, L, C/H)

        # 4. 合并多头输出
        output = output.transpose(1, 2).contiguous().view(B, L, D)  # (B, L, D)
        output = self.out_proj(output)

        # 5. 残差连接 + Dropout
        output = self.dropout(output)
        return output + residual