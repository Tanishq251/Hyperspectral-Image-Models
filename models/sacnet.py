"""
SACNet: Self-Attention Context Network for Hyperspectral Image Classification

Paper: https://ieeexplore.ieee.org/document/9557629
GitHub: https://github.com/YonghaoXu/SACNet
Venue: IEEE Transactions on Image Processing (TIP)
Year: 2021
"""

import torch
from torch import nn
from torch.nn import functional as F
import numpy as np


def scaled_l2(X, C, S):
    """scaled_l2 distance"""
    assert X.shape[-1] == C.shape[-1], "input, codeword feature dim mismatch"
    assert S.numel() == C.shape[0], "scale, codeword num mismatch"

    b, n, d = X.shape
    X = X.reshape(-1, d)  # Use reshape for non-contiguous tensors
    Ct = C.t()
    X2 = X.pow(2.0).sum(-1, keepdim=True)
    C2 = Ct.pow(2.0).sum(0, keepdim=True)
    norm = X2 + C2 - 2.0 * X.mm(Ct)
    scaled_norm = S * norm
    D = scaled_norm.reshape(b, n, -1)  # Use reshape
    return D


def aggregate(A, X, C):
    """aggregate residuals from N samples"""
    assert X.shape[-1] == C.shape[-1], "input, codeword feature dim mismatch"
    assert A.shape[:2] == X.shape[:2], "weight, input dim mismatch"
    X = X.unsqueeze(2)
    C = C[None, None, ...]
    A = A.unsqueeze(-1)
    R = (X - C) * A
    E = R.sum(dim=1)
    return E


class SACNetBase(nn.Module):
    """Base SACNet - outputs spatial segmentation map"""
    def __init__(self, num_features=103, num_classes=9, conv_features=64, trans_features=32, K=48, D=32):
        super(SACNetBase, self).__init__()

        self.conv0 = nn.Conv2d(num_features, conv_features, kernel_size=3, stride=1, padding=1, dilation=1, bias=True)
        self.conv1 = nn.Conv2d(conv_features, conv_features, kernel_size=3, stride=1, padding=2, dilation=2, bias=True)
        self.conv2 = nn.Conv2d(conv_features, conv_features, kernel_size=3, stride=1, padding=3, dilation=3, bias=True)

        self.alpha3 = nn.Conv2d(conv_features, trans_features, kernel_size=1, stride=1, padding=0, bias=False)
        self.beta3 = nn.Conv2d(conv_features, trans_features, kernel_size=1, stride=1, padding=0, bias=False)
        self.gamma3 = nn.Conv2d(conv_features, trans_features, kernel_size=1, stride=1, padding=0, bias=False)
        self.deta3 = nn.Conv2d(trans_features, conv_features, kernel_size=1, stride=1, padding=0, bias=False)

        self.encoding = nn.Conv2d(conv_features, D, kernel_size=1, stride=1, padding=0, bias=False)

        self.codewords = nn.Parameter(torch.Tensor(K, D), requires_grad=True)
        self.scale = nn.Parameter(torch.Tensor(K), requires_grad=True)
        self.attention = nn.Linear(D, conv_features)

        self.relu = nn.ReLU(inplace=True)
        self.sigmoid = nn.Sigmoid()
        self.avgpool = nn.AvgPool2d(kernel_size=2, stride=2, padding=0)

        self.conv_cls = nn.Conv2d(conv_features * 3, num_classes, kernel_size=1, stride=1, padding=0, bias=True)

        self.conv_features = conv_features
        self.trans_features = trans_features
        self.K = K
        self.D = D

        std1 = 1. / ((self.K * self.D) ** (1 / 2))
        self.codewords.data.uniform_(-std1, std1)
        self.scale.data.uniform_(-1, 0)
        self.BN = nn.BatchNorm1d(K)

    def forward(self, x):
        interpolation = nn.UpsamplingBilinear2d(size=x.shape[2:4])
        x = self.relu(self.conv0(x))
        conv1 = x

        x = self.relu(self.conv1(x))
        conv2 = x
        x = self.avgpool(x)

        x = self.relu(self.conv2(x))
        n, c, h, w = x.size()
        interpolation_context3 = nn.UpsamplingBilinear2d(size=x.shape[2:4])

        x_half = self.avgpool(x)
        n, c, h, w = x_half.size()
        alpha_x = self.alpha3(x_half)  # [n, trans_features, h, w]
        beta_x = self.beta3(x_half)
        gamma_x = self.relu(self.gamma3(x_half))

        # Reshape for batch matrix multiplication
        # alpha_x: [n, trans_features, h, w] -> [n, h*w, trans_features]
        alpha_x = alpha_x.view(n, self.trans_features, -1).permute(0, 2, 1)
        # beta_x: [n, trans_features, h, w] -> [n, trans_features, h*w]
        beta_x = beta_x.view(n, self.trans_features, -1)
        # gamma_x: [n, trans_features, h, w] -> [n, trans_features, h*w]
        gamma_x = gamma_x.view(n, self.trans_features, -1)

        # Batch matrix multiplication
        context_x = torch.bmm(alpha_x, beta_x)  # [n, h*w, h*w]
        context_x = F.softmax(context_x, dim=-1)
        context_x = torch.bmm(gamma_x, context_x)  # [n, trans_features, h*w]
        context_x = context_x.view(n, self.trans_features, h, w)
        context_x = interpolation_context3(context_x)

        deta_x = self.relu(self.deta3(context_x))
        x = deta_x + x

        n, c, h, w = x.size()
        Z = self.relu(self.encoding(x)).view(n, self.D, -1).permute(0, 2, 1)

        A = F.softmax(scaled_l2(Z, self.codewords, self.scale), dim=2)
        E = aggregate(A, Z, self.codewords)
        E_sum = torch.sum(self.relu(self.BN(E)), 1)
        gamma = self.sigmoid(self.attention(E_sum))
        gamma = gamma.view(-1, self.conv_features, 1, 1)
        x = x + x * gamma
        context3 = interpolation(x)
        conv2 = interpolation(conv2)
        conv1 = interpolation(conv1)

        x = torch.cat((conv1, conv2, context3), 1)
        x = self.conv_cls(x)

        return x


class SACNet(nn.Module):
    """SACNet wrapper for patch-based classification - extracts center pixel prediction"""
    def __init__(self, num_features=103, num_classes=9, conv_features=64, trans_features=32, K=48, D=32, **kwargs):
        super(SACNet, self).__init__()
        self.base = SACNetBase(num_features, num_classes, conv_features, trans_features, K, D)
        self.num_classes = num_classes

    def forward(self, x):
        # Handle 5D input [B, 1, C, H, W] -> [B, C, H, W]
        if x.dim() == 5:
            x = x.squeeze(1)
        
        # Get segmentation map [B, num_classes, H, W]
        seg_map = self.base(x)
        
        # Extract center pixel prediction
        h, w = seg_map.shape[2], seg_map.shape[3]
        center_h, center_w = h // 2, w // 2
        
        # Get center pixel logits [B, num_classes]
        output = seg_map[:, :, center_h, center_w]
        
        return output


# Register model
from .registry import register_model


@register_model('SACNet', conv_features=64, trans_features=32, K=48, D=32)
def sacnet_model(pretrained: bool = False, **kwargs) -> SACNet:
    """Constructs a SACNet model with center-pixel extraction for patch classification."""
    if 'bands' in kwargs:
        kwargs['num_features'] = kwargs.pop('bands')
    kwargs.pop('patch_size', None)
    return SACNet(**kwargs)
