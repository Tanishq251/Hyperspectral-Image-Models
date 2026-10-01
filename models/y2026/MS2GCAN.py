"""
MS2GCAN: Multi-Scale Spiking Graph Convolutional Network for Hyperspectral Image Classification

Paper/Source: https://github.com/fwx0314/MS2GCAN
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

from models.registry import register_model

# Try to import spikingjelly dependencies; use a mock/deferred error pattern if unavailable
HAS_SPIKINGJELLY = True
try:
    from spikingjelly.activation_based import neuron, functional, surrogate, layer
except ImportError:
    HAS_SPIKINGJELLY = False
    neuron = None
    functional = None
    surrogate = None
    layer = None


class MultiScaleSpikingFeaturesExtraction(nn.Module):
    def __init__(self, T: int = 3, img_size=15, hidden: int = 64):
        super().__init__()
        if not HAS_SPIKINGJELLY:
            raise ImportError(
                "spikingjelly is required to instantiate MS2GCAN. "
                "Please run `pip install spikingjelly` in your environment."
            )
        self.T = T 
        self.img_size = img_size
        self.hidden = hidden
        
        self.b0 = layer.Conv2d(hidden, hidden, kernel_size=1)
        self.b1 = nn.Sequential(
            layer.Conv2d(hidden, hidden, kernel_size=3, padding=1, groups=hidden),
            layer.BatchNorm2d(hidden),
            neuron.LIFNode(decay_input=False, detach_reset=True, surrogate_function=surrogate.ATan()),
            layer.Conv2d(hidden, hidden, kernel_size=1),
        )
        self.b2 = nn.Sequential(
            layer.Conv2d(hidden, hidden, kernel_size=5, padding=2, groups=hidden),
            layer.BatchNorm2d(hidden),
            neuron.LIFNode(decay_input=False, detach_reset=True, surrogate_function=surrogate.ATan()),
            layer.Conv2d(hidden, hidden, kernel_size=1),
        )
        
        self.gate = nn.Sequential(
            nn.LayerNorm(hidden * 3),
            layer.Linear(hidden * 3, hidden),
            nn.SiLU(),
            layer.Linear(hidden, 3),
            nn.Softmax(-1)
        )

        self.ms_bn = layer.BatchNorm1d(hidden)
        self.ms_lif = neuron.LIFNode(decay_input=False, detach_reset=True, surrogate_function=surrogate.ATan())

        self.proj = layer.Linear(hidden * 2, hidden)
        self.proj_bn = layer.BatchNorm1d(hidden)
        self.proj_lif = neuron.LIFNode(decay_input=False, detach_reset=True, surrogate_function=surrogate.ATan())

    def forward(self, x: torch.Tensor):
        T, N, L, C = x.shape
        x_spa = x.permute(0, 1, 3, 2).reshape(T, N, C, self.img_size, self.img_size)

        y0 = self.b0(x_spa).permute(0, 1, 3, 4, 2).reshape(T, N, L, C)
        y1 = self.b1(x_spa).permute(0, 1, 3, 4, 2).reshape(T, N, L, C)
        y2 = self.b2(x_spa).permute(0, 1, 3, 4, 2).reshape(T, N, L, C)

        m = torch.stack([y0, y1, y2], dim=2)
        m_cat = torch.cat([y0, y1, y2], dim=-1)
        m_mean = (m_cat.mean(dim=-2))
        weights = self.gate(m_mean).unsqueeze(-1).unsqueeze(-1)

        m = (weights * m).sum(dim=2)
        m = self.ms_bn(m.permute(0, 1, 3, 2)).permute(0, 1, 3, 2)
        m = self.ms_lif(m)

        out = torch.cat([x, m], dim=-1)
        out = self.proj(out)
        out = self.proj_bn(out.permute(0, 1, 3, 2)).permute(0, 1, 3, 2)
        out = self.proj_lif(out)
        return out


class GraphConvLayer(nn.Module):
    def __init__(self, in_features, out_features, bias=False):
        super().__init__()
        self.linear = layer.Linear(in_features, out_features, bias=bias)

    def forward(self, x, adj):
        x = self.linear(x)
        x = torch.matmul(adj, x)
        return x


class MS2GCAN(nn.Module):
    def __init__(self, T: int = 3, img_size=15, num_cls=16, input_dim=144, hidden: int = 32, 
                 gcn_layers: int = 2, K_hop: int = 2, use_cupy=False):
        super().__init__()
        if not HAS_SPIKINGJELLY:
            raise ImportError(
                "spikingjelly is required to instantiate MS2GCAN. "
                "Please run `pip install spikingjelly` in your environment."
            )
        self.T = T 
        self.img_size = img_size
        self.hidden = hidden
        self.l_squared = img_size * img_size

        self.fc1 = layer.Linear(input_dim, hidden * 2)
        self.fc1_bn = layer.BatchNorm1d(hidden * 2)
        self.fc1_lif = neuron.LIFNode(decay_input=False, detach_reset=True, surrogate_function=surrogate.ATan())
        self.fc2 = layer.Linear(hidden * 2, hidden)
        self.fc2_bn = layer.BatchNorm1d(hidden)
        self.fc2_lif = neuron.LIFNode(decay_input=False, detach_reset=True, surrogate_function=surrogate.ATan())

        self.mse = MultiScaleSpikingFeaturesExtraction(T=T, img_size=img_size, hidden=hidden)

        self.gcn_layers = nn.ModuleList([
            GraphConvLayer(hidden, hidden) for _ in range(gcn_layers)
        ])
        self.gcn_bns = nn.ModuleList([
            layer.BatchNorm1d(hidden) for _ in range(gcn_layers)
        ])
        self.gcn_lifs = nn.ModuleList([
            neuron.LIFNode(decay_input=False, detach_reset=True, surrogate_function=surrogate.ATan())
            for _ in range(gcn_layers)
        ])

        self.aggregator = KHopsCenterAggregator(K_hop, hidden, True, 1e-6)
        self.fusion_bn = layer.BatchNorm1d(hidden)
        self.logits = nn.Parameter(torch.zeros(T))

        self.out_linear = nn.Linear(hidden, num_cls)

        functional.set_step_mode(self, step_mode='m')
        if use_cupy:
            functional.set_backend(self, backend='cupy')
        
        self.init_weights()
    
    def forward(self, x: torch.Tensor):
        functional.reset_net(self)

        # The reference model consumes flattened spectral tokens (N, L^2, S).
        # The library's loader emits (B, 1, bands, H, W) (or (B, bands, H, W)),
        # so fold the spatial grid into the token axis and put bands last.
        if x.dim() == 5:            # (B, 1, bands, H, W) -> (B, bands, H, W)
            x = x.squeeze(1)
        if x.dim() == 4:            # (B, bands, H, W) -> (B, H*W, bands)
            b, bands, h, w = x.shape
            x = x.reshape(b, bands, h * w).transpose(1, 2)

        N, l_squared, S = x.shape
        x_seq = x.unsqueeze(0).expand(self.T, -1, -1, -1)

        x_seq = self.fc1(x_seq)
        x_seq = self.fc1_bn(x_seq.permute(0, 1, 3, 2)).permute(0, 1, 3, 2)
        x_seq = self.fc1_lif(x_seq)
        x_seq = self.fc2(x_seq)
        x_seq = self.fc2_bn(x_seq.permute(0, 1, 3, 2)).permute(0, 1, 3, 2)
        x_seq = self.fc2_lif(x_seq)

        x_seq = self.mse(x_seq)

        gcn_out = x_seq
        adj = self.AdaptivePearsonAdj(gcn_out)

        for gcn_layer, gcn_bn, gcn_lif in zip(self.gcn_layers, self.gcn_bns, self.gcn_lifs):
            res = gcn_out
            gcn_out = gcn_layer(gcn_out, adj)
            gcn_out = gcn_bn(gcn_out.permute(0, 1, 3, 2)).permute(0, 1, 3, 2)
            gcn_out = gcn_out + res
            gcn_out = gcn_lif(gcn_out)

        center_idx = int((self.l_squared - 1) / 2)
        x_fused = self.aggregator(gcn_out, adj, center_idx)
        x_center = x_fused.unsqueeze(-2)
        x_center = self.fusion_bn(x_center.permute(0, 1, 3, 2)).squeeze(-1)
        alpha = torch.softmax(self.logits, dim=0)
        x_center = (alpha.view(-1, 1, 1) * x_center).sum(dim=0)

        out = self.out_linear(x_center)
        return out

    def AdaptivePearsonAdj(self, spike: torch.Tensor):
        mu = spike.mean(dim=-1, keepdim=True)
        centered_features = spike - mu
        feature_norm = F.normalize(centered_features, p=2, dim=-1)
        feature_norm_t = feature_norm.transpose(-2, -1)
        sim = torch.matmul(feature_norm, feature_norm_t)
        adj = F.relu(sim)
        deg = adj.sum(-1, keepdim=True)
        deg_inv_sqrt = torch.pow(deg + 1e-6, -0.5)
        adj_norm = adj * deg_inv_sqrt * deg_inv_sqrt.transpose(-2, -1)
        return adj_norm
    
    def init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear) or isinstance(m, layer.Linear):
                torch.nn.init.normal_(m.weight, mean=0.0, std=0.02)
                if m.bias is not None:
                    torch.nn.init.zeros_(m.bias)
            elif isinstance(m, nn.Conv2d) or isinstance(m, layer.Conv2d):
                torch.nn.init.normal_(m.weight, mean=0.0, std=0.02)
                if m.bias is not None:
                    torch.nn.init.zeros_(m.bias)
            elif isinstance(m, nn.BatchNorm1d) or isinstance(m, layer.BatchNorm1d) or isinstance(m, layer.BatchNorm2d):
                torch.nn.init.ones_(m.weight)
                torch.nn.init.zeros_(m.bias)


class KHopsCenterAggregator(nn.Module):
    def __init__(self, K: int = 3, hidden: int = 64, include_self_in_hops: bool = False, eps: float = 1e-6):
        super().__init__()
        assert K >= 0
        self.K = K
        self.eps = eps
        self.include_self_in_hops = include_self_in_hops
        self.linear = layer.Linear(hidden, hidden)
        self.gate = nn.Sequential(
            nn.LayerNorm(hidden * (K + 1)),
            layer.Linear(hidden * (K + 1), hidden),
            nn.SiLU(),
            layer.Linear(hidden, K + 1),
            nn.Softmax(dim=-1)
        )

    def forward(self, gcn_out: torch.Tensor, adj: torch.Tensor, center_idx: int) -> torch.Tensor:
        gcn_out = self.linear(gcn_out)

        if self.K == 0:
            return gcn_out[:, :, center_idx, :]
        
        T, N, L2, H = gcn_out.shape
        device = adj.device
        
        adj_powers = [torch.eye(L2, device=device).unsqueeze(0).repeat(T * N, 1, 1)]
        adj_flat = adj.reshape(T * N, L2, L2)
        
        current_power = adj_flat.clone()
        for k in range(1, self.K + 1):
            adj_powers.append(current_power)
            if k < self.K:
                current_power = torch.bmm(current_power, adj_flat)
        
        center_masks = []
        for k in range(self.K + 1):
            mask = adj_powers[k][:, center_idx, :]
            
            if not self.include_self_in_hops and k > 0:
                zero_mask = torch.zeros_like(mask)
                indices = torch.arange(L2, device=device) != center_idx
                zero_mask[:, indices] = mask[:, indices]
                mask = zero_mask
            
            mask = mask / (mask.sum(dim=-1, keepdim=True) + self.eps)
            center_masks.append(mask)
        
        feats = []
        gcn_flat = gcn_out.reshape(T * N, L2, H)
        
        for k, mask in enumerate(center_masks):
            weighted_feat = torch.bmm(mask.unsqueeze(1), gcn_flat).squeeze(1)
            feats.append(weighted_feat)
        
        feat_stack = torch.stack(feats, dim=-1)
        feat_cat = torch.cat(feats, dim=-1)

        weights = self.gate(feat_cat).unsqueeze(-2)
        fused = (weights * feat_stack).sum(-1)
        return fused.view(T, N, H)


@register_model('MS2GCAN', expects_4d=False, T=3, hidden=32, gcn_layers=2, K_hop=2)
def ms2gcan_model(pretrained: bool = False, **kwargs) -> MS2GCAN:
    """Constructs a MS2GCAN model."""
    if not HAS_SPIKINGJELLY:
        raise ImportError(
            "spikingjelly is required to instantiate MS2GCAN. "
            "Please run `pip install spikingjelly` in your environment."
        )
    if 'bands' in kwargs:
        kwargs['input_dim'] = kwargs.pop('bands')
    if 'num_classes' in kwargs:
        kwargs['num_cls'] = kwargs.pop('num_classes')
    if 'patch_size' in kwargs:
        kwargs['img_size'] = kwargs.pop('patch_size')
    return MS2GCAN(**kwargs)
