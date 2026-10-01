from models.registry import register_model
"""
HSIC-FM: Hyperspectral Image Classification Full Model

Paper: Overcoming the Barrier of Incompleteness: A Hyperspectral Image Classification Full Model
GitHub: https://github.com/YichuXu/HSIC-FM (assumed)
Venue: IEEE TGRS
Year: 2023

This model combines RNN-based spectral processing with Transformer-based spatial processing.
It expects dual inputs: spectral (1D) and spatial (2D) features.
"""

import torch
from torch import nn, einsum
from einops import rearrange
from einops.layers.torch import Reduce


def exists(val):
    return val is not None


class LayerNorm(nn.Module):
    def __init__(self, dim, eps=1e-5):
        super().__init__()
        self.eps = eps
        self.g = nn.Parameter(torch.ones(1, dim, 1, 1))
        self.b = nn.Parameter(torch.zeros(1, dim, 1, 1))

    def forward(self, x):
        std = torch.var(x, dim=1, unbiased=False, keepdim=True).sqrt()
        mean = torch.mean(x, dim=1, keepdim=True)
        return (x - mean) / (std + self.eps) * self.g + self.b


def FeedForward(dim, mult=4, dropout=0.):
    return nn.Sequential(
        LayerNorm(dim),
        nn.Conv2d(dim, dim * mult, 1),
        nn.GELU(),
        nn.Dropout(dropout),
        nn.Conv2d(dim * mult, dim, 1)
    )


class LocalMHRA(nn.Module):
    """Local Multi-Head Relative Attention"""
    def __init__(self, dim, heads, dim_head=64, local_aggr_kernel=5):
        super().__init__()
        self.heads = heads
        inner_dim = dim_head * heads

        self.norm = nn.BatchNorm2d(dim)
        self.to_v = nn.Conv2d(dim, inner_dim, 1, bias=False)
        self.rel_pos = nn.Conv2d(heads, heads, local_aggr_kernel, 
                                 padding=local_aggr_kernel // 2, groups=heads)
        self.to_out = nn.Conv2d(inner_dim, dim, 1)

    def forward(self, x):
        x = self.norm(x)
        b, c, *_, h = *x.shape, self.heads

        # to values
        v = self.to_v(x)

        # split out heads
        v = rearrange(v, 'b (c h) ... -> (b c) h ...', h=h)

        # aggregate by relative positions
        out = self.rel_pos(v)
        out = rearrange(out, '(b c) h ... -> b (c h) ...', b=b)

        return self.to_out(out)


class GlobalMHRA(nn.Module):
    """Global Multi-Head Relative Attention"""
    def __init__(self, dim, heads, dim_head=64, dropout=0.):
        super().__init__()
        self.heads = heads
        self.scale = dim_head ** -0.5
        inner_dim = dim_head * heads

        self.norm = LayerNorm(dim)
        self.to_qkv = nn.Conv1d(dim, inner_dim * 3, 1, bias=False)
        self.to_out = nn.Conv1d(inner_dim, dim, 1)

    def forward(self, x):
        x = self.norm(x)
        shape, h = x.shape, self.heads

        x = rearrange(x, 'b c ... -> b c (...)')

        q, k, v = self.to_qkv(x).chunk(3, dim=1)
        q, k, v = map(lambda t: rearrange(t, 'b (h d) n -> b h n d', h=h), (q, k, v))

        q = q * self.scale
        sim = einsum('b h i d, b h j d -> b h i j', q, k)

        attn = sim.softmax(dim=-1)

        out = einsum('b h i j, b h j d -> b h i d', attn, v)
        out = rearrange(out, 'b h n d -> b (h d) n', h=h)
        out = self.to_out(out)

        return out.view(*shape)


class Transformer(nn.Module):
    def __init__(self, dim, depth, heads, mhsa_type='g', local_aggr_kernel=5,
                 dim_head=64, ff_mult=4, ff_dropout=0., attn_dropout=0.):
        super().__init__()
        self.layers = nn.ModuleList([])

        for _ in range(depth):
            if mhsa_type == 'l':
                attn = LocalMHRA(dim, heads=heads, dim_head=dim_head, 
                               local_aggr_kernel=local_aggr_kernel)
            elif mhsa_type == 'g':
                attn = GlobalMHRA(dim, heads=heads, dim_head=dim_head, 
                                dropout=attn_dropout)
            else:
                raise ValueError('unknown mhsa_type')

            self.layers.append(nn.ModuleList([
                nn.Conv2d(dim, dim, (3, 3), padding=1),
                attn,
                FeedForward(dim, mult=ff_mult, dropout=ff_dropout),
            ]))

    def forward(self, x):
        attnMap = None
        for dpe, attn, ff in self.layers:
            x = dpe(x) + x
            attnMap = attn(x)
            x = attn(x) + x
            x = ff(x) + x
        return x, attnMap


class HSIC_FM(nn.Module):
    """
    Hyperspectral Image Classification Full Model
    
    Combines RNN-based spectral processing with Transformer-based spatial processing.
    """
    def __init__(self, num_classes, bands, patch_size,
                 dims=(64, 128, 256, 512),
                 depths=(3, 4, 8, 3),
                 mhsa_types=('l', 'l', 'g', 'g'),
                 channels=1,
                 ff_mult=4,
                 dim_head=64,
                 ff_dropout=0.,
                 attn_dropout=0.,
                 rnn_type='lstm',
                 rnn_hidden=64,
                 rnn_depth=1,
                 numseq=1):
        super().__init__()
        
        self.num_classes = num_classes
        self.bands = bands
        self.patch_size = patch_size
        self.rnn_hidden = rnn_hidden
        self.dims = dims
        
        # RNN for spectral processing
        if rnn_type == 'lstm':
            net = nn.LSTM
        elif rnn_type == 'gru':
            net = nn.GRU
        elif rnn_type == 'rnn':
            net = nn.RNN
        else:
            raise ValueError(f"Unknown rnn_type: {rnn_type}")
        
        self.recurrent = net(1, rnn_hidden, rnn_depth)
        
        # Feature sizes - will be computed dynamically based on actual feature dimensions
        # Placeholder for batch norm and fc layers
        self.bn_Down = None
        self.fc_Down = None
        self.down_proj = None  # For downsampling stage features
        
        # Spatial processing with Transformers
        init_dim, *_, last_dim = dims
        self.to_tokens = nn.Conv2d(channels, init_dim, (3, 3), padding=(1, 1))
        self.to_tokens3 = nn.MaxPool2d((3, 3), padding=1, stride=1)

        mhsa_types = tuple(map(lambda t: t.lower(), mhsa_types))
        self.stages = nn.ModuleList([])

        for ind, (depth, mhsa_type) in enumerate(zip(depths, mhsa_types)):
            is_last = ind == len(depths) - 1
            stage_dim = dims[ind]
            heads = stage_dim // dim_head

            self.stages.append(nn.ModuleList([
                Transformer(
                    dim=stage_dim,
                    depth=depth,
                    heads=heads,
                    mhsa_type=mhsa_type,
                    ff_mult=ff_mult,
                    ff_dropout=ff_dropout,
                    attn_dropout=attn_dropout
                ),
                nn.Sequential(
                    LayerNorm(stage_dim),
                    nn.Conv2d(stage_dim, dims[ind + 1], (1, 1), stride=(1, 1)),
                ) if not is_last else None
            ]))

        # Output layers
        self.to_logits1 = nn.Sequential(
            Reduce('b c h w -> b c', 'mean'),
            nn.LayerNorm(last_dim),
        )
        
        self.to_logits = nn.Sequential(
            nn.LayerNorm(512),
            nn.Linear(512, num_classes)
        )
        
        self.tanh = nn.Tanh()

    def _init_dynamic_layers(self, feat_size):
        """Initialize layers that depend on feature dimensions"""
        if self.bn_Down is None:
            # track_running_stats must stay True (the default). With it set to
            # False, BatchNorm1d always normalises with *batch* statistics --
            # even under .eval() -- so a batch of 1 (e.g. the trailing test
            # batch) raises "Expected more than 1 value per channel". Keeping
            # running stats is what actually makes batch_size=1 work.
            self.bn_Down = nn.BatchNorm1d(feat_size).to(next(self.parameters()).device)
            self.fc_Down = nn.Linear(feat_size, 512).to(next(self.parameters()).device)
            # These layers are built on the first forward, i.e. *after* any
            # .eval() call, so they would default to training=True and make
            # BatchNorm use batch stats (crashing on a batch of 1). Align them
            # with the parent module's current mode.
            self.bn_Down.train(self.training)
            self.fc_Down.train(self.training)
    
    def _init_down_proj(self, in_features):
        """Initialize downsampling projection layer"""
        if self.down_proj is None:
            self.down_proj = nn.Linear(in_features, 64).to(next(self.parameters()).device)

    def forward(self, x):
        """
        Args:
            x: Input tensor of shape (B, 1, bands, H, W) for 4D input
               or (B, bands, H, W) for 3D input
        
        Returns:
            logits: Classification logits of shape (B, num_classes)
        """
        # Handle both 3D and 4D inputs
        if x.dim() == 4:
            # (B, bands, H, W) -> add channel dimension
            x = x.unsqueeze(1)  # (B, 1, bands, H, W)
        
        B, C, bands, H, W = x.shape
        
        # Extract spectral and spatial features
        # Spectral: take center pixel across all bands
        center_h, center_w = H // 2, W // 2
        x_spe = x[:, :, :, center_h, center_w]  # (B, 1, bands)
        
        # Spatial: average across bands
        x_spa = x.mean(dim=2)  # (B, 1, H, W)
        
        # Process spatial features with Transformers
        x_spatial = self.to_tokens(x_spa)
        x_spatial = self.to_tokens3(x_spatial)

        # Store intermediate features for fusion
        stage_features = []
        for transformer, conv in self.stages:
            x_spatial, attnMap = transformer(x_spatial)
            stage_features.append(x_spatial)
            
            if exists(conv):
                x_spatial = conv(x_spatial)

        # Process spectral features with RNN
        x_spe = x_spe.transpose(1, 2).contiguous()  # (B, bands, 1)
        x_spe = x_spe.transpose(0, 1).contiguous()  # (bands, B, 1)
        
        x_RNN = self.recurrent(x_spe)[0]  # (bands, B, hidden)
        x_RNN = x_RNN.permute(1, 0, 2).contiguous()  # (B, bands, hidden)
        
        # Downsample stage features for fusion
        Down2 = stage_features[1].reshape(B, stage_features[1].shape[1], -1).transpose(1, 2)
        Down2_flat = Down2.reshape(B * Down2.shape[1], -1)
        
        # Initialize projection layer if needed
        self._init_down_proj(Down2_flat.shape[1])
        Down2_proj = self.down_proj(Down2_flat)
        Down2_proj = Down2_proj.reshape(B, Down2.shape[1], -1)
        
        # Fuse spectral and spatial features
        x_RNN = torch.cat([x_RNN, Down2_proj], 1)
        x_RNN = x_RNN.view(B, -1)
        
        # Initialize dynamic layers if needed
        self._init_dynamic_layers(x_RNN.shape[1])
        
        x_RNN = self.tanh(self.bn_Down(x_RNN))
        x_RNN = self.fc_Down(x_RNN)
        
        # Spatial features
        x_spatial = self.to_logits1(x_spatial)
        
        # Combine features
        x = x_RNN + x_spatial
        
        # Final classification
        x = self.to_logits(x)
        
        return x


@register_model('HSIC_FM', expects_4d=True, 
                dims=(64, 128, 256, 512),
                depths=(3, 4, 8, 3),
                mhsa_types=('l', 'l', 'g', 'g'),
                rnn_type='lstm',
                rnn_hidden=64,
                rnn_depth=1,
                numseq=1)
def hsic_fm(pretrained: bool = False, **kwargs) -> HSIC_FM:
    """
    Constructs a HSIC-FM model.
    
    Args:
        pretrained: Not used (no pretrained weights available)
        **kwargs: Additional arguments including num_classes, bands, patch_size
    
    Returns:
        HSIC_FM model instance
    """
    return HSIC_FM(**kwargs)
