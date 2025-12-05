"""
MASSFormer: Memory-Augmented Spectral–Spatial Transformer for Hyperspectral Image Classification

Paper: https://ieeexplore.ieee.org/abstract/document/10506482
GitHub: https://github.com/hz63/MASSFormer
Venue: IEEE Transactions on Geoscience and Remote Sensing (TGRS)
Year: 2024
"""

import torch
from torch import nn
import torch.nn.functional as F
from einops import rearrange, repeat


class FeedForward(nn.Module):
    def __init__(self, dim, hidden_dim, dropout = 0.1):
        super().__init__()
        self.layernorm = nn.LayerNorm(dim)
        self.nn1 = nn.Linear(dim, hidden_dim)
        self.gelu = nn.GELU()
        self.drop = nn.Dropout(dropout)
        self.nn2 = nn.Linear(hidden_dim, dim)

    def forward(self, x):
        x = self.layernorm(x)
        x = self.nn1(x)
        x = self.gelu(x)
        x = self.drop(x)
        x = self.nn2(x)
        x = self.drop(x)
        return x
    

class MAA(nn.Module):
    def __init__(self, dim, heads = 8, dim_head = 64, dropout = 0.):
        super().__init__()
        inner_dim = dim_head *  heads

        self.heads = heads
        self.scale = dim_head ** -0.5
        self.norm = nn.LayerNorm(dim)

        self.attend = nn.Softmax(dim = -1)
        self.dropout = nn.Dropout(dropout)

        self.to_q = nn.Linear(dim, inner_dim, bias = False)
        self.to_kv = nn.Linear(dim, inner_dim * 2, bias = False)
        self.to_memory = nn.Linear(dim, inner_dim, bias = False)

        self.to_out = nn.Sequential(
            nn.Linear(inner_dim, dim),
            nn.Dropout(dropout)
        )

    def forward(self, x, memories):
        x = self.norm(x)
        x_kv = x 

        q, k, v = (self.to_q(x), *self.to_kv(x_kv).chunk(2, dim = -1))
        memories = self.to_memory(memories)

        q, k, v = map(lambda t: rearrange(t, 'b n (h d) -> b h n d', h = self.heads), (q, k, v))
        memories = rearrange(memories, 'b n (h d) -> b h n d', h = self.heads)
        
        k = torch.cat((k, memories), dim = 2)
        v = torch.cat((v, memories), dim = 2)

        dots = torch.einsum('bhid,bhjd->bhij', q, k) * self.scale
        attn = self.attend(dots)
        attn = self.dropout(attn)
        out = torch.einsum('bhij,bhjd->bhid', attn, v)
        out = rearrange(out, 'b h n d -> b n (h d)')
        out = self.to_out(out)
        return out


class Transformer(nn.Module):
    def __init__(self, dim, depth, heads, dim_head, mlp_dim, dropout = 0.1):
        super().__init__()
        self.layers = nn.ModuleList([])
        for _ in range(depth):
            self.layers.append(nn.ModuleList([
                MAA(dim, heads, dim_head, dropout),
                FeedForward(dim, mlp_dim, dropout)
            ]))

    def forward(self, x, memories):
        for _, (attn, ff) in enumerate(self.layers):
            x = attn(x, memories = memories) + x
            x = ff(x) + x

        return x

class massformer(nn.Module):
    def __init__(
        self,
        *,
        num_classes = 0,
        num_tokens = 0,
        dim = 64,
        depth = 2,
        heads = 8,
        dim_head = 8,
        mlp_dim = 512,
        dropout = 0.2,
        emb_dropout = 0.1,
        bands = 30,
        patch_size = 13,
    ):
        super().__init__()
        
        self.dim = dim
        self.depth = depth
        self.heads = heads
        self.dim_head = dim_head
        self.mlp_dim = mlp_dim
        self.emb_dropout = emb_dropout
        self.num_classes = num_classes

        self.conv3d = nn.Sequential(
            nn.Conv3d(in_channels=1, out_channels=8, kernel_size=(3, 3, 3)),
            nn.ReLU(),
        )
        self.flatten = nn.Flatten(1, 2)
        
        # Calculate dimensions after conv3d
        # Input: (B, 1, bands, patch_size, patch_size)
        # After conv3d: (B, 8, bands-2, patch_size-2, patch_size-2)
        conv3d_out_bands = bands - 2
        conv3d_out_spatial = patch_size - 2
        conv2d_in_channels = 8 * conv3d_out_bands
        
        self.conv2d = nn.Sequential(
            nn.Conv2d(in_channels=conv2d_in_channels, out_channels=64, kernel_size=(3, 3)),
            nn.ReLU(),
        )
        
        # After conv2d: spatial becomes (patch_size-2-2) = patch_size-4
        conv2d_out_spatial = conv3d_out_spatial - 2
        actual_num_tokens = conv2d_out_spatial * conv2d_out_spatial
        
        self.nn = nn.Linear(actual_num_tokens, dim)
        torch.nn.init.xavier_uniform_(self.nn.weight)
        self.dropout = nn.Dropout(dropout)

        self.cls_token = nn.Parameter(torch.zeros(1, 1, dim))
        self.pos_embedding = nn.Parameter(torch.empty(1, actual_num_tokens + 1, dim))
        torch.nn.init.normal_(self.pos_embedding, std=.001)
        
        self.transformer = Transformer(dim, depth, heads, dim_head, mlp_dim, emb_dropout)

        self.mlp_head = nn.Sequential(
            nn.LayerNorm(dim),
            nn.Linear(dim, num_classes)
        )

        self.drop = nn.Dropout(emb_dropout)

        # Dynamic pooling based on actual token count
        # Ensure kernel sizes don't exceed input size
        max_kernel = max(1, actual_num_tokens // 3)
        avg_kernel = max(1, actual_num_tokens // 5)
        self.maxpool = nn.MaxPool2d(kernel_size=(min(7, max_kernel), 1), stride=(max(1, min(5, max_kernel-2)), 1))
        self.avgpool = nn.AvgPool2d(kernel_size=(min(3, avg_kernel), 1), stride=(max(1, min(3, avg_kernel)), 1))


    def forward(self, x):
        x = self.conv3d(x)
        x = self.flatten(x)
        x = self.conv2d(x)
        x = rearrange(x,'b c h w -> b (h w) c')        

        max_token= self.maxpool(x)
        avg_token = self.avgpool(x)
        memories = torch.cat((avg_token, max_token), dim=1)
        memories = self.drop(memories)   
        
        cls_token = repeat(self.cls_token, '1 n d -> b n d', b = x.shape[0])
        x = torch.cat((cls_token, x), dim = 1)
        x += self.pos_embedding
        x = self.dropout(x)
        
        x = self.transformer(x,memories)

        token = x[:, 0]
        x = self.mlp_head(token)

        return x


if __name__ == '__main__':
    inputs = torch.randn(8, 1, 30, 13, 13)
    logits = massformer(num_classes=9, num_tokens=81)(inputs)
    print(logits.size())


# Register model
from .registry import register_model


@register_model('MASSFormer', dim=64, depth=2, heads=8, dim_head=8, mlp_dim=512, dropout=0.2, emb_dropout=0.1)
def massformer_model(pretrained: bool = False, **kwargs) -> massformer:
    """Constructs a MASSFormer model."""
    # bands and patch_size are now used
    return massformer(**kwargs)
