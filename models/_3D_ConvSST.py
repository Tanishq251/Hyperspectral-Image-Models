


"""
3DConvSST: 3D-Convolution Guided Spectral-Spatial Transformer for Hyperspectral Image Classification

Paper: https://ieeexplore.ieee.org/document/10605285
GitHub: https://github.com/ShyamVarahagiri/3D-ConvSST
Venue: IEEE Transactions on Geoscience and Remote Sensing (TGRS)
Year: 2024
"""

import torch
import torch.nn as nn


class HetConv(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1,padding = None, bias = None,p = 64, g = 64):
        super(HetConv, self).__init__()
        # Groupwise Convolution
        self.gwc = nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size,groups=g,padding = kernel_size//3, stride = stride)
        # Pointwise Convolution
        self.pwc = nn.Conv2d(in_channels, out_channels, kernel_size=1,groups=p, stride = stride)
    def forward(self, x):
        return self.gwc(x) + self.pwc(x)



class Attention(nn.Module):
    """Attention mechanism.

    Parameters
    ----------
    dim : int
        The input and out dimension of per token features.

    n_heads : int
        Number of attention heads.

    qkv_bias : bool
        If True then we include bias to the query, key and value projections.

    attn_p : float
        Dropout probability applied to the query, key and value tensors.

    proj_p : float
        Dropout probability applied to the output tensor.


    Attributes
    ----------
    scale : float
        Normalizing consant for the dot product.

    qkv : nn.Linear
        Linear projection for the query, key and value.

    proj : nn.Linear
        Linear mapping that takes in the concatenated output of all attention
        heads and maps it into a new space.

    attn_drop, proj_drop : nn.Dropout
        Dropout layers.
    """
    def __init__(self, dim, n_heads=12, qkv_bias=True, attn_p=0., proj_p=0.):
        super().__init__()
        self.n_heads = n_heads
        self.dim = dim
        self.head_dim = dim // n_heads
        self.scale = self.head_dim ** -0.5

        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_p)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_p)

    def forward(self, x):
        """Run forward pass.

        Parameters
        ----------
        x : torch.Tensor
            Shape `(n_samples, n_patches + 1, dim)`.

        Returns
        -------
        torch.Tensor
            Shape `(n_samples, n_patches + 1, dim)`.
        """
        n_samples, n_tokens, dim = x.shape

        if dim != self.dim:
            raise ValueError

        qkv = self.qkv(x)  # (samples, n_patches, 3 * dim)
        qkv = qkv.reshape(
                n_samples, n_tokens, 3, self.n_heads, self.head_dim
        )  # (samples, n_patches, 3, n_heads, head_dim)
        qkv = qkv.permute(
                2, 0, 3, 1, 4
        )  # (3, n_samples, n_heads, n_patches, head_dim)

        q, k, v = qkv[0], qkv[1], qkv[2]
        k_t = k.transpose(-2, -1)  # (n_samples, n_heads, head_dim, n_patches)
        dp = (
           q @ k_t
        ) * self.scale # (n_samples, n_heads, n_patches, n_patches)
        attn = dp.softmax(dim=-1)  # (n_samples, n_heads, n_patches, n_patches)
        attn = self.attn_drop(attn)

        weighted_avg = attn @ v  # (n_samples, n_heads, n_patches, head_dim)
        weighted_avg = weighted_avg.transpose(
                1, 2
        )  # (n_samples, n_patches + 1, n_heads, head_dim)
        weighted_avg = weighted_avg.flatten(2)  # (n_samples, n_patches, dim)

        x = self.proj(weighted_avg)  # (n_samples, n_patches, dim)
        x = self.proj_drop(x)  # (n_samples, n_patches, dim)

        return x


class MLP(nn.Module):
    """Multilayer perceptron.

    Parameters
    ----------
    in_features : int
        Number of input features.

    hidden_features : int
        Number of nodes in the hidden layer.

    out_features : int
        Number of output features.

    p : float
        Dropout probability.

    Attributes
    ----------
    fc : nn.Linear
        The First linear layer.

    act : nn.GELU
        GELU activation function.

    fc2 : nn.Linear
        The second linear layer.

    drop : nn.Dropout
        Dropout layer.
    """
    def __init__(self, in_features, hidden_features, out_features, p=0.):
        super().__init__()
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = nn.GELU()
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop = nn.Dropout(p)

    def forward(self, x):
        """Run forward pass.

        Parameters
        ----------
        x : torch.Tensor
            Shape `(n_samples, n_patches + 1, in_features)`.

        Returns
        -------
        torch.Tensor
            Shape `(n_samples, n_patches +1, out_features)`
        """
        x = self.fc1(
                x
        ) # (n_samples, n_patches, hidden_features)
        x = self.act(x)  # (n_samples, n_patches, hidden_features)
        x = self.drop(x)  # (n_samples, n_patches, hidden_features)
        x = self.fc2(x)  # (n_samples, n_patches, out_features)
        x = self.drop(x)  # (n_samples, n_patches, out_features)

        return x


class Block(nn.Module):
    """Transformer block.

    Parameters
    ----------
    dim : int
        Embeddinig dimension.

    n_heads : int
        Number of attention heads.

    mlp_ratio : float
        Determines the hidden dimension size of the `MLP` module with respect
        to `dim`.

    qkv_bias : bool
        If True then we include bias to the query, key and value projections.

    p, attn_p : float
        Dropout probability.

    Attributes
    ----------
    norm1, norm2 : LayerNorm
        Layer normalization.

    attn : Attention
        Attention module.

    mlp : MLP
        MLP module.
    """
    def __init__(self, dim, n_heads, mlp_ratio=4.0, qkv_bias=True, p=0., attn_p=0.):
        super().__init__()
        self.norm1 = nn.LayerNorm(dim, eps=1e-6)
        self.attn = Attention(
                dim,
                n_heads=n_heads,
                qkv_bias=qkv_bias,
                attn_p=attn_p,
                proj_p=p
        )
        self.norm2 = nn.LayerNorm(dim, eps=1e-6)
        hidden_features = int(dim * mlp_ratio)
        self.mlp = MLP(
                in_features=dim,
                hidden_features=hidden_features,
                out_features=dim,
        )

    def forward(self, x):
        """Run forward pass.

        Parameters
        ----------
        x : torch.Tensor
            Shape `(n_samples, n_patches + 1, dim)`.

        Returns
        -------
        torch.Tensor
            Shape `(n_samples, n_patches + 1, dim)`.
        """
        x = x + self.attn(self.norm1(x))
        x = x + self.mlp(self.norm2(x))

        return x


class VisionTransformer(nn.Module):
    """Simplified implementation of the Vision transformer.

    Parameters
    ----------

    in_chans : int
        Number of input channels.

    n_classes : int
        Number of classes.

    embed_dim : int
        Dimensionality of the token/patch embeddings.

    depth : int
        Number of blocks.

    n_heads : int
        Number of attention heads.

    mlp_ratio : float
        Determines the hidden dimension of the `MLP` module.

    qkv_bias : bool
        If True then we include bias to the query, key and value projections.

    p, attn_p : float
        Dropout probability.

    Attributes
    ----------
    patch_embed : PatchEmbed
        Instance of `PatchEmbed` layer.

    cls_token : nn.Parameter
        Learnable parameter that will represent the first token in the sequence.
        It has `embed_dim` elements.

    pos_emb : nn.Parameter
        Positional embedding of the cls token + all the patches.
        It has `(n_patches + 1) * embed_dim` elements.

    pos_drop : nn.Dropout
        Dropout layer.

    blocks : nn.ModuleList
        List of `Block` modules.

    norm : nn.LayerNorm
        Layer normalization.
    """
    def __init__(
            self,
            in_chans=3,
            # n_classes=1000,
            embed_dim=768,
            depth=8,
            n_heads=12,
            mlp_ratio=4.,
            qkv_bias=True,
            p=0.,
            attn_p=0.,
    ):
        super().__init__()

        self.conv5 = nn.Sequential(
            nn.Conv3d(1, 8, (9, 3, 3), padding=(0,1,1), stride = 1),
            nn.BatchNorm3d(8),
            nn.ReLU()
        )
        self.conv6 = nn.Sequential(
            HetConv(8 * (in_chans - 8), 64, p=1, g = 8),
            nn.BatchNorm2d(64),
            nn.ReLU()
        )
        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        self.pos_embed = nn.Parameter(
                torch.zeros(1, 121, embed_dim)
        )
        self.pos_drop = nn.Dropout(p=p)
        self.blocks = nn.ModuleList(
            [
                Block(
                    dim=embed_dim,
                    n_heads=n_heads,
                    mlp_ratio=mlp_ratio,
                    qkv_bias=qkv_bias,
                    p=p,
                    attn_p=attn_p,
                )
                for _ in range(depth)
            ]
        )
        self.convBlock = nn.Conv3d(64, 64, (2, 3, 3), padding=(0,1,1), stride = 1)
        self.norm = nn.LayerNorm(embed_dim, eps=1e-6)
        self.pool = nn.AvgPool1d(121)
        # self.head = nn.Linear(embed_dim, n_classes)


    def forward(self, x):
        """Run the forward pass.

        Parameters
        ----------
        x : torch.Tensor
            Shape `(n_samples, in_chans, img_size, img_size)`.

        Returns
        -------
        logits : torch.Tensor
            Logits over all the classes - `(n_samples, n_classes)`.
        """
        n_samples = x.shape[0]
        x = x.unsqueeze(1)
        x = self.conv5(x)
        x = x.reshape(x.shape[0],-1,11,11)
        x = self.conv6(x) # (samples, channels, h, w)
        x = x.reshape(x.shape[0], x.shape[1], -1) # (samples, embedDim, patches)
        x = x.transpose(1,2) # (samples, patches, embedDim)
        x = x + self.pos_embed  # (n_samples, n_patches, embed_dim)
        x = self.pos_drop(x)

        for block in self.blocks:
            temp = x.transpose(1, 2) # (samples, embedDim, patches)
            xNew = block(x) # (samples, patches, embedDim)
            xNew = xNew.transpose(1,2) # (samples, embedDim, patches)
            temp = temp.unsqueeze(2) # (samples, embedDim, 1, patches)
            temp = temp.reshape(temp.shape[0], temp.shape[1], 1, 11, 11) # (samples, embedDim, 1, 11, 11)
            xNew = xNew.unsqueeze(2) # (samples, embedDim, 1, patches)
            xNew = xNew.reshape(xNew.shape[0], xNew.shape[1], 1, 11, 11) # (samples, embedDim, 1, 11, 11)
            x = torch.cat((temp, xNew), dim=2) # (samples, embedDim, 2, 11, 11)
            x = self.convBlock(x) # (samples, embedDim, 1, 11, 11)
            x = x.reshape(x.shape[0], x.shape[1], 11, 11) # (samples, embedDim, 11, 11)
            x = x.reshape(x.shape[0], x.shape[1], -1) # (samples, embedDim, patches)
            x = x.transpose(1, 2) # (samples, patches, embedDim)

        x = self.norm(x) # (samples, patches, embedDim)
        x = x.transpose(1,2)
        x = self.pool(x)[:, :, 0] # (samples, embedDim)
        # x = self.head(x)


        return x



# Register model
from .registry import register_model

@register_model('3DConvSST', expects_4d=True, embed_dim=64, depth=5, n_heads=4, mlp_ratio=4.0, qkv_bias=True, p=0.1, attn_p=0.1)
def convst_3d(pretrained: bool = False, **kwargs) -> VisionTransformer:
    """Constructs a 3D ConvSST model."""
    # Map standardized names to model-specific names
    if 'bands' in kwargs:
        kwargs['in_chans'] = kwargs.pop('bands')
    # Remove unused parameters
    kwargs.pop('num_classes', None)
    kwargs.pop('patch_size', None)
    return VisionTransformer(**kwargs)
