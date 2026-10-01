import torch
import torch.nn as nn
import torch.nn.functional as F
from models.registry import register_model

class SPCModuleIN(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.s1 = nn.Conv3d(in_channels, out_channels, kernel_size=(7, 1, 1), stride=(2, 1, 1), bias=False)

    def forward(self, x):
        x = x.unsqueeze(1)
        out = self.s1(x)
        return out.squeeze(1)

class ResSPC(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.spc1 = nn.Sequential(
            nn.Conv3d(in_channels, in_channels, kernel_size=(7, 1, 1), padding=(3, 0, 0), bias=False),
            nn.LeakyReLU(inplace=True),
            nn.BatchNorm3d(in_channels),
        )
        self.spc2 = nn.Sequential(
            nn.Conv3d(in_channels, in_channels, kernel_size=(7, 1, 1), padding=(3, 0, 0), bias=False),
            nn.LeakyReLU(inplace=True),
        )
        self.bn2 = nn.BatchNorm3d(out_channels)

    def forward(self, x):
        out = self.spc1(x)
        out = self.bn2(self.spc2(out))
        return F.leaky_relu(out + x)

class SPAModuleIN(nn.Module):
    def __init__(self, in_channels, out_channels, k):
        super().__init__()
        self.s1 = nn.Conv3d(in_channels, out_channels, kernel_size=(k, 3, 3), bias=False)

    def forward(self, x):
        out = self.s1(x)
        out = out.squeeze(2)
        return out

class ResSPA(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.spa1 = nn.Sequential(
            nn.Conv2d(in_channels, in_channels, kernel_size=3, padding=1),
            nn.LeakyReLU(inplace=True),
            nn.BatchNorm2d(in_channels),
        )
        self.spa2 = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.LeakyReLU(inplace=True),
        )
        self.bn2 = nn.BatchNorm2d(out_channels)

    def forward(self, x):
        out = self.spa1(x)
        out = self.bn2(self.spa2(out))
        return F.leaky_relu(out + x)

class SSRN(nn.Module):
    def __init__(self, num_classes, bands=200, patch_size=7):
        super().__init__()
        # Calculate size after Conv3D downsampling to determine k for SPAModuleIN
        # Spectral dimension is halved by self.layer1's stride (2,1,1)
        k = (bands - 7) // 2 + 1
        
        self.layer1 = SPCModuleIN(1, 28)
        self.layer2 = ResSPC(28, 28)
        self.layer3 = ResSPC(28, 28)
        self.layer4 = SPAModuleIN(28, 28, k=k)
        self.bn4 = nn.BatchNorm2d(28)
        self.layer5 = ResSPA(28, 28)
        self.layer6 = ResSPA(28, 28)
        self.fc = nn.Linear(28, num_classes)

    def forward(self, x):
        # layer1 (SPCModuleIN) unsqueezes to 5D internally, so it must be fed a
        # 4D (B, bands, H, W) tensor. The loader hands us 5D (B, 1, bands, H, W);
        # unsqueezing here as well produced a 6D tensor and crashed conv3d.
        if x.dim() == 5:
            x = x.squeeze(1)
        x = F.leaky_relu(self.layer1(x))
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.bn4(F.leaky_relu(self.layer4(x)))
        x = self.layer5(x)
        x = self.layer6(x)
        x = F.avg_pool2d(x, x.size()[-1])
        # flatten(1) rather than a bare squeeze(): squeeze() also drops the batch
        # dim when batch size is 1 (e.g. a trailing test batch).
        x = self.fc(x.flatten(1))
        return x

@register_model('SSRN')
def ssrn_model(pretrained: bool = False, **kwargs) -> SSRN:
    """Constructs an SSRN model."""
    kwargs.pop('patch_size', None)
    return SSRN(**kwargs)
