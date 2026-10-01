import math
import torch
import torch.nn as nn
from models.registry import register_model

class PAM_Module(nn.Module):
    def __init__(self, in_dim):
        super().__init__()
        self.query_conv = nn.Conv2d(in_channels=in_dim, out_channels=in_dim // 8, kernel_size=1)
        self.key_conv = nn.Conv2d(in_channels=in_dim, out_channels=in_dim // 8, kernel_size=1)
        self.value_conv = nn.Conv2d(in_channels=in_dim, out_channels=in_dim, kernel_size=1)
        self.gamma = nn.Parameter(torch.zeros(1))
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, x):
        x = x.squeeze(-1)
        m_batchsize, C, height, width = x.size()
        proj_query = self.query_conv(x).view(m_batchsize, -1, width * height).permute(0, 2, 1)
        proj_key = self.key_conv(x).view(m_batchsize, -1, width * height)
        energy = torch.bmm(proj_query, proj_key)
        attention = self.softmax(energy)
        proj_value = self.value_conv(x).view(m_batchsize, -1, width * height)
        out = torch.bmm(proj_value, attention.permute(0, 2, 1))
        out = out.view(m_batchsize, C, height, width)
        out = (self.gamma * out + x).unsqueeze(-1)
        return out

class CAM_Module(nn.Module):
    def __init__(self, in_dim):
        super().__init__()
        self.gamma = nn.Parameter(torch.zeros(1))
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, x):
        m_batchsize, C, height, width, channle = x.size()
        proj_query = x.view(m_batchsize, C, -1)
        proj_key = x.view(m_batchsize, C, -1).permute(0, 2, 1)
        energy = torch.bmm(proj_query, proj_key)
        energy_new = torch.max(energy, -1, keepdim=True)[0].expand_as(energy) - energy
        attention = self.softmax(energy_new)
        proj_value = x.view(m_batchsize, C, -1)
        out = torch.bmm(attention, proj_value)
        out = out.view(m_batchsize, C, height, width, channle)
        out = self.gamma * out + x
        return out

class DBDA(nn.Module):
    def __init__(self, num_classes, bands=200, patch_size=9):
        super().__init__()
        # conv11
        self.conv11 = nn.Conv3d(in_channels=1, out_channels=24, kernel_size=(1, 1, 7), stride=(1, 1, 2))
        self.batch_norm11 = nn.Sequential(
            nn.BatchNorm3d(24, eps=0.001, momentum=0.1, affine=True),
            nn.ReLU(inplace=True)
        )
        self.conv12 = nn.Conv3d(in_channels=24, out_channels=24, padding=(0, 0, 3), kernel_size=(1, 1, 7), stride=(1, 1, 1))
        self.batch_norm12 = nn.Sequential(
            nn.BatchNorm3d(48, eps=0.001, momentum=0.1, affine=True),
            nn.ReLU(inplace=True)
        )
        self.conv13 = nn.Conv3d(in_channels=48, out_channels=24, padding=(0, 0, 3), kernel_size=(1, 1, 7), stride=(1, 1, 1))
        self.batch_norm13 = nn.Sequential(
            nn.BatchNorm3d(72, eps=0.001, momentum=0.1, affine=True),
            nn.ReLU(inplace=True)
        )
        self.conv14 = nn.Conv3d(in_channels=72, out_channels=24, padding=(0, 0, 3), kernel_size=(1, 1, 7), stride=(1, 1, 1))
        self.batch_norm14 = nn.Sequential(
            nn.BatchNorm3d(96, eps=0.001, momentum=0.1, affine=True),
            nn.ReLU(inplace=True)
        )
        kernel_3d = math.floor((bands - 6) / 2)
        self.conv15 = nn.Conv3d(in_channels=96, out_channels=60, kernel_size=(1, 1, kernel_3d), stride=(1, 1, 1))

        self.conv21 = nn.Conv3d(in_channels=1, out_channels=24, kernel_size=(1, 1, bands), stride=(1, 1, 1))
        self.batch_norm21 = nn.Sequential(
            nn.BatchNorm3d(24, eps=0.001, momentum=0.1, affine=True),
            nn.ReLU(inplace=True)
        )
        self.conv22 = nn.Conv3d(in_channels=24, out_channels=12, padding=(1, 1, 0), kernel_size=(3, 3, 1), stride=(1, 1, 1))
        self.batch_norm22 = nn.Sequential(
            nn.BatchNorm3d(36, eps=0.001, momentum=0.1, affine=True),
            nn.ReLU(inplace=True)
        )
        self.conv23 = nn.Conv3d(in_channels=36, out_channels=12, padding=(1, 1, 0), kernel_size=(3, 3, 1), stride=(1, 1, 1))
        self.batch_norm23 = nn.Sequential(
            nn.BatchNorm3d(48, eps=0.001, momentum=0.1, affine=True),
            nn.ReLU(inplace=True)
        )
        self.conv24 = nn.Conv3d(in_channels=48, out_channels=12, padding=(1, 1, 0), kernel_size=(3, 3, 1), stride=(1, 1, 1))

        self.global_pooling = nn.AdaptiveAvgPool3d(1)
        self.full_connection = nn.Linear(120, num_classes)

        self.attention_spectral = CAM_Module(60)
        self.attention_spatial = PAM_Module(60)

    def forward(self, x):
        if x.dim() == 4:
            x = x.unsqueeze(1)
        
        # Permute [B, 1, C, H, W] to [B, 1, H, W, C] to match DBDA architecture expectations
        x = x.permute(0, 1, 3, 4, 2)
        
        # spectral branch
        x11 = self.conv11(x)
        x12 = self.batch_norm11(x11)
        x12 = self.conv12(x12)
        x13 = torch.cat((x11, x12), dim=1)
        x13 = self.batch_norm12(x13)
        x13 = self.conv13(x13)
        x14 = torch.cat((x11, x12, x13), dim=1)
        x14 = self.batch_norm13(x14)
        x14 = self.conv14(x14)
        x15 = torch.cat((x11, x12, x13, x14), dim=1)
        x16 = self.batch_norm14(x15)
        x16 = self.conv15(x16)

        x1 = self.attention_spectral(x16)
        x1 = torch.mul(x1, x16)

        # spatial branch
        x21 = self.conv21(x)
        x22 = self.batch_norm21(x21)
        x22 = self.conv22(x22)
        x23 = torch.cat((x21, x22), dim=1)
        x23 = self.batch_norm22(x23)
        x23 = self.conv23(x23)
        # Mirrors the spectral branch's `x14 = cat(x11, x12, x13)`. The reference
        # code cats a 4th `x24` here, which is the tensor being defined -> it
        # raised UnboundLocalError. Channel math confirms three inputs is right:
        # x21(24) + x22(12) + x23(12) = 48 = conv24.in_channels.
        x24 = torch.cat((x21, x22, x23), dim=1)
        x24 = self.batch_norm23(x24)
        x24 = self.conv24(x24)
        x25 = torch.cat((x21, x22, x23, x24), dim=1)

        x2 = self.attention_spatial(x25)
        x2 = torch.mul(x2, x25)

        # classification
        x1 = self.global_pooling(x1).squeeze(-1).squeeze(-1).squeeze(-1)
        x2 = self.global_pooling(x2).squeeze(-1).squeeze(-1).squeeze(-1)
        x_pre = torch.cat((x1, x2), dim=1)
        output = self.full_connection(x_pre)
        return output

@register_model('DBDA')
def dbda_model(pretrained: bool = False, **kwargs) -> DBDA:
    """Constructs a DBDA model."""
    kwargs.pop('patch_size', None)
    return DBDA(**kwargs)
