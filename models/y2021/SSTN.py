import torch
import torch.nn as nn
import torch.nn.functional as F
from models.registry import register_model

class SpatAttn(nn.Module):
    def __init__(self, in_dim, ratio=8):
        super().__init__()
        self.query_conv = nn.Conv2d(in_channels=in_dim, out_channels=in_dim//ratio, kernel_size=1)
        self.key_conv = nn.Conv2d(in_channels=in_dim, out_channels=in_dim//ratio, kernel_size=1)
        self.value_conv = nn.Conv2d(in_channels=in_dim, out_channels=in_dim, kernel_size=1)
        self.gamma = nn.Parameter(torch.zeros(1))
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, x):
        m_batchsize, C, height, width = x.size()
        proj_query = self.query_conv(x).view(m_batchsize, -1, width*height).permute(0, 2, 1)
        proj_key = self.key_conv(x).view(m_batchsize, -1, width*height)
        energy = torch.bmm(proj_query, proj_key)
        attention = self.softmax(energy)
        proj_value = self.value_conv(x).view(m_batchsize, -1, width*height)
        out = torch.bmm(proj_value, attention.permute(0, 2, 1))
        out = out.view(m_batchsize, C, height, width)
        return self.gamma * out + x

class SpatAttn_(nn.Module):
    def __init__(self, in_dim, ratio=8):
        super().__init__()
        self.query_conv = nn.Conv2d(in_channels=in_dim, out_channels=in_dim//ratio, kernel_size=1)
        self.key_conv = nn.Conv2d(in_channels=in_dim, out_channels=in_dim//ratio, kernel_size=1)
        self.value_conv = nn.Conv2d(in_channels=in_dim, out_channels=in_dim, kernel_size=1)
        self.gamma = nn.Parameter(torch.zeros(1))
        self.softmax = nn.Softmax(dim=-1)
        self.bn = nn.Sequential(nn.ReLU(), nn.BatchNorm2d(in_dim))

    def forward(self, x):
        m_batchsize, C, height, width = x.size()
        proj_query = self.query_conv(x).view(m_batchsize, -1, width*height).permute(0, 2, 1)
        proj_key = self.key_conv(x).view(m_batchsize, -1, width*height)
        energy = torch.bmm(proj_query, proj_key)
        attention = self.softmax(energy)
        proj_value = self.value_conv(x).view(m_batchsize, -1, width*height)
        out = torch.bmm(proj_value, attention.permute(0, 2, 1))
        out = out.view(m_batchsize, C, height, width)
        return self.bn(self.gamma * out)

class SARes(nn.Module):
    def __init__(self, in_dim, ratio=8, resin=False):
        super().__init__()
        if resin:
            self.sa1 = SpatAttn(in_dim, ratio)
            self.sa2 = SpatAttn(in_dim, ratio)
        else:
            self.sa1 = SpatAttn_(in_dim, ratio)
            self.sa2 = SpatAttn_(in_dim, ratio)

    def forward(self, x):
        identity = x
        x = self.sa1(x)
        x = self.sa2(x)
        return F.relu(x + identity)

class SPC32(nn.Module):
    def __init__(self, msize=16, outplane=49, kernel_size=None, padding=None):
        super().__init__()
        if kernel_size is None:
            kernel_size = [outplane, 1, 1]
        if padding is None:
            padding = [0, 0, 0]
        self.convm0 = nn.Conv3d(1, msize, kernel_size=kernel_size, padding=padding)
        self.bn1 = nn.BatchNorm2d(outplane)
        self.convm2 = nn.Conv3d(1, msize, kernel_size=kernel_size, padding=padding)
        self.bn2 = nn.BatchNorm2d(outplane)

    def forward(self, x, identity=None):
        if identity is None:
            identity = x
        n, c, h, w = identity.size()
        mask0 = self.convm0(x.unsqueeze(1)).squeeze(2)
        mask0 = torch.softmax(mask0.view(n, -1, h*w), -1).view(n, -1, h, w)
        fk = torch.einsum('ndhw,nchw->ncd', mask0, x)
        out = torch.einsum('ncd,ndhw->ncdhw', fk, mask0).sum(2)
        out0 = self.bn1(out.view(n, -1, h, w))
        
        mask2 = self.convm2(out0.unsqueeze(1)).squeeze(2)
        mask2 = torch.softmax(mask2.view(n, -1, h*w), -1).view(n, -1, h, w)
        fk = torch.einsum('ndhw,nchw->ncd', mask2, x)
        out = torch.einsum('ncd,ndhw->ncdhw', fk, mask2).sum(2)
        out = out + identity
        return self.bn2(out.view(n, -1, h, w))

class SSTN(nn.Module):
    def __init__(self, num_classes, bands=200, patch_size=9, msize=16, inter_size=49):
        super().__init__()
        self.layer1 = nn.Sequential(
            nn.Conv2d(bands, inter_size, 1),
            nn.BatchNorm2d(inter_size),
        )
        self.layer2 = SPC32(msize, outplane=inter_size, kernel_size=[inter_size, 1, 1], padding=[0, 0, 0])
        self.layer3 = SARes(inter_size, ratio=8)
        self.layer4 = nn.Conv2d(inter_size, msize, kernel_size=1)
        self.bn4 = nn.BatchNorm2d(msize)
        self.layer5 = SARes(msize, ratio=8)
        self.layer6 = SPC32(msize, outplane=msize, kernel_size=[msize, 1, 1], padding=[0, 0, 0])
        self.fc = nn.Linear(msize, num_classes)

    def forward(self, x):
        if x.dim() == 5:
            x = x.squeeze(1) # resnets expect 4D
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.bn4(F.leaky_relu(self.layer4(x)))
        x = self.layer5(x)
        x = self.layer6(x)
        x = F.avg_pool2d(x, x.size()[-1])
        # flatten(1) rather than a bare squeeze(): squeeze() also drops the batch
        # dim when batch size is 1, returning (num_classes,) instead of
        # (1, num_classes) and breaking the loss/metrics on trailing batches.
        x = self.fc(x.flatten(1))
        return x

@register_model('SSTN', expects_4d=True)
def sstn_model(pretrained: bool = False, **kwargs) -> SSTN:
    """Constructs an SSTN model."""
    kwargs.pop('patch_size', None)
    if 'bands' in kwargs:
        kwargs['bands'] = kwargs.get('bands')
    return SSTN(**kwargs)
