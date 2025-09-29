import torch
from torch import nn
from torch.nn import functional as F

class Hswish(nn.Module):
    def __init__(self, inplace=True):
        super(Hswish, self).__init__()
        self.inplace = inplace

    def forward(self, x):
        # H-swish 激活函数
        return x * F.relu6(x + 3., inplace=self.inplace) / 6.

# Dynamic Contrast Attention 3D
class DCA3D(nn.Module):
    def __init__(self, in_ch, eps=1e-6):
        super().__init__()
        self.alpha = nn.Parameter(torch.ones(in_ch, 1, 1, 1))
        self.beta = nn.Parameter(torch.zeros(in_ch, 1, 1, 1))
        self.gamma = nn.Parameter(torch.ones(in_ch, 1, 1, 1))
        self.eps = eps
        self.norm = nn.InstanceNorm3d(in_ch, affine=True)
        self.ac = Hswish()

    def forward(self, x):
        # local contrast
        local_mean = F.avg_pool3d(x, kernel_size=3, stride=1, padding=1)
        out = x + self.alpha * (x - local_mean)

        # global contrast
        global_max = torch.amax(x, dim=(2, 3, 4), keepdim=True)
        global_min = torch.amin(x, dim=(2, 3, 4), keepdim=True)
        scale_factor = self.gamma * (global_max - global_min + self.eps)
        out = scale_factor * out + self.beta
        out = self.ac(self.norm(out))
        return out

# Coordinate Attention 3D
class CA3D(nn.Module):
    def __init__(self, in_channels, reduction=16):
        super().__init__()
        mip = max(8, int(in_channels // reduction))

        self.pool_d = nn.AdaptiveAvgPool3d((None, 1, 1))
        self.pool_h = nn.AdaptiveAvgPool3d((1, None, 1))
        self.pool_w = nn.AdaptiveAvgPool3d((1, 1, None))
        self.conv1 = nn.Conv3d(in_channels, mip, kernel_size=1, stride=1, padding=0)
        self.conv_d = nn.Conv3d(mip, in_channels, kernel_size=1, stride=1, padding=0)
        self.conv_h = nn.Conv3d(mip, in_channels, kernel_size=1, stride=1, padding=0)
        self.conv_w = nn.Conv3d(mip, in_channels, kernel_size=1, stride=1, padding=0)

        self.ac = Hswish()
        self.sigmoid = nn.Sigmoid()
        self.norm = nn.BatchNorm3d(mip)

    def forward(self, x):
        n, c, d, h, w = x.size()
        x_d = self.pool_d(x)
        x_h = self.pool_h(x).permute(0, 1, 3, 2, 4)
        x_w = self.pool_w(x).permute(0, 1, 4, 3, 2)

        y1 = torch.cat([x_d, x_h, x_w], dim=2)
        y1 = self.ac(self.norm(self.conv1(y1)))

        x_d, x_h, x_w = torch.split(y1, [d, h, w], dim=2)
        x_h = x_h.permute(0, 1, 3, 2, 4)
        x_w = x_w.permute(0, 1, 4, 3, 2)

        a_d = self.sigmoid(self.conv_d(x_d))
        a_h = self.sigmoid(self.conv_h(x_h))
        a_w = self.sigmoid(self.conv_w(x_w))

        out = x * a_d * a_h * a_w
        return out

# Convolutional Block Attention Module 3D
class CBAM3D(nn.Module):
    def __init__(self, in_channels, kernel_size=5, reduction=16):
        super().__init__()
        # Channel Attention Module
        self.avg_pool = nn.AdaptiveAvgPool3d(1)
        self.max_pool = nn.AdaptiveMaxPool3d(1)
        self.fc1 = nn.Conv3d(in_channels, in_channels // reduction, 1, bias=False)
        self.ac = Hswish()
        self.fc2 = nn.Conv3d(in_channels // reduction, in_channels, 1, bias=False)

        # Spatial Attention Module
        self.conv1 = nn.Conv3d(2, 1, kernel_size, padding=kernel_size // 2, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        # Channel Attention Module
        avg_out = self.fc2(self.ac(self.fc1(self.avg_pool(x))))
        max_out = self.fc2(self.ac(self.fc1(self.max_pool(x))))
        out = self.sigmoid(avg_out + max_out)
        x = x * out

        # Spatial Attention Module
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        out = torch.cat([avg_out, max_out], dim=1)
        out = self.sigmoid(self.conv1(out))
        x = x * out
        return x

# Weighted Channel Coordinate Attention 3D
class WCCA3D(nn.Module):
    def __init__(self, in_channels, kernel_size=5, reduction=16):
        super().__init__()
        # weights
        self.alpha = nn.Parameter(torch.tensor(0.5))
        self.beta = nn.Parameter(torch.tensor(0.5))

        # branches
        self.ca = CA3D(in_channels, reduction)
        self.cbam = CBAM3D(in_channels, kernel_size, reduction)
        self.norm = nn.BatchNorm3d(in_channels)
        self.ac = Hswish()

    def forward(self, x):
        x1 = self.ca(x)
        x2 = self.cbam(x)
        out = self.alpha * x1 + self.beta * x2
        out = self.ac(self.norm(out))
        return out

if __name__ == '__main__':
    x = torch.rand((2, 128, 5, 5, 5), dtype=torch.float32)
    dca = WCCA3D(128)
    y = dca(x)
    print(y.shape)

