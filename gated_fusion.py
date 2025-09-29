import torch
from torch import nn
from torch.nn import functional as F


class GatedTransFusionLayer(nn.Module):
    def __init__(self, in_ch, downsample=True):
        super().__init__()
        self.conv1 = nn.Conv3d(2 * in_ch, in_ch, kernel_size=1)
        self.bn = nn.BatchNorm3d(in_ch)
        self.k = nn.Conv3d(in_ch, in_ch, kernel_size=1)
        self.v = nn.Conv3d(in_ch, in_ch, kernel_size=1)
        self.q = nn.Conv3d(in_ch, in_ch, kernel_size=1)
        self.ln = nn.LayerNorm(in_ch)
        self.alpha = nn.Parameter(torch.ones(in_ch, 1, 1, 1))
        self.beta = nn.Parameter(torch.zeros(in_ch, 1, 1, 1))
        self.sigmoid = nn.Sigmoid()
        self.gelu = nn.GELU(approximate='tanh')

        if downsample:
            self.downsample = nn.Sequential(
                nn.Conv3d(in_ch, 2 * in_ch, kernel_size=3, stride=2, padding=1),
                nn.BatchNorm3d(2 * in_ch),
                nn.GELU(approximate='tanh')
            )
        else:
            self.downsample = nn.Sequential(
                nn.Conv3d(in_ch, in_ch, kernel_size=1, stride=1, padding=0),
                nn.BatchNorm3d(in_ch),
                nn.GELU(approximate='tanh')
            )

    def forward(self, x1, x2, m):
        g = torch.cat((x1, x2), dim=1)
        g = self.bn(self.conv1(g))
        m = m * self.sigmoid(g)

        k = self.k(x2).permute(0, 2, 3, 4, 1)
        v = self.v(x1).permute(0, 2, 3, 4, 1)
        k = self.ln(k).permute(0, 4, 1, 2, 3)
        v = self.ln(v).permute(0, 4, 1, 2, 3)
        m = m + self.sigmoid(k * v)

        q = self.q(x1)
        out = q * self.sigmoid(m)

        r = self.alpha * x1 + self.beta * x2
        out = r * self.sigmoid(out)
        out = self.gelu(self.bn(x1 + out))

        m = self.downsample(m)

        return out, m


class GatedAgeFusion(nn.Module):
    def __init__(self, dim, num_heads=8, qkv_bias=False):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.ln = nn.LayerNorm(self.head_dim)
        self.ln2 = nn.LayerNorm(dim)
        self.ac = nn.GELU(approximate="tanh")

        self.qv = nn.Linear(dim, 2 * dim, bias=qkv_bias)
        self.k1 = nn.Linear(dim, dim, bias=qkv_bias)
        self.k2 = nn.Linear(dim, dim, bias=qkv_bias)
        self.reduce = nn.Linear(dim, 1, bias=qkv_bias)
        self.expand = nn.Linear(2, dim, bias=qkv_bias)

        self.conv1 = nn.Conv1d(self.num_heads, self.num_heads, 3, 1, 1, bias=False)
        self.conv2 = nn.Conv1d(self.num_heads, self.num_heads, 3, 1, 1, bias=False)

    def forward(self, x1, x2, m, a):
        k1 = self.k1(m)
        k1 = k1.reshape(-1, self.num_heads, self.head_dim)
        k1_norm = self.ln(k1)

        k2 = self.reduce(x2)
        k2 = torch.cat((k2, a), dim=1)
        k2 = self.expand(k2) + x2
        k2 = k2.reshape(-1, self.num_heads, self.head_dim)
        k2_norm = self.ln(k2)

        qv = self.qv(x1)
        q, v = torch.chunk(qv, chunks=2, dim=1)
        q = q.reshape(-1, self.num_heads, self.head_dim)
        v = v.reshape(-1, self.num_heads, self.head_dim)
        q_norm = self.ln(q)

        out1 = F.scaled_dot_product_attention(q_norm, k1_norm, v)
        out2 = F.scaled_dot_product_attention(q_norm, k2_norm, v)

        out1 = self.ln(self.conv1(out1))
        out2 = self.ln(self.conv2(out2))
        out = self.ac(self.ln(out1 + out2 + v))

        out = out.reshape(-1, self.num_heads * self.head_dim)
        out = self.ac(self.ln2(x1 + out))

        return out


if __name__ == '__main__':

    x1, x2, m = torch.rand(3, 2, 512).unbind(0)
    a = torch.tensor([[1.], [2.]], dtype=torch.float32)

    layer = GatedAgeFusion(dim=512, num_heads=16)
    y = layer(x1, x2, m, a)
    print(y.shape)
