import numpy as np
import torch
import torch.nn as nn
from timm.layers import DropPath
import torch.nn.functional as F
from attn import Hswish, DCA3D, WCCA3D
from kan import KAN


class PatchEmbed3D(nn.Module):
    """
    3D Patch embedding block
    """

    def __init__(self, in_chans=3, in_dim=64, dim=96):
        """
        Args:
            in_chans: number of input channels.
            in_dim: intermediate dimension.
            dim: output feature size dimension.
        """
        super().__init__()
        self.proj = nn.Identity()
        self.conv_down = nn.Sequential(
            nn.Conv3d(in_chans, in_dim, 3, 2, 1, bias=False),
            nn.BatchNorm3d(in_dim, eps=1e-4),
            nn.ReLU(),
            nn.Conv3d(in_dim, dim, 3, 1, 1, bias=False),
            nn.BatchNorm3d(dim, eps=1e-4),
            nn.ReLU()
            )

    def forward(self, x):
        x = self.proj(x)
        x = self.conv_down(x)
        return x


class Downsample3D(nn.Module):
    def __init__(self,
                 dim,
                 keep_dim=False,
                 ):
        super().__init__()
        if keep_dim:
            dim_out = dim
        else:
            dim_out = 2 * dim
        self.reduction = nn.Sequential(
            nn.Conv3d(dim, dim_out, 3, 2, 1, bias=False),
        )

    def forward(self, x):
        x = self.reduction(x)
        return x


class ConvBN3D(torch.nn.Sequential):
    def __init__(self, inp, oup, ks=1, stride=1, pad=0, dilation=1,
                 groups=1, bn_weight_init=1, resolution=-10000):
        super().__init__()
        self.add_module('c', torch.nn.Conv3d(
            inp, oup, ks, stride, pad, dilation, groups, bias=False))
        self.add_module('bn', torch.nn.BatchNorm3d(oup))
        torch.nn.init.constant_(self.bn.weight, bn_weight_init)
        torch.nn.init.constant_(self.bn.bias, 0)


class DWMixer(torch.nn.Module):
    def __init__(self, ed) -> None:
        super().__init__()
        self.conv = ConvBN3D(ed, ed, 3, 1, 1, groups=ed)
        self.conv1 = torch.nn.Conv3d(ed, ed, 1, 1, 0, groups=ed)
        self.dim = ed
        self.bn = torch.nn.BatchNorm3d(ed)
        self.ac = Hswish()
    
    def forward(self, x):
        return self.ac(self.bn((self.conv(x) + self.conv1(x)) + x))


class MixerBlock(nn.Module):
    def __init__(self, inp, group, drop_path=0.):
        super(MixerBlock, self).__init__()
        self.group = group

        self.token_mixer = nn.Sequential(
            DWMixer(inp),
            DCA3D(inp),
        )

        self.dw_conv = ConvBN3D(inp, inp, ks=3, stride=1, pad=1, groups=inp)
        self.down_conv = ConvBN3D(inp, inp // group, ks=1, stride=1)
        self.sigmoid = nn.Sigmoid()
        self.group_conv = ConvBN3D(inp, inp, ks=3, stride=1, pad=1, groups=group)
        self.bn = nn.BatchNorm3d(inp // group)
        self.ac = Hswish()

        self.expend_mixer = nn.Sequential(
            ConvBN3D(inp // group, inp, ks=1, stride=1, pad=0),
            ConvBN3D(inp, inp, ks=3, stride=1, pad=1, groups=inp),
            Hswish()
        )

        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()

    def forward(self, x):
        out1 = x + self.drop_path(self.token_mixer(x))

        out2 = self.dw_conv(out1)
        attn = self.sigmoid(self.down_conv(out2))
        out2 = self.group_conv(out2)

        b, c, d, h, w = out2.shape
        out2 = out2.reshape(b, self.group, c // self.group, d, h, w)
        out2 = self.ac(self.bn(torch.mean(out2, dim=1) * attn))

        out3 = out1 + self.drop_path(self.expend_mixer(out2))

        return out3


# from timm.models.vision_transformer import trunc_normal_
def trunc_normal_(tensor, mean=0., std=1., a=-2., b=2.):
    """
    Initialize tensor with truncated normal distribution.

    Args:
        tensor (torch.Tensor): Tensor to be initialized.
        mean (float): Mean of the distribution.
        std (float): Standard deviation of the distribution.
        a (float): Lower bound for truncation.
        b (float): Upper bound for truncation.
    """
    with torch.no_grad():
        # Number of elements in the tensor
        num_elements = tensor.numel()

        # Generate values from a normal distribution
        values = torch.normal(mean, std, size=(num_elements,))

        # Apply truncation
        values = values.clamp(min=a, max=b)

        # Reshape values to fit the tensor shape and assign them
        tensor.copy_(values.view_as(tensor))


class MixerNet(nn.Module):
    def __init__(self,
                in_channel,
                in_dim,
                dim,
                num_classes,
                drop_path_rate,
                depth,
                group
                ):
        super(MixerNet, self).__init__()
        dprs = [x.item() for x in torch.linspace(0, drop_path_rate, sum(depth))]
        self.patch_embed = PatchEmbed3D(in_chans=in_channel, in_dim=in_dim, dim=dim)

        cur_layer = 0
        layer_dim = dim
        block = MixerBlock
        self.layers = nn.ModuleList()
        for idx, repeat in enumerate(depth):
            level = nn.Sequential(
                *[block(
                    inp=dim * 2 ** idx,
                    group=group[idx],
                    drop_path=dprs[cur_layer]
                ) for _ in range(repeat)]
            )
            if idx < 3:
                level.append(Downsample3D(dim=dim * 2 ** idx))
                layer_dim *= 2
            self.layers.append(level)
            cur_layer += repeat

        self.avgpool = nn.AdaptiveAvgPool3d(1)
        self.gender_emb = nn.Embedding(num_embeddings=2, embedding_dim=layer_dim)
        self.fuse_emb = nn.Linear(layer_dim * 2, layer_dim)
        self.classifier = KAN([layer_dim, dim, num_classes])

    def forward(self, x, sex):
        x = self.patch_embed(x)
        
        features = []
        for layer in self.layers:
            x = layer(x)
            features.append(x.clone().detach())

        x = self.avgpool(x).flatten(1)
        sex = self.gender_emb(sex)
        x = torch.cat([x, sex], dim=1)
        x = self.fuse_emb(x)
        features.append(x.clone().detach())
        x = self.classifier(x)
        features.append(x.clone().detach())
        return x, features


def mixer_custom(num_classes=1):
    return MixerNet(
        in_channel=1,
        in_dim=32,
        dim=64,
        num_classes=num_classes,
        drop_path_rate=0.,
        depth=[1, 2, 4, 2],
        group=[2, 2, 4, 4]
    )

if __name__ == '__main__':
    x = torch.rand(1, 1, 96, 114, 96).to(torch.float32)
    sex = torch.tensor([0]).to(torch.int64)
    model = mixer_custom(num_classes=1)
    y, feats = model(x, sex)
    print(y)
    print(len(feats))
    for i in feats:
        print(i.shape)

    from torchinfo import summary
    summary(mixer_custom(num_classes=1), input_data=(x, sex))

    # x = torch.rand(1, 128, 12, 14, 12)
    # layer = MixerBlock(128, 8, 2, 0.2)
    # y = layer(x)
    # print(y.shape)
