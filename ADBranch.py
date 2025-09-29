from timm.layers import layer_scale
import torch
import torch.nn as nn
from timm.models.registry import register_model
import math
from timm.models.layers import trunc_normal_, DropPath, LayerNorm2d
from timm.models._builder import resolve_pretrained_cfg
try:
    from timm.models._builder import _update_default_kwargs as update_args
except:
    from timm.models._builder import _update_default_model_kwargs as update_args
from timm.models.vision_transformer import Mlp, PatchEmbed
from timm.models.layers import DropPath, trunc_normal_
import torch.nn.functional as F
from mamba_ssm.ops.selective_scan_interface import selective_scan_fn
from einops import rearrange, repeat
from pathlib import Path
from attn import DCA3D
from gated_fusion import GatedTransFusionLayer, GatedAgeFusion
from kan import KAN


def window_partition_3d(x, window_size):
    """
    Args:
        x: (B, C, D, H, W)
        window_size: window size
    Returns:
        local window features (num_windows*B, window_size*window_size*window_size, C)
    """
    B, C, D, H, W = x.shape
    x = x.view(B, C, 
               D // window_size, window_size, 
               H // window_size, window_size, 
               W // window_size, window_size)
    windows = x.permute(0, 2, 4, 6, 3, 5, 7, 1).reshape(
        -1, window_size*window_size*window_size, C)
    return windows


def window_reverse_3d(windows, window_size, D, H, W):
    """
    Args:
        windows: local window features (num_windows*B, window_size*window_size*window_size, C)
        window_size: Window size
        D: Depth of volume
        H: Height of volume
        W: Width of volume
    Returns:
        x: (B, C, D, H, W)
    """
    B = int(windows.shape[0] / (D * H * W / window_size / window_size / window_size))
    x = windows.reshape(
        B, D // window_size, H // window_size, W // window_size, 
        window_size, window_size, window_size, -1)
    x = x.permute(0, 7, 1, 4, 2, 5, 3, 6).reshape(B, windows.shape[2], D, H, W)
    return x


class Downsample3D(nn.Module):
    """
    3D Down-sampling block
    """

    def __init__(self,
                 dim,
                 keep_dim=False,
                 ):
        """
        Args:
            dim: feature size dimension.
            keep_dim: bool argument for maintaining the resolution.
        """

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


class ConvBlock3D(nn.Module):
    def __init__(self, dim,
                 drop_path=0.,
                 layer_scale=None,
                 kernel_size=3):
        super().__init__()

        self.conv1 = nn.Conv3d(dim, dim, kernel_size=kernel_size, stride=1, padding=1)
        self.norm1 = nn.BatchNorm3d(dim, eps=1e-5)
        self.act1 = nn.GELU(approximate='tanh')
        self.conv2 = nn.Conv3d(dim, dim, kernel_size=kernel_size, stride=1, padding=1)
        self.norm2 = nn.BatchNorm3d(dim, eps=1e-5)
        self.layer_scale = layer_scale
        if layer_scale is not None and type(layer_scale) in [int, float]:
            self.gamma = nn.Parameter(layer_scale * torch.ones(dim))
            self.layer_scale = True
        else:
            self.layer_scale = False
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.dca = DCA3D(in_ch=dim)

    def forward(self, x):
        input = x
        x = self.conv1(x)
        x = self.norm1(x)
        x = self.act1(x)
        x = self.conv2(x)
        x = self.norm2(x)
        if self.layer_scale:
            x = x * self.gamma.view(1, -1, 1, 1, 1)
        x = self.dca(x)
        x = input + self.drop_path(x)
        return x


class MambaVisionMixer(nn.Module):
    def __init__(
        self,
        d_model,
        d_state=16,
        d_conv=4,
        expand=2,
        dt_rank="auto",
        dt_min=0.001,
        dt_max=0.1,
        dt_init="random",
        dt_scale=1.0,
        dt_init_floor=1e-4,
        conv_bias=True,
        bias=False,
        use_fast_path=True, 
        layer_idx=None,
        device=None,
        dtype=None,
    ):
        factory_kwargs = {"device": device, "dtype": dtype}
        super().__init__()
        self.d_model = d_model
        self.d_state = d_state
        self.d_conv = d_conv
        self.expand = expand
        self.d_inner = int(self.expand * self.d_model)
        self.dt_rank = math.ceil(self.d_model / 16) if dt_rank == "auto" else dt_rank
        self.use_fast_path = use_fast_path
        self.layer_idx = layer_idx
        self.in_proj = nn.Linear(self.d_model, self.d_inner, bias=bias, **factory_kwargs)    
        self.x_proj = nn.Linear(
            self.d_inner//2, self.dt_rank + self.d_state * 2, bias=False, **factory_kwargs
        )
        self.dt_proj = nn.Linear(self.dt_rank, self.d_inner//2, bias=True, **factory_kwargs)
        dt_init_std = self.dt_rank**-0.5 * dt_scale
        if dt_init == "constant":
            nn.init.constant_(self.dt_proj.weight, dt_init_std)
        elif dt_init == "random":
            nn.init.uniform_(self.dt_proj.weight, -dt_init_std, dt_init_std)
        else:
            raise NotImplementedError
        dt = torch.exp(
            torch.rand(self.d_inner//2, **factory_kwargs) * (math.log(dt_max) - math.log(dt_min))
            + math.log(dt_min)
        ).clamp(min=dt_init_floor)
        inv_dt = dt + torch.log(-torch.expm1(-dt))
        with torch.no_grad():
            self.dt_proj.bias.copy_(inv_dt)
        self.dt_proj.bias._no_reinit = True
        A = repeat(
            torch.arange(1, self.d_state + 1, dtype=torch.float32, device=device),
            "n -> d n",
            d=self.d_inner//2,
        ).contiguous()
        A_log = torch.log(A)
        self.A_log = nn.Parameter(A_log)
        self.A_log._no_weight_decay = True
        self.D = nn.Parameter(torch.ones(self.d_inner//2, device=device))
        self.D._no_weight_decay = True
        self.out_proj = nn.Linear(self.d_inner, self.d_model, bias=bias, **factory_kwargs)
        self.conv1d_x = nn.Conv1d(
            in_channels=self.d_inner//2,
            out_channels=self.d_inner//2,
            bias=conv_bias//2,
            kernel_size=d_conv,
            groups=self.d_inner//2,
            **factory_kwargs,
        )
        self.conv1d_z = nn.Conv1d(
            in_channels=self.d_inner//2,
            out_channels=self.d_inner//2,
            bias=conv_bias//2,
            kernel_size=d_conv,
            groups=self.d_inner//2,
            **factory_kwargs,
        )

    def forward(self, hidden_states):
        """
        hidden_states: (B, L, D)
        Returns: same shape as hidden_states
        """
        _, seqlen, _ = hidden_states.shape
        xz = self.in_proj(hidden_states)
        xz = rearrange(xz, "b l d -> b d l")
        x, z = xz.chunk(2, dim=1)
        A = -torch.exp(self.A_log.float())
        x = F.silu(F.conv1d(input=x, weight=self.conv1d_x.weight, bias=self.conv1d_x.bias, padding='same', groups=self.d_inner//2))
        z = F.silu(F.conv1d(input=z, weight=self.conv1d_z.weight, bias=self.conv1d_z.bias, padding='same', groups=self.d_inner//2))
        x_dbl = self.x_proj(rearrange(x, "b d l -> (b l) d"))
        dt, B, C = torch.split(x_dbl, [self.dt_rank, self.d_state, self.d_state], dim=-1)
        dt = rearrange(self.dt_proj(dt), "(b l) d -> b d l", l=seqlen)
        B = rearrange(B, "(b l) dstate -> b dstate l", l=seqlen).contiguous()
        C = rearrange(C, "(b l) dstate -> b dstate l", l=seqlen).contiguous()
        y = selective_scan_fn(x, 
                              dt, 
                              A, 
                              B, 
                              C, 
                              self.D.float(), 
                              z=None, 
                              delta_bias=self.dt_proj.bias.float(), 
                              delta_softplus=True, 
                              return_last_state=None)
        
        y = torch.cat([y, z], dim=1)
        y = rearrange(y, "b d l -> b l d")
        out = self.out_proj(y)
        return out
    

class Attention(nn.Module):

    def __init__(
            self,
            dim,
            num_heads=8,
            qkv_bias=False,
            qk_norm=False,
            attn_drop=0.,
            proj_drop=0.,
            norm_layer=nn.LayerNorm,
    ):
        super().__init__()
        assert dim % num_heads == 0
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.scale = self.head_dim ** -0.5
        self.fused_attn = True

        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.q_norm = norm_layer(self.head_dim) if qk_norm else nn.Identity()
        self.k_norm = norm_layer(self.head_dim) if qk_norm else nn.Identity()
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

    def forward(self, x):
        B, N, C = x.shape
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, self.head_dim).permute(2, 0, 3, 1, 4)
        q, k, v = qkv.unbind(0)
        q, k = self.q_norm(q), self.k_norm(k)

        if self.fused_attn:
            x = F.scaled_dot_product_attention(
             q, k, v,
                dropout_p=self.attn_drop.p,
            )
        else:
            q = q * self.scale
            attn = q @ k.transpose(-2, -1)
            attn = attn.softmax(dim=-1)
            attn = self.attn_drop(attn)
            x = attn @ v

        x = x.transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x


class Block(nn.Module):
    def __init__(self, 
                 dim, 
                 num_heads, 
                 counter, 
                 transformer_blocks, 
                 mlp_ratio=4., 
                 qkv_bias=False, 
                 qk_scale=False, 
                 drop=0., 
                 attn_drop=0.,
                 drop_path=0., 
                 act_layer=nn.GELU, 
                 norm_layer=nn.LayerNorm, 
                 Mlp_block=Mlp,
                 layer_scale=None,
                 ):
        super().__init__()
        self.norm1 = norm_layer(dim)
        if counter in transformer_blocks:
            self.mixer = Attention(
            dim,
            num_heads=num_heads,
            qkv_bias=qkv_bias,
            qk_norm=qk_scale,
            attn_drop=attn_drop,
            proj_drop=drop,
            norm_layer=norm_layer,
        )
        else:
            self.mixer = MambaVisionMixer(d_model=dim, 
                                          d_state=8,  
                                          d_conv=3,    
                                          expand=1
                                          )

        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.norm2 = norm_layer(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = Mlp_block(in_features=dim, hidden_features=mlp_hidden_dim, act_layer=act_layer, drop=drop)
        use_layer_scale = layer_scale is not None and type(layer_scale) in [int, float]
        self.gamma_1 = nn.Parameter(layer_scale * torch.ones(dim))  if use_layer_scale else 1
        self.gamma_2 = nn.Parameter(layer_scale * torch.ones(dim))  if use_layer_scale else 1

    def forward(self, x):
        x = x + self.drop_path(self.gamma_1 * self.mixer(self.norm1(x)))
        x = x + self.drop_path(self.gamma_2 * self.mlp(self.norm2(x)))
        return x


class MambaVisionLayer(nn.Module):
    """
    MambaVision layer"
    """

    def __init__(self,
                 dim,
                 depth,
                 num_heads,
                 window_size,
                 conv=False,
                 downsample=True,
                 mlp_ratio=4.,
                 qkv_bias=True,
                 qk_scale=None,
                 drop=0.,
                 attn_drop=0.,
                 drop_path=0.,
                 layer_scale=None,
                 layer_scale_conv=None,
                 transformer_blocks = [],
    ):
        """
        Args:
            dim: feature size dimension.
            depth: number of layers in each stage.
            window_size: window size in each stage.
            conv: bool argument for conv stage flag.
            downsample: bool argument for down-sampling.
            mlp_ratio: MLP ratio.
            num_heads: number of heads in each stage.
            qkv_bias: bool argument for query, key, value learnable bias.
            qk_scale: bool argument to scaling query, key.
            drop: dropout rate.
            attn_drop: attention dropout rate.
            drop_path: drop path rate.
            norm_layer: normalization layer.
            layer_scale: layer scaling coefficient.
            layer_scale_conv: conv layer scaling coefficient.
            transformer_blocks: list of transformer blocks.
        """

        super().__init__()
        self.conv = conv
        self.transformer_block = False
        if conv:
            self.blocks = nn.ModuleList([ConvBlock3D(dim=dim,
                                                   drop_path=drop_path[i] if isinstance(drop_path, list) else drop_path,
                                                   layer_scale=layer_scale_conv)
                                                   for i in range(depth)])
            self.transformer_block = False
        else:
            self.blocks = nn.ModuleList([Block(dim=dim,
                                               counter=i, 
                                               transformer_blocks=transformer_blocks,
                                               num_heads=num_heads,
                                               mlp_ratio=mlp_ratio,
                                               qkv_bias=qkv_bias,
                                               qk_scale=qk_scale,
                                               drop=drop,
                                               attn_drop=attn_drop,
                                               drop_path=drop_path[i] if isinstance(drop_path, list) else drop_path,
                                               layer_scale=layer_scale)
                                               for i in range(depth)])
            self.transformer_block = True

        self.downsample = None if not downsample else Downsample3D(dim=dim)
        self.do_gt = False
        self.window_size = window_size

    def forward(self, x):
        _, _, D, H, W = x.shape

        if self.transformer_block:
            # 计算三个维度上需要的填充
            pad_d = (self.window_size - D % self.window_size) % self.window_size
            pad_h = (self.window_size - H % self.window_size) % self.window_size
            pad_w = (self.window_size - W % self.window_size) % self.window_size
            
            # 应用3D填充
            if pad_d > 0 or pad_h > 0 or pad_w > 0:
                x = torch.nn.functional.pad(x, (0, pad_w, 0, pad_h, 0, pad_d))
                _, _, Dp, Hp, Wp = x.shape
            else:
                Dp, Hp, Wp = D, H, W
                
            # 使用3D窗口分区
            x = window_partition_3d(x, self.window_size)

        for _, blk in enumerate(self.blocks):
            x = blk(x)
            
        if self.transformer_block:
            # 使用3D窗口还原
            x = window_reverse_3d(x, self.window_size, Dp, Hp, Wp)
            
            # 如果有填充，则去除填充部分
            if pad_d > 0 or pad_h > 0 or pad_w > 0:
                x = x[:, :, :D, :H, :W].contiguous()
                
        if self.downsample is None:
            return x
        return self.downsample(x)


class MambaVision(nn.Module):
    """
    MambaVision,
    """

    def __init__(self,
                 dim,
                 in_dim,
                 depths,
                 window_size,
                 mlp_ratio,
                 num_heads,
                 drop_path_rate=0.2,
                 in_chans=1,
                 num_classes=1000,
                 qkv_bias=True,
                 qk_scale=None,
                 drop_rate=0.,
                 attn_drop_rate=0.,
                 layer_scale=None,
                 layer_scale_conv=None,
                 **kwargs):
        """
        Args:
            dim: feature size dimension.
            depths: number of layers in each stage.
            window_size: window size in each stage.
            mlp_ratio: MLP ratio.
            num_heads: number of heads in each stage.
            drop_path_rate: drop path rate.
            in_chans: number of input channels.
            num_classes: number of classes.
            qkv_bias: bool argument for query, key, value learnable bias.
            qk_scale: bool argument to scaling query, key.
            drop_rate: dropout rate.
            attn_drop_rate: attention dropout rate.
            norm_layer: normalization layer.
            layer_scale: layer scaling coefficient.
            layer_scale_conv: conv layer scaling coefficient.
        """
        super().__init__()
        num_features = int(dim * 2 ** (len(depths) - 1))
        self.num_classes = num_classes
        self.patch_embed = PatchEmbed3D(in_chans=in_chans, in_dim=in_dim, dim=dim)
        self.memory_embed = nn.Sequential(
            PatchEmbed3D(in_chans=in_chans, in_dim=in_dim, dim=dim),
            Downsample3D(dim=dim)
        )
        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, sum(depths))]

        layer_dim = dim
        self.levels = nn.ModuleList()
        self.gated_fusions = nn.ModuleList()

        for i in range(len(depths)):
            conv = True if (i == 0 or i == 1) else False
            level = MambaVisionLayer(dim=int(dim * 2 ** i),
                                     depth=depths[i],
                                     num_heads=num_heads[i],
                                     window_size=window_size[i],
                                     mlp_ratio=mlp_ratio,
                                     qkv_bias=qkv_bias,
                                     qk_scale=qk_scale,
                                     conv=conv,
                                     drop=drop_rate,
                                     attn_drop=attn_drop_rate,
                                     drop_path=dpr[sum(depths[:i]):sum(depths[:i + 1])],
                                     downsample=(i < 3),
                                     layer_scale=layer_scale,
                                     layer_scale_conv=layer_scale_conv,
                                     transformer_blocks=list(range(depths[i]//2+1, depths[i])) if depths[i]%2!=0 else list(range(depths[i]//2, depths[i])),
                                     )

            self.levels.append(level)
            if i < 3:
                self.gated_fusions.append(GatedTransFusionLayer(in_ch=int(2 * dim * 2 ** i), downsample=(i < 2)))
                layer_dim *= 2

        self.norm = nn.BatchNorm3d(num_features)
        self.avgpool = nn.AdaptiveAvgPool3d(1)
        self.gender_emb = nn.Embedding(num_embeddings=2, embedding_dim=512)
        self.fuse_emb = nn.Linear(layer_dim * 2, layer_dim)
        self.head = KAN([layer_dim, dim, num_classes])
        self.age_fusion = GatedAgeFusion(dim=layer_dim, num_heads=16, qkv_bias=False)
        # self.head = nn.Linear(num_features, num_classes) if num_classes > 0 else nn.Identity()
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)
        elif isinstance(m, LayerNorm2d):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)
        elif isinstance(m, nn.BatchNorm3d):
            nn.init.ones_(m.weight)
            nn.init.zeros_(m.bias)
        elif isinstance(m, nn.BatchNorm2d):
            nn.init.ones_(m.weight)
            nn.init.zeros_(m.bias)


    def forward(self, x, sex, age, fms):
        m = self.memory_embed(x)
        x = self.patch_embed(x)

        features = []
        for i in range(3):
            x = self.levels[i](x)
            x, m = self.gated_fusions[i](x, fms[i], m)
            features.append(x.clone().detach())

        x = self.levels[-1](x)
        features.append(x.clone().detach())

        x = self.norm(x)
        x = self.avgpool(x).flatten(1)

        m = self.avgpool(m).flatten(1)

        sex = self.gender_emb(sex)
        x = torch.cat([x, sex], dim=1)
        x = self.fuse_emb(x)
        features.append(x.clone().detach())

        # x = self.age_fusion(x, fms[-2], m, fms[-1])
        x = self.age_fusion(x, fms[-2], m, fms[-1] - age)
        # x = self.age_fusion(x, fms[-2], m, abs(fms[-1] - age))

        x = self.head(x)

        return x, features


def mamba_custom(pretrained=False, **kwargs):
    model_path = kwargs.pop("model_path", "/tmp/mamba_vision_T.pth.tar")
    in_chans = kwargs.pop("in_chans", 1)
    # depths = kwargs.pop("depths", [1, 3, 8, 4])
    depths = kwargs.pop("depths", [1, 1, 2, 2])
    num_heads = kwargs.pop("num_heads", [2, 4, 8, 16])
    num_classes = kwargs.pop("num_classes", 2)
    # window_size = kwargs.pop("window_size", [8, 8, 14, 7])
    window_size = kwargs.pop("window_size", [8, 8, 15, 8])
    dim = kwargs.pop("dim", 64)
    in_dim = kwargs.pop("in_dim", 32)
    mlp_ratio = kwargs.pop("mlp_ratio", 4)
    # resolution = kwargs.pop("resolution", 224)
    drop_path_rate = kwargs.pop("drop_path_rate", 0.25)
    layer_scale = kwargs.pop("layer_scale", 1.)
    # pretrained_cfg = resolve_pretrained_cfg('mamba_vision_T').to_dict()
    # update_args(pretrained_cfg, kwargs, kwargs_filter=None)
    model = MambaVision(depths=depths,
                        in_chans=in_chans,
                        num_heads=num_heads,
                        num_classes=num_classes,
                        window_size=window_size,
                        dim=dim,
                        in_dim=in_dim,
                        mlp_ratio=mlp_ratio,
                        drop_path_rate=drop_path_rate,
                        layer_scale=layer_scale
                        )
                        # resolution=resolution,
                        # **kwargs)
    # model.pretrained_cfg = pretrained_cfg
    # model.default_cfg = model.pretrained_cfg
    # if pretrained:
    #     if not Path(model_path).is_file():
    #         url = model.default_cfg['url']
    #         torch.hub.download_url_to_file(url=url, dst=model_path)
    #     model._load_state_dict(model_path)
    # print("Final kwargs:", kwargs)
    return model


if __name__ == '__main__':
    device = 'cuda:7'
    feas_list = [
        torch.rand(1, 128, 24, 29, 24),
        torch.rand(1, 256, 12, 15, 12),
        torch.rand(1, 512, 6, 8, 6),
        torch.rand(1, 512, 6, 8, 6),
        torch.rand(1, 512),
        torch.rand(1, 1)
    ]
    feas_on_device = [f.to(device) for f in feas_list]
    age = torch.tensor([1.], dtype=torch.float32).to(device)

    x = torch.rand(1, 1, 96, 114, 96).to(device)
    sex = torch.tensor([1], dtype=torch.int64).to(device)
    model = mamba_custom(num_classes=2, drop_path_rate=0.2).to(device)
    y, features = model(x, sex, age, feas_on_device)

    print(y.shape)
    for f in features:
        print(f.shape)

    from torchinfo import summary
    summary(model, input_data=(x, sex, age, feas_on_device))
