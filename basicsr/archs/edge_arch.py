import numpy as np
import math
import torch
import torch.nn as nn
import torch.utils.checkpoint as checkpoint
import torch.nn.functional as F
from timm.layers import DropPath, to_2tuple, trunc_normal_, Mlp, PatchEmbed
from basicsr.utils.registry import ARCH_REGISTRY

from einops import rearrange
from collections import OrderedDict
from einops.layers.torch import Rearrange

def token2feature(x, x_size):
    B, N, C = x.shape
    h, w = x_size
    x = x.permute(0, 2, 1).reshape(B, C, h, w)
    return x

def feature2token(x):
    B, C, H, W = x.shape
    x = x.view(B, C, -1).transpose(1, 2)
    return x
class AttentionBlock(nn.Module):
    def __init__(
            self,
            dim: int,
            num_heads: int,
            mlp_ratio=4.,
            qkv_bias=False,
            qk_norm=False,
            proj_drop=0.,
            attn_drop=0.,
            drop_path=0.,
            act_layer=nn.GELU,
            norm_layer=nn.LayerNorm,
            mlp_layer=Mlp,
    ) -> None:
        super().__init__()
        self.norm1 = norm_layer(dim)
        self.attn = QKVAttention(
            dim,
            num_heads=num_heads,
            qkv_bias=qkv_bias,
            qk_norm=qk_norm,
            attn_drop=attn_drop,
            proj_drop=proj_drop,
            norm_layer=norm_layer, # 层归一化 q and k
        )

        self.drop_path1 = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.norm2 = norm_layer(dim)
        self.mlp = mlp_layer(
            in_features=dim,
            hidden_features=int(dim * mlp_ratio),
            act_layer=act_layer,
            drop=proj_drop,
        )

        self.drop_path2 = DropPath(drop_path) if drop_path > 0. else nn.Identity()
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.drop_path1(self.attn(self.norm1(x)))
        x = x + self.drop_path2(self.mlp(self.norm2(x)))
        return x
class QKVAttention(nn.Module):
    """
    A module which performs QKV attention and splits in a different order.
    """

    def __init__(self, dim,
                 num_heads,
                 qkv_bias=False,
                 qk_norm=False,
                 attn_drop=0.,
                 proj_drop=0.,
                 norm_layer=nn.LayerNorm):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.scale = self.head_dim ** -0.5

        self.q = nn.Linear(dim, dim, bias=qkv_bias)
        self.k = nn.Linear(dim, dim, bias=qkv_bias)
        self.v = nn.Linear(dim, dim, bias=qkv_bias)
        self.q_norm = norm_layer(self.head_dim) if qk_norm else nn.Identity()
        self.k_norm = norm_layer(self.head_dim) if qk_norm else nn.Identity()
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

        #  # down
        self.down_scale = nn.Conv2d(dim, dim, kernel_size=3, stride=2, padding=1)
        self.conv = nn.Conv2d(dim, dim, kernel_size=1, stride=1, padding=0)
        self.act = nn.ReLU(inplace=True)


    def forward(self, x):
        B, N, C = x.shape
        h, w = int(math.sqrt(N)), int(math.sqrt(N))

        _x = x.permute(0, 2, 1).reshape(B, C, h, w).contiguous()
        _x = self.conv(self.down_scale(_x))
        _x = feature2token(self.act(_x))

        q = self.q(x).reshape(B, N, self.num_heads, self.head_dim).permute(0, 2, 1, 3)
        k = self.k(_x).reshape(B, -1, self.num_heads, self.head_dim).permute(0, 2, 1, 3)
        v = self.v(_x).reshape(B, -1, self.num_heads, self.head_dim).permute(0, 2, 1, 3)
        q, k = self.q_norm(q), self.k_norm(k)

        q = q * self.scale
        attn = q @ k.transpose(-2, -1)
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)
        x = attn @ v

        x = x.transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)

        return x

class Upsample(nn.Sequential):
    """Upsample module.

    Args:
        scale (int): Scale factor. Supported scales: 2^n and 3.
        num_feat (int): Channel number of intermediate features.
    """

    def __init__(self, scale, num_feat):
        m = []
        if (scale & (scale - 1)) == 0:  # scale = 2^n
            for _ in range(int(math.log(scale, 2))):
                m.append(nn.Conv2d(num_feat, 4 * num_feat, 3, 1, 1))
                m.append(nn.PixelShuffle(2))
        elif scale == 3:
            m.append(nn.Conv2d(num_feat, 9 * num_feat, 3, 1, 1))
            m.append(nn.PixelShuffle(3))
        else:
            raise ValueError(f'scale {scale} is not supported. ' 'Supported scales: 2^n and 3.')
        super(Upsample, self).__init__(*m)

class Conv2dLayer(nn.Module):
    def __init__(self,
                 in_channels,  # Number of input channels.
                 out_channels,  # Number of output channels.
                 kernel_size,  # Width and height of the convolution kernel
                 up=1,  # Integer upsampling factor.
                 down=1,  # Integer downsampling factor.
                 ):
        super().__init__()
        self.padding = kernel_size // 2
        if up == 2:
            self.conv = Upsample(scale=2, num_feat=out_channels)
        elif down == 2:
            self.conv = nn.Conv2d(in_channels, out_channels, kernel_size,  stride=2, padding=self.padding)
        else:
            self.conv = nn.Conv2d(in_channels, out_channels, kernel_size, 1, self.padding)

    def forward(self, x):
        x = self.conv(x)
        return x

class Conv2dLayerPartial(nn.Module):
    def __init__(self,
                 in_channels,                    # Number of input channels.
                 out_channels,                   # Number of output channels.
                 kernel_size,                    # Width and height of the convolution kernel.
                 up= 1,                          # Integer upsampling factor.
                 down= 1,                        # Integer downsampling factor.
                 with_res= False,
                 ):
        super().__init__()
        self.conv = Conv2dLayer(in_channels, out_channels, kernel_size,  up, down)

        self.weight_maskUpdater = torch.ones(1, 1, kernel_size, kernel_size)
        self.slide_winsize = kernel_size ** 2
        self.stride = down
        self.padding = kernel_size // 2 if kernel_size % 2 == 1 else 0
        self.res = ResidualBlockNoBN(num_feat=out_channels, res_scale=1) if with_res else None

    def forward(self, x, mask=None):
        if mask is not None:
            with torch.no_grad():
                if self.weight_maskUpdater.type() != x.type():
                    self.weight_maskUpdater = self.weight_maskUpdater.to(x)
                update_mask = F.conv2d(mask, self.weight_maskUpdater, bias=None, stride=self.stride, padding=self.padding)
                mask_ratio = self.slide_winsize / (update_mask + 1e-8)
                update_mask = torch.clamp(update_mask, 0, 1)  # 0 or 1
                mask_ratio = torch.mul(mask_ratio, update_mask)
            x = self.conv(x)
            if self.res is not None:
                x = self.res(x)
            x = torch.mul(x, mask_ratio)
            return x, update_mask
        else:
            x = self.conv(x)
            if self.res is not None:
                x = self.res(x)
            return x, None


class ResidualBlockNoBN(nn.Module):
    """Residual block without BN.

    Args:
        num_feat (int): Channel number of intermediate features.
            Default: 64.
        res_scale (float): Residual scale. Default: 1.
        pytorch_init (bool): If set to True, use pytorch default init,
            otherwise, use default_init_weights. Default: False.
    """

    def __init__(self, num_feat=64, res_scale=1):
        super(ResidualBlockNoBN, self).__init__()
        self.res_scale = res_scale
        self.conv1 = nn.Conv2d(num_feat, num_feat, 3, 1, 1, bias=True)
        self.conv2 = nn.Conv2d(num_feat, num_feat, 3, 1, 1, bias=True)
        self.relu = nn.ReLU(inplace=True)


    def forward(self, x):
        identity = x
        out = self.conv2(self.relu(self.conv1(x)))
        return identity + out * self.res_scale

class DecoderLayers(nn.Module):
    def __init__(self, in_channels, out_channels, scale):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, 3, 1, 1, bias=True)
        self.Upsampler = Upsample(scale, out_channels)
        self.ResidualBlock = ResidualBlockNoBN(out_channels)

    def forward(self, x, skip=None):
        x = self.conv1(x)
        x = self.Upsampler(x)
        if skip is not None:
            x = x + skip
        x = self.ResidualBlock(x)
        return x

class PatchUnEmbed(nn.Module):
    def __init__(self, img_size, patch_size, embed_dim, output_channels):
        super().__init__()
        self.img_size = img_size
        self.patch_size = patch_size
        self.H = img_size // patch_size  # 计算块的行数
        self.W = img_size // patch_size  # 计算块的列数
        # 使用转置卷积将序列恢复为图像
        # self.proj = nn.ConvTranspose2d(
        #     in_channels=embed_dim,
        #     out_channels=output_channels,
        #     kernel_size=patch_size,
        #     stride=patch_size
        # )
        self.proj = DecoderLayers(embed_dim, output_channels, scale=2)


    def forward(self, x):
        B, N, D = x.shape
        # 将序列转换为 (B, C, H, W) 形式
        x = x.transpose(1, 2).view(B, D, self.H, self.W)
        # 用转置卷积恢复空间维度
        x = self.proj(x)
        return x
class Encoderlayer(nn.Module):
    def __init__(self,current_dim, next_dim, next_res, attn_res):
        super().__init__()

        self.down_conv = Conv2dLayerPartial(in_channels=current_dim,
                                                    out_channels=next_dim,
                                                    kernel_size=3, down=2,
                                                    with_res=False)

        if next_res in attn_res:
            self.attn = nn.ModuleList([
                AttentionBlock(next_dim, num_heads=8)
                for i in range(2)])
        else:
            self.attn = None
    def forward(self, x, mask):
       x, mask = self.down_conv(x, mask)
       b, c, h, w = x.shape
       if self.attn is not None:
           x = feature2token(x)
           for block in self.attn:
               x = block(x)
           x = token2feature(x, (h, w))

       return x, mask

class MutilScaleEncoder(nn.Module):
    def __init__(self, img_resolution, img_channels, res, res_channel_map, attn_res):
        super().__init__()

        self.res_channel_map = res_channel_map
        initial_dim = self.res_channel_map[img_resolution]
        self.conv_first = Conv2dLayerPartial(in_channels=img_channels + 1, out_channels=initial_dim, kernel_size=3)

        self.enc_layers = nn.ModuleList()
        down_time = int(np.log2(img_resolution // res))

        current_res = img_resolution
        current_dim = initial_dim

        # 构件编码器路径
        for i in range(down_time):
            next_res = current_res // 2
            next_dim = self.res_channel_map.get(next_res, current_dim)  # 默认保持通道不变
            self.enc_layers.append(Encoderlayer(current_dim=current_dim,
                                             next_dim=next_dim,
                                             next_res=next_res,
                                             attn_res=attn_res))
            current_res = next_res
            current_dim = next_dim

    def forward(self, x, mask):
        skips  = []
        x, mask = self.conv_first(x, mask)
        skips.append(x)
        en_outputs= []
        for i, layer in enumerate(self.enc_layers):
            x, mask = layer(x, mask)
            en_outputs.append(x)
            if i != len(self.enc_layers) - 1:
                skips.append(x)

        return x, mask, skips, en_outputs

class Decoder(nn.Module):
    def __init__(self, img_resolution, res, res_channel_map):
        super().__init__()
        current_res = res
        current_dim = res_channel_map[res]
        down_time = int(np.log2(img_resolution // res))
        self.res_channel_map = res_channel_map

        # 构建解码器路径
        self.dec_conv = nn.ModuleList()
        for i in range(down_time):
            target_res = current_res * 2
            target_dim = self.res_channel_map.get(target_res, current_dim)
            self.dec_conv.append(DecoderLayers(in_channels=current_dim, out_channels=target_dim, scale=2))

            current_res = target_res
            current_dim = target_dim

    def forward(self, x, skips):
        de_outputs = []
        for i, block in enumerate(self.dec_conv):
            x = block(x, skips[len(self.dec_conv) - i - 1])
            de_outputs.append(x)
        return x, de_outputs


@ARCH_REGISTRY.register()
class Edge(nn.Module):
    def __init__(self, img_channels=1, img_resolution=128, dim=32, res=8, attn_res=[32, 16, 8]):
        super().__init__()
        self.res_channel_map = {
            128: dim * 1,   # 128x128分辨率对应dim
            64:  dim * 2,   # 64x64 对应dim*2
            32:  dim * 4,   # 32x32 对应dim*4
            16:  dim * 8,   # 16x16 对应dim*8
            8:   dim * 8    # 8x8保持dim*8
        }
        self.mutil_scale_encoder = MutilScaleEncoder(img_resolution,
                                                     img_channels=img_channels,
                                                     res=res,
                                                     res_channel_map=self.res_channel_map,
                                                     attn_res=attn_res)
        self.decoder = Decoder(img_resolution=img_resolution,
                               res=res,
                               res_channel_map=self.res_channel_map)

        self.conv_out = nn.Conv2d(self.res_channel_map[img_resolution],1, 3, 1, 1, bias=True)

    def forward(self, images_in, masks_in):
        x, mask, skips, en_outputs = self.mutil_scale_encoder(images_in, masks_in)
        x, de_outputs = self.decoder(x,skips)
        output = self.conv_out(x)
        return output, en_outputs, de_outputs

if __name__ == '__main__':
    device = torch.device('cuda:0')
    img_resolution = 128
    batch = 1
    G = Edge(img_resolution=128, img_channels=1).to(device)
    img = torch.randn(batch, 2, img_resolution, img_resolution).to(device)
    mask = torch.randn(batch, 1, img_resolution, img_resolution).to(device)
    output, _, _ = G(img, mask)
    print(G)
    print(output.shape)



