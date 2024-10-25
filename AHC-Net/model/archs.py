import torch
import torch.nn.functional as F
from model.caandsa import MSHA
from model.meda import MetaNeXtStage
from timm.models.layers import DropPath, to_2tuple, trunc_normal_
import math
import torch
from torch import nn
from einops import rearrange
from torch.nn import init
# from utils import *



def conv1x1(in_planes: int, out_planes: int, stride: int = 1) -> nn.Conv2d:
    return nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=1, bias=False)



class HGLM(nn.Module):
    def __init__(self, dim, num_heads, mlp_ratio=4., qkv_bias=False, qk_scale=None, drop=0., attn_drop=0.,
                 drop_path=0., act_layer=nn.GELU, norm_layer=nn.LayerNorm, sr_ratio=1):
        super().__init__()

        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.norm2 = norm_layer(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = MetaNeXtStage(in_chs=dim,out_chs=dim)
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)
        elif isinstance(m, nn.Conv2d):
            fan_out = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
            fan_out //= m.groups
            m.weight.data.normal_(0, math.sqrt(2.0 / fan_out))
            if m.bias is not None:
                m.bias.data.zero_()

    def forward(self, x, H, W):
        B,N,C = x.shape
        x = self.drop_path(self.mlp(x, H, W))
        out = x.flatten(2).transpose(1, 2)
        return out


class DWConv(nn.Module):
    def __init__(self, dim=768):
        super(DWConv, self).__init__()
        self.dwconv = nn.Conv2d(dim, dim, 3, 1, 1, bias=True, groups=dim)
        self.point_conv = nn.Conv2d(dim, dim, 1, 1, 0, bias=True, groups=1)

    def forward(self, x, H, W):
        x = self.dwconv(x)
        x = self.point_conv(x)
        return x


class Conv_Act(nn.Module):
    def __init__(self, img_size=224, patch_size=7, stride=4, in_chans=3, embed_dim=768):
        super().__init__()
        img_size = to_2tuple(img_size)
        patch_size = to_2tuple(patch_size)

        self.img_size = img_size
        self.patch_size = patch_size
        self.H, self.W = img_size[0] // patch_size[0], img_size[1] // patch_size[1]
        self.num_patches = self.H * self.W
        self.proj = nn.Conv2d(in_chans, embed_dim, kernel_size=patch_size, stride=stride,
                              padding=(patch_size[0] // 2, patch_size[1] // 2))
        self.norm = nn.LayerNorm(embed_dim)
        self.act = nn.GELU()
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)
        elif isinstance(m, nn.Conv2d):
            fan_out = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
            fan_out //= m.groups
            m.weight.data.normal_(0, math.sqrt(2.0 / fan_out))
            if m.bias is not None:
                m.bias.data.zero_()

    def forward(self, x):
        x = self.proj(x)
        _, _, H, W = x.shape
        x = x.flatten(2).transpose(1, 2)
        x = self.act(x)
        x = self.norm(x)
        return x, H, W


class DoubleConv(nn.Module):
    def __init__(self, in_ch, out_ch):
        super(DoubleConv, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 3, padding=1),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, 3, padding=1),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True)
        )

    def forward(self, input):
        return self.conv(input)


class D_DoubleConv(nn.Module):
    def __init__(self, in_ch, out_ch):
        super(D_DoubleConv, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_ch, in_ch, 3, padding=1),
            nn.BatchNorm2d(in_ch),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_ch, out_ch, 3, padding=1),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True)
        )

    def forward(self, input):
        return self.conv(input)
    



# 改进跳跃连接


class LayerNorm(nn.Module):
    r""" From ConvNeXt (https://arxiv.org/pdf/2201.03545.pdf)
    """
    def __init__(self, normalized_shape, eps=1e-6, data_format="channels_last"):
        """
        normalized_shape 参数表示输入张量中需要归一化的特征的形状。它可以是一个整数（当 data_format 为 "channels_last" 时），也可以是一个元组（当 data_format 为 "channels_first" 时）。
        """
        super().__init__()
        self.weight = nn.Parameter(torch.ones(normalized_shape))  # 定义了一个可学习的参数 weight，用于缩放归一化后的张量。参数的形状与 normalized_shape 一致，并初始化为全1。
        self.bias = nn.Parameter(torch.zeros(normalized_shape))   # 定义了一个可学习的参数 bias，用于在归一化后的张量上进行平移。参数的形状与 normalized_shape 一致，并初始化为全0。
        self.eps = eps
        self.data_format = data_format
        if self.data_format not in ["channels_last", "channels_first"]:
            raise NotImplementedError 
        self.normalized_shape = (normalized_shape, )  # 保存了归一化的特征形状，以元组的形式存储。
    
    def forward(self, x):
        if self.data_format == "channels_last":
            # 调用 PyTorch 的 F.layer_norm 函数对输入张量 x 进行归一化，使用的参数为 weight、bias 和 eps。该函数会根据归一化的特征形状和参数对输入张量进行归一化操作。
            return F.layer_norm(x, self.normalized_shape, self.weight, self.bias, self.eps)
        elif self.data_format == "channels_first":
            u = x.mean(1, keepdim=True)   # 表示沿着x的第二个维度进行均值计算
            s = (x - u).pow(2).mean(1, keepdim=True)  # 计算输入张量 x 沿着通道维度的方差。
            x = (x - u) / torch.sqrt(s + self.eps)  # 输入张量 x 进行归一化，使用均值和方差对其进行标准化，并添加 eps 以避免除零错误。
            """
            self.weight[:, None, None] : 切片操作，表示对weight的第一个维度全选，另外在增加一个维度，使得x的维度有[normalized_shape] -> [normalized_shape,1,1]
            """
            x = self.weight[:, None, None] * x + self.bias[:, None, None]
            return x
    



class DWconvBlock(nn.Module):
    def __init__(self, in_channel , kernel_size=7, drop_path=0., layer_scale_init_value=1e-6):
        super().__init__()
        self.dwconv = nn.Conv2d(in_channel, in_channel, kernel_size=kernel_size, padding=kernel_size//2, groups=in_channel)  # depthwise conv   # kernel_size = 7 
        self.norm = LayerNorm(in_channel, eps=1e-6)
        self.pwconv1 = nn.Linear(in_channel, 4 * in_channel)  # pointwise/1x1 convs, implemented with linear layers
        self.act = nn.GELU()
        self.pwconv2 = nn.Linear(4 * in_channel, in_channel)
        self.gamma = nn.Parameter(layer_scale_init_value * torch.ones((in_channel)),requires_grad=True) if layer_scale_init_value > 0 else None
        self.drop_path = nn.DropPath(drop_path) if drop_path > 0. else nn.Identity()

    def forward(self, x):
        input = x
        # print("x.shape  = " , x.shape)
        # DWConv
        x = self.dwconv(x)
        # print("self.dwconv(x).shape = " , x.shape)
        x = x.permute(0, 2, 3, 1)  # (N, C, H, W) -> (N, H, W, C)
        # print("x.permute(0, 2, 3, 1).shape = " , x.shape)
        # MLP
        x = self.norm(x)
        x = self.pwconv1(x)
        x = self.act(x)
        x = self.pwconv2(x)
        # print("mlp_x.shape = " , x.shape)
        # exit()
        if self.gamma is not None:
            x = self.gamma * x
        x = x.permute(0, 3, 1, 2)  # (N, H, W, C) -> (N, C, H, W)
        x = input + self.drop_path(x)
        return x


# def elu_feature_map(x):
#     return torch.nn.functional.elu(x) + 1


# class LinearAttention(nn.Module):
#     def __init__(self, eps=1e-6):
#         super().__init__()
#         self.feature_map = elu_feature_map
#         self.eps = eps

#     def forward(self, queries, keys, values):
#         queries = queries.permute(0, 3, 1, 2)
#         keys = keys.permute(0, 3, 1, 2)
#         values = values.permute(0, 3, 1, 2)

#         Q = self.feature_map(queries)
#         K = self.feature_map(keys)

#         v_length = values.size(1)
#         values = values / v_length  # prevent fp16 overflow
#         KV = torch.einsum("nshd,nshv->nhdv", K, values)  # (S,D)' @ S,V
#         Z = 1 / (torch.einsum("nlhd,nhd->nlh", Q, K.sum(dim=1)) + self.eps)
#         queried_values = torch.einsum("nlhd,nhdv,nlh->nlhv", Q, KV, Z) * v_length

#         return queried_values.contiguous()



# class AttentionLayer(nn.Module):
#     def __init__(self, hidden_dim , attention_type='linear'):
#         super().__init__()
#         self.channel = hidden_dim
#         self.q = nn.Linear( self.channel ,  self.channel)
#         self.k = nn.Linear( self.channel ,  self.channel)
#         self.v = nn.Linear( self.channel ,  self.channel)

#         if attention_type == 'linear':
#             self.attention = LinearAttention()
#         else:
#             raise NotImplementedError
#     def forward(self, x):
#         q = self.q(x)
#         k = self.k(x)
#         v = self.v(x)
#         out = self.attention(q, k, v)
#         return out




def elu_feature_map(x):
    return F.elu(x) + 1

class LinearAttention(nn.Module):
    def __init__(self, eps=1e-6):
        super().__init__()
        self.feature_map = elu_feature_map
        self.eps = eps

    def forward(self, queries, keys, values):
        Q = self.feature_map(queries)
        K = self.feature_map(keys)

        v_length = values.size(-2)
        values = values / v_length  # prevent fp16 overflow
        KV = torch.einsum("bhld,bhlv->bhdv", K, values)
        Z = 1 / (torch.einsum("bhld,bhd->bhl", Q, K.sum(dim=-2)) + self.eps)
        queried_values = torch.einsum("bhld,bhdv,bhl->bhlv", Q, KV, Z) * v_length

        return queried_values.contiguous()

class AttentionLayer(nn.Module):
    def __init__(self, in_channels, num_heads=8, attention_type='linear'):
        super().__init__()
        self.in_channels = in_channels
        self.num_heads = num_heads
        self.head_dim = in_channels // num_heads

        self.q = nn.Conv2d(in_channels, in_channels, 1)
        self.k = nn.Conv2d(in_channels, in_channels, 1)
        self.v = nn.Conv2d(in_channels, in_channels, 1)
        self.out = nn.Conv2d(in_channels, in_channels, 1)

        if attention_type == 'linear':
            self.attention = LinearAttention()
        else:
            raise NotImplementedError

    def forward(self, x):
        b, c, h, w = x.size()
        q = self.q(x).view(b, self.num_heads, self.head_dim, h * w).permute(0, 1, 3, 2)
        k = self.k(x).view(b, self.num_heads, self.head_dim, h * w).permute(0, 1, 3, 2)
        v = self.v(x).view(b, self.num_heads, self.head_dim, h * w).permute(0, 1, 3, 2)
        out = self.attention(q, k, v)
        out = out.permute(0, 1, 3, 2).contiguous().view(b, c, h, w)
        return self.out(out)



# class AttentionBlock(nn.Module):
#     def __init__(self, in_channel, out_channel ,  attention_type='linear', pooling_size=(4, 4)) -> None:
#         super().__init__()
#         self.pool = nn.AvgPool2d(pooling_size)
#         self.attention = AttentionLayer(in_channel , attention_type=attention_type)
#         # self.attention = ExternalAttention(in_channel)
#         self.MLP = nn.Sequential(
#             # nn.Linear(in_channel, in_channel * 4),
#             # nn.ReLU(),
#             # nn.Linear(in_channel * 4, in_channel)

#             nn.Linear(in_channel, 4 * in_channel),  # pointwise/1x1 convs, implemented with linear layers
#             nn.ReLU(),
#             nn.Linear(4 * in_channel, in_channel)
#         )

#         self.norm1 = nn.LayerNorm(in_channel)
#         self.norm2 = nn.LayerNorm(in_channel)        
#         # self.norm1 = LayerNorm(in_channel, eps=1e-6, data_format='channels_first')
#         # self.norm2 = LayerNorm(in_channel, eps=1e-6, data_format='channels_first')

#         # self.norm1 = LayerNorm(in_channel, eps=1e-6, data_format='channels_first')
#         # self.norm2 = LayerNorm(in_channel, eps=1e-6, data_format='channels_first')
#         # self.norm2 = LayerNorm(in_channel, eps=1e-6)

#         self.ldw = nn.Sequential(
#                 nn.Conv2d(in_channel, in_channel, kernel_size=3, padding=1, groups=in_channel),
#                 nn.ReLU(),
#                 nn.Conv2d(in_channel, out_channel, 1),
#         )
    
#     def pool_features(self, x):
#         """
#         Intermediate pooling layer for computational efficiency.
#         Arguments:
#             x: B, C, T, H, W
#         """
#         B = x.size(0)
#         x = rearrange(x, 'B C T H W -> (B T) C H W')
#         x = self.pool(x)
#         x = rearrange(x, '(B T) C H W -> B C T H W', B=B)
#         return x

#     def forward(self, x):
#         """
#         Arguments:
#             x: B, C, T, H, W
#             guidance: B, T, C
#         """
#         B, _, H, W = x.size()
#         x_pool = self.pool_features(x)
#         # x_pool = self.pool(x)
#         *_, H_pool, W_pool = x_pool.size()
#         x_pool_attention  = x_pool

#         # if guidance is not None:
#         #     guidance = repeat(guidance, 'B T C -> (B H W) T C', H=H_pool, W=W_pool)

#         x_pool = x_pool.permute(0, 2, 3, 1)
#         x_pool = x_pool_attention + self.attention(self.norm1(x_pool)).permute(0, 3, 1, 2) # Attention

#         x_pool_res = x_pool

#         x_pool = x_pool.permute(0, 2, 3, 1)
#         x_pool = x_pool_res + self.MLP(self.norm2(x_pool)).permute(0, 3, 1, 2) # MLP

#         x_pool = F.interpolate(x_pool, size=(H, W), mode='bilinear', align_corners=True)

#         x = self.ldw(x + x_pool) # Residual

#         return x




class AttentionBlock(nn.Module):
    def __init__(self, in_channel, out_channel ,  attention_type='linear', pooling_size=(4, 4)) -> None:
        super().__init__()
        self.pool = nn.AvgPool2d(pooling_size)
        self.attention = AttentionLayer(in_channel , attention_type=attention_type)
        self.MLP = nn.Sequential(
            nn.Linear(in_channel, 4 * in_channel),  # pointwise/1x1 convs, implemented with linear layers
            nn.ReLU(),
            nn.Linear(4 * in_channel, in_channel)
        )

        self.norm1 = nn.LayerNorm(in_channel)
        self.norm2 = nn.LayerNorm(in_channel)        

        self.ldw = nn.Sequential(
                nn.Conv2d(in_channel, in_channel, kernel_size=3, padding=1, groups=in_channel),
                nn.ReLU(),
                nn.Conv2d(in_channel, out_channel, 1),
        )
    
    def pool_features(self, x):
        """
        Intermediate pooling layer for computational efficiency.
        Arguments:
            x: B, C, T, H, W
        """
        B = x.size(0)
        x = rearrange(x, 'B C T H W -> (B T) C H W')
        x = self.pool(x)
        x = rearrange(x, '(B T) C H W -> B C T H W', B=B)
        return x

    def forward(self, x):
        """
        Arguments:
            x: B, C, T, H, W
            guidance: B, T, C
        """
        B, _, H, W = x.size()
        x_pool = self.pool(x)
        *_, H_pool, W_pool = x_pool.size()
        x_pool_attention  = x_pool

        x_pool = x_pool.permute(0, 2, 3, 1)
        x_pool = x_pool_attention + self.attention(self.norm1(x_pool).permute(0, 3, 1, 2)) # Attention

        x_pool_res = x_pool

        x_pool = x_pool.permute(0, 2, 3, 1)
        x_pool = x_pool_res + self.MLP(self.norm2(x_pool)).permute(0, 3, 1, 2) # MLP

        x_pool = F.interpolate(x_pool, size=(H, W), mode='bilinear', align_corners=True)

        # print("x.shape = " , x.shape)
        # print("x_pool.shape = " ,x_pool.shape)
        # exit()
        x = self.ldw(x + x_pool) # Residual

        return x



# class ExternalAttention(nn.Module):

#     def __init__(self, d_model,S=64):
#         super().__init__()
#         self.mk=nn.Linear(d_model,S,bias=False)
#         self.mv=nn.Linear(S,d_model,bias=False)
#         self.softmax=nn.Softmax(dim=1)
#         self.init_weights()


#     def init_weights(self):
#         for m in self.modules():
#             if isinstance(m, nn.Conv2d):
#                 init.kaiming_normal_(m.weight, mode='fan_out')
#                 if m.bias is not None:
#                     init.constant_(m.bias, 0)
#             elif isinstance(m, nn.BatchNorm2d):
#                 init.constant_(m.weight, 1)
#                 init.constant_(m.bias, 0)
#             elif isinstance(m, nn.Linear):
#                 init.normal_(m.weight, std=0.001)
#                 if m.bias is not None:
#                     init.constant_(m.bias, 0)

#     def forward(self, queries):
#         attn=self.mk(queries) #bs,n,S
#         attn=self.softmax(attn) #bs,n,S
#         attn=attn/torch.sum(attn,dim=2,keepdim=True) #bs,n,S
#         out=self.mv(attn) #bs,n,d_model

#         return out


class Conv2d_batchnorm(torch.nn.Module):

    def __init__(
        self,
        num_in_filters,
        num_out_filters,
        kernel_size,
        stride=(1, 1),
        activation="LeakyReLU",
        # activation="PReLU",
    ):

        super().__init__()
        self.activation = nn.ReLU()
        # self.activation = torch.nn.PReLU()
        self.conv1 = torch.nn.Conv2d(
            in_channels=num_in_filters,
            out_channels=num_out_filters,
            kernel_size=kernel_size,
            stride=stride,
            # padding="same",   # padding: 填充的方式，"same" 表示使用足够的填充使得输出特征图的大小与输入特征图的大小相同。
        )
        self.batchnorm = torch.nn.BatchNorm2d(num_out_filters)
        # self.sqe = ChannelSELayer(num_out_filters)


    def forward(self, x):
        x = self.conv1(x)
        x = self.batchnorm(x)

        # return self.sqe(self.activation(x))
        return self.activation(x)
    


class DepthWiseConv2d(nn.Module):
    def __init__(self, dim_in, dim_out, kernel_size=3, padding=1, stride=1, dilation=1):
        super().__init__()
        
        self.conv1 = nn.Conv2d(dim_in, dim_in, kernel_size=kernel_size, padding=padding, 
                      stride=stride, dilation=dilation, groups=dim_in)
        self.norm_layer = nn.GroupNorm(4, dim_in)
        self.conv2 = nn.Conv2d(dim_in, dim_out, kernel_size=1)

    def forward(self, x):
        return self.conv2(self.norm_layer(self.conv1(x)))



class Fuse_up_skip(nn.Module):
    def __init__(self,in_channel):
        super().__init__()
        self.in_channel = in_channel
        self.sum_channel = (in_channel + in_channel)
        # self.cnv_blk = Conv2d_batchnorm(self.sum_channel , self.in_channel , (1,1))
        self.cnv_blk = DepthWiseConv2d(self.sum_channel , self.in_channel)
        self.bns_blk= nn.BatchNorm2d(in_channel)

        self.act = nn.ReLU()

        # self.fc = nn.Linear(self.in_channel, self.in_channel)
        self.fc = nn.Conv2d(self.in_channel, self.in_channel, 1, stride=1, padding=0)
        

    def forward(self,up,skip_x):
        B,C,H,W = skip_x.shape
        up_c = up
        skip_x_c = skip_x
        fuse = self.act(self.bns_blk(self.cnv_blk(torch.cat((up,skip_x),dim=1))))

        up_pool = F.avg_pool2d( up, (up.size(2), up.size(3)), stride=(up.size(2), up.size(3)))  
        skip_x_pool = F.avg_pool2d( skip_x, (skip_x.size(2), skip_x.size(3)), stride=(skip_x.size(2), skip_x.size(3)))  
        up_pool = F.softmax(self.act(self.fc(up_pool)),dim=1)
        skip_x_pool = F.softmax(self.act(self.fc(skip_x_pool)),dim=1)
        # cross = F.softmax(torch.matmul(up_pool, skip_x_pool),dim=1)
        # cross_fuse = torch.matmul(cross, fuse)
        up_out = ( up_c * up_pool ) + up_c
        skip_out = (skip_x_c * skip_x_pool) + skip_x_c
        # cross_fuse = cross * fuse
        output = torch.cat([fuse , up_out , skip_out],dim=1)
        return output



class ConvBatchNorm(nn.Module):
    """(convolution => [BN] => ReLU)"""

    def __init__(self, in_channels, out_channels, activation='ReLU',kernel_size = 3):
        super(ConvBatchNorm, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels,
                              kernel_size=kernel_size, padding=(kernel_size - 1) // 2)
        self.norm = nn.BatchNorm2d(out_channels)
        self.activation = nn.ReLU()

    def forward(self, x):
        out = self.conv(x)
        out = self.norm(out)
        return self.activation(out)



class SENet2D(nn.Module):
    def __init__(self, in_channel, ratio_rate=16):
        super(SENet2D, self).__init__()

        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
        nn.Linear(in_channel, in_channel // ratio_rate, False),
            nn.ReLU(),
            nn.Linear(in_channel // ratio_rate, in_channel, False),
            nn.Sigmoid(),
        )

    def forward(self, x):
        b, c, h, w = x.size()

        # b, c, h, w, d -> b, c, 1, 1, 1
        avg = self.avg_pool(x).view([b, c])

        # b, c -> b, c // ratio_rate -> b, c -> b, c, 1, 1
        fc = self.fc(avg).view([b, c, 1, 1])
        return x * fc




class AFFM(nn.Module):
    """Upscaling then double conv"""
    def __init__(self, embed_dims=[16, 32, 64, 128, 256], attention_type='linear', pooling_size=(4, 4), cnext_type="V1", kernel_size=7 , mid_channel = 64):
        super().__init__()
        self.mid_channel = mid_channel
        self.in_place = ConvBatchNorm(embed_dims[3] , self.mid_channel)
        self.in_down02 = nn.Conv2d(embed_dims[2], self.mid_channel, kernel_size=3, stride=2, padding=1)
        self.in_down04 = nn.Conv2d(embed_dims[1], self.mid_channel, kernel_size=3, stride=4, padding=1)
        self.in_down08 = nn.Conv2d(embed_dims[0], self.mid_channel, kernel_size=3, stride=8, padding=1)

        self.ca = SENet2D(in_channel=self.mid_channel * 4)

        self.in_up2 = nn.ConvTranspose2d(self.mid_channel * 4, embed_dims[2], kernel_size=2, stride=2, padding=0)
        self.in_up4 = nn.ConvTranspose2d(self.mid_channel * 4, embed_dims[1], kernel_size=4, stride=4, padding=0)
        self.in_up8 = nn.ConvTranspose2d(self.mid_channel * 4, embed_dims[0], kernel_size=8, stride=8, padding=0)


        if cnext_type=="V1":
            self.cnext = DWconvBlock(self.mid_channel * 4 , kernel_size, 0.0, 1.0)
        else:
            raise NotImplementedError
        self.attention = AttentionBlock(self.mid_channel * 4 , self.mid_channel * 4 , attention_type=attention_type, pooling_size=pooling_size)
        # self.attention = ExternalAttention(in_channels)

        self.conv = ConvBatchNorm(self.mid_channel * 8 , self.mid_channel * 4)

        self.cnv_blk1 = Conv2d_batchnorm(self.mid_channel * 4 , embed_dims[3], (1,1))
        # self.cnv_blk2 = Conv2d_batchnorm(embed_dims[3] * 4 , embed_dims[2], (1,1))

    def forward(self, t1,t2,t3,t4):
        t1_1 = self.in_down08(t1)
        t2_2 = self.in_down04(t2)
        t3_3 = self.in_down02(t3)
        t4_4 = self.in_place(t4)

        t_fuse = self.ca(torch.cat([t1_1, t2_2, t3_3, t4_4], dim=1))

        feat1 = self.cnext(t_fuse)
        feat2 = self.attention(t_fuse)


        feat = self.conv(torch.cat([feat1,feat2],dim=1))

        t4_out = self.cnv_blk1(feat)
        t3_out = self.in_up2(feat)
        t2_out = self.in_up4(feat)
        t1_out = self.in_up8(feat)

        return t1_out , t2_out , t3_out , t4_out




class DSCA(nn.Module):
    def __init__(self, in_channels):  # in_channels = 64 ， out_channels = 48
        super().__init__()
        self.fuse = Fuse_up_skip(in_channels)
        self.nConvs = ConvBatchNorm(in_channels * 3,in_channels)


    def forward(self, x, skip_x):

        fuse = self.fuse(x,skip_x)
        return self.nConvs(fuse)
    



class AHC_Net(nn.Module):
    def __init__(self, num_classes, input_channels=3, deep_supervision=False, img_size=224,
                 embed_dims=[16, 32, 64, 128, 256],
                 num_heads=[1, 2, 4, 8], qkv_bias=False, qk_scale=None, drop_rate=0., attn_drop_rate=0.,
                 drop_path_rate=0., norm_layer=nn.LayerNorm, depths=[1, 1, 1], sr_ratios=[8, 4, 2, 1], **kwargs):
        super().__init__()

        self.conv1 = DoubleConv(input_channels, embed_dims[0])
        self.pool1 = nn.MaxPool2d(2)
        self.conv2 = DoubleConv(embed_dims[0], embed_dims[1])
        self.pool2 = nn.MaxPool2d(2)
        self.conv3 = DoubleConv(embed_dims[1], embed_dims[2])
        self.pool3 = nn.MaxPool2d(2)

        self.pool4 = nn.MaxPool2d(2)

        self.Conv_Act1 = Conv_Act(img_size=img_size // 4, patch_size=3, stride=1,
                                                    in_chans=embed_dims[2],
                                                    embed_dim=embed_dims[3])
        # self.Conv_Act2 = Conv_Act(img_size=img_size // 8, patch_size=3, stride=1,
        #                                             in_chans=embed_dims[3],
        #                                             embed_dim=embed_dims[4])

        self.Conv_Act2 = Conv_Act(img_size=img_size // 8, patch_size=3, stride=1,
                                                    in_chans=embed_dims[4],
                                                    embed_dim=embed_dims[4])

        self.Conv_Act3 = Conv_Act(img_size=img_size // 8, patch_size=3, stride=1,
                                                    in_chans=embed_dims[4],
                                                    embed_dim=embed_dims[3])

        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, sum(depths))]
        self.block1 = nn.ModuleList([HGLM(
            dim=embed_dims[3], num_heads=num_heads[0], mlp_ratio=1, qkv_bias=qkv_bias, qk_scale=qk_scale,
            drop=drop_rate, attn_drop=attn_drop_rate, drop_path=dpr[0], norm_layer=norm_layer,
            sr_ratio=sr_ratios[0])])
        self.block2 = nn.ModuleList([HGLM(
            dim=embed_dims[4], num_heads=num_heads[0], mlp_ratio=1, qkv_bias=qkv_bias, qk_scale=qk_scale,
            drop=drop_rate + 0.1, attn_drop=attn_drop_rate, drop_path=dpr[1], norm_layer=norm_layer,
            sr_ratio=sr_ratios[0])])
        self.block3 = nn.ModuleList([HGLM(
            dim=embed_dims[3], num_heads=num_heads[0], mlp_ratio=1, qkv_bias=qkv_bias, qk_scale=qk_scale,
            drop=drop_rate, attn_drop=attn_drop_rate, drop_path=dpr[0], norm_layer=norm_layer,
            sr_ratio=sr_ratios[0])])

        self.norm1 = norm_layer(embed_dims[3])
        self.norm2 = norm_layer(embed_dims[4])
        self.norm3 = norm_layer(embed_dims[3])

        self.Conv_Act4 = nn.Conv2d(embed_dims[3], embed_dims[2], 3, stride=1, padding=1)
        self.dbn4 = nn.BatchNorm2d(embed_dims[2])

        self.decoder3 = D_DoubleConv(embed_dims[2], embed_dims[1])
        self.decoder2 = D_DoubleConv(embed_dims[1], embed_dims[0])
        self.decoder1 = D_DoubleConv(embed_dims[0], 8)

        ### 跳跃连接改造 embed_dims=[8, 16, 32, 64 , 128 , 256],
        self.fuse = AFFM(embed_dims=embed_dims)
        self.up1 = DSCA(embed_dims[3])
        self.up2 = DSCA(embed_dims[2])
        self.up3 = DSCA(embed_dims[1])
        self.up4 = DSCA(embed_dims[0])


        ### bottlenck改造
        self.down8Pooling = nn.Conv2d(embed_dims[0], embed_dims[4], kernel_size=8, stride=8, padding=0)   # 将输入特征图的尺寸缩小到原来的1/8。
        self.down4Pooling = nn.Conv2d(embed_dims[1], embed_dims[4], kernel_size=4, stride=4, padding=0)   # 将输入特征图的尺寸缩小到原来的1/4。
        self.down2Pooling = nn.Conv2d(embed_dims[2], embed_dims[4], kernel_size=2, stride=2, padding=0) 
        self.down1Pooling = nn.Conv2d(embed_dims[3], embed_dims[4], kernel_size=1)   # 将输入特征图的尺寸缩小到原来的1/4。

        self.ca = ConvBatchNorm( embed_dims[4] * 4 , embed_dims[4] , kernel_size=1)
        self.Trans_fuse = MSHA(in_chans=embed_dims[4])

        self.final = nn.Conv2d(8, num_classes, kernel_size=1)

    def forward(self, x):
        B = x.shape[0]
        ### Encoder
        # conv1
        out = self.conv1(x)
        t1 = out
        out = self.pool1(out)
        # conv2
        out = self.conv2(out)
        t2 = out
        out = self.pool2(out)
        # conv3
        out = self.conv3(out)
        t3 = out
        out = self.pool3(out)
        # GL_Block
        out, H, W = self.Conv_Act1(out)
        for i, blk in enumerate(self.block1):
            out = blk(out, H, W)
        out = self.norm1(out)
        out = out.reshape(B, H, W, -1).permute(0, 3, 1, 2).contiguous()
        t4 = out
        out = self.pool4(out)

        t1_1 = self.down8Pooling(t1)
        t2_2 = self.down4Pooling(t2)
        t3_3 = self.down2Pooling(t3)
        t4_4 = self.down1Pooling(t4)
        t_fuse = self.pool4(self.ca(torch.cat([ t1_1 , t2_2 , t3_3 , t4_4],dim=1)))

        # print("t1.shape   = " , t1.shape)
        # print("t2.shape   = " , t2.shape)
        # print("t3.shape   = " , t3.shape)
        # print("t4.shape   = " , t4.shape)

        # print("t2_2.shape   = " , t2_2.shape)
        # print("t3_3.shape   = " , t3_3.shape)
        # print("t4_4.shape   = " , t4_4.shape)

        ### Bottleneck
        out, H, W = self.Conv_Act2(t_fuse)
        out = self.Trans_fuse(out,H,W)
        out = self.norm2(out)
        out = out.reshape(B, H, W, -1).permute(0, 3, 1, 2).contiguous()
        out, H, W = self.Conv_Act3(out)
        out = out.reshape(B, H, W, -1).permute(0, 3, 1, 2).contiguous()
        out = F.interpolate(out, scale_factor=(2, 2), mode='bilinear')
        # print("out.shape = " , out.shape)


        t1_out , t2_out , t3_out , t4_out = self.fuse(t1,t2,t3,t4)

        # print("t1_out.shape   = " , t1_out.shape)
        # print("t2_out.shape   = " , t2_out.shape)
        # print("t3_out.shape   = " , t3_out.shape)
        # print("t4_out.shape   = " , t4_out.shape)
        # exit()

        ### Decoder
        # GL_Block
        # out = torch.add(out, t4)
        out = self.up1(out, t4_out)
        out = out.flatten(2).transpose(1, 2)
        for i, blk in enumerate(self.block3):
            out = blk(out, H * 2, W * 2)
        out = self.norm3(out)
        out = out.reshape(B, H * 2, W * 2, -1).permute(0, 3, 1, 2).contiguous()
        out = F.interpolate(F.relu(self.dbn4(self.Conv_Act4(out))), scale_factor=(2, 2), mode='bilinear')
        #conv1
        # out = torch.add(out, t3)
        out = self.up2(out, t3_out)
        out = F.interpolate(self.decoder3(out), scale_factor=(2, 2), mode='bilinear')
        # conv2
        # out = torch.add(out, t2)
        out = self.up3(out, t2_out)
        out = F.interpolate(self.decoder2(out), scale_factor=(2, 2), mode='bilinear')
        # conv3
        # out = torch.add(out, t1)
        out = self.up4(out, t1_out)
        out = self.decoder1(out)

        out = self.final(out)

        return out
