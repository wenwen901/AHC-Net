import inspect
import math

import torch
import torch.nn as nn
import torch.nn.init as init
import torch.nn.functional as F

from torch.nn import BatchNorm2d as _BatchNorm
from torch.nn import InstanceNorm2d as _InstanceNorm
from timm.models.layers import DropPath, to_2tuple, trunc_normal_



class Mlp(nn.Module):

    def __init__(self,
                 in_features,
                 hidden_features=None,
                 out_features=None,
                 act_layer=nn.GELU,
                 drop=0.):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Conv2d(in_features, hidden_features, 1)
        self.dwconv = DWConv(hidden_features)
        self.act = act_layer()
        self.fc2 = nn.Conv2d(hidden_features, out_features, 1)
        self.drop = nn.Dropout(drop)

    def forward(self, x):
        x = self.fc1(x)

        x = self.dwconv(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)

        return x



# ----------------------------pam_cam------------------------------


class PAM_Module(nn.Module):
    """ Position attention module"""
    #Ref from SAGAN
    def __init__(self, in_dim):
        super(PAM_Module, self).__init__()
        self.chanel_in = in_dim

        self.query_conv = nn.Conv2d(in_channels=in_dim, out_channels=in_dim//8, kernel_size=1)
        self.key_conv = nn.Conv2d(in_channels=in_dim, out_channels=in_dim//8, kernel_size=1)
        self.value_conv = nn.Conv2d(in_channels=in_dim, out_channels=in_dim, kernel_size=1)
        self.gamma = nn.Parameter(torch.zeros(1))

        self.softmax = nn.Softmax(dim=-1)
    def forward(self, x):
        m_batchsize, C, height, width = x.size()   # x:[B,C,H,W]
        proj_query = self.query_conv(x).view(m_batchsize, -1, width*height).permute(0, 2, 1)   # [B,C,H,W] -> [B,C///8 , H,W] -> [B, C//8 , H*W] -> [B , H*W ,C//8]
        proj_key = self.key_conv(x).view(m_batchsize, -1, width*height)    # [B,C,H,W] -> [B,C///8 , H,W] -> [B, C//8 , H*W]

        energy = torch.bmm(proj_query, proj_key)   # 使用 PyTorch 中的 torch.bmm() 函数执行了批量矩阵乘法（batch matrix multiplication）操作。
        attention = self.softmax(energy)
        proj_value = self.value_conv(x).view(m_batchsize, -1, width*height)

        out = torch.bmm(proj_value, attention.permute(0, 2, 1))
        out = out.view(m_batchsize, C, height, width)

        out = self.gamma * out + x
        return out


class CAM_Module(nn.Module):

    def __init__(self, in_dim):
        super(CAM_Module, self).__init__()
        self.chanel_in = in_dim
        
        self.gamma = nn.Parameter(torch.zeros(1))
        self.softmax  = nn.Softmax(dim=-1)
    def forward(self,x):
        m_batchsize, C, height, width = x.size()   #  x:[B,C,H,W]  
        proj_query = x.view(m_batchsize, C, -1)    #  x:[B,C,H,W]  -> [B,C,H*W]
        proj_key = x.view(m_batchsize, C, -1).permute(0, 2, 1)   #  [B,C,H,W]  -> [B,C,H*W] -> [B , H*W ,C ]
       
        energy = torch.bmm(proj_query, proj_key)  # [B,C,C]
        energy_new = torch.max(energy, -1, keepdim=True)[0].expand_as(energy)-energy  # 这句代码我没看懂
        attention = self.softmax(energy_new)
        proj_value = x.view(m_batchsize, C, -1)

        out = torch.bmm(attention, proj_value)
        out = out.view(m_batchsize, C, height, width)

        out = self.gamma * out + x
        return out


class basicBlock_dual(nn.Module):
    def __init__(self,
                 in_channels, out_channels):
        super(basicBlock_dual, self).__init__()

        self.body = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 1, 1, 0),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, 3, 1, 1),
        )
    def forward(self, x):
        out = self.body(x)
        out = F.relu(out)
        return out




# class DAF(nn.Module):
#     def __init__(self, in_channels):
#         super(DAF, self).__init__()
#         self.pam = PAM_Module(in_channels)
#         self.cam = CAM_Module(in_channels)
#         self.basic1 = basicBlock_dual(in_channels=in_channels, out_channels=in_channels)   # conv11 + relu + conv33 + relu
#         self.basic2 = basicBlock_dual(in_channels=in_channels, out_channels=in_channels)
#         # 定义通道交互模块
#         self.channel_interaction = nn.Sequential(
#             nn.AdaptiveAvgPool2d(1),
#             nn.Conv2d(in_channels, in_channels // 8, kernel_size=1),
#             nn.BatchNorm2d(in_channels // 8),
#             nn.GELU(),
#             nn.Conv2d(in_channels // 8, in_channels, kernel_size=1),
#             nn.Sigmoid()
#         )
#         # 定义空间交互模块
#         self.spatial_interaction = nn.Sequential(
#             nn.Conv2d(in_channels, in_channels // 16, kernel_size=1),
#             nn.BatchNorm2d(in_channels // 16),
#             nn.GELU(),
#             nn.Conv2d(in_channels // 16, 1, kernel_size=1),
#             nn.Sigmoid()
#         )
#         # 深度卷积分支
#         self.dwconv = nn.Sequential(
#             nn.Conv2d(2 * in_channels , in_channels ,1),
#             nn.Conv2d(in_channels, in_channels, kernel_size=3, stride=1, padding=1,groups=in_channels ),
#             nn.BatchNorm2d(in_channels),
#             nn.GELU()
#         )

#         self.atten_conv = nn.Sequential(nn.Conv2d(in_channels,in_channels,3,1,1),
#                                         nn.BatchNorm2d(in_channels),
#                                         nn.ReLU())

#         self.final = nn.Sequential(nn.Conv2d(in_channels=in_channels,out_channels=in_channels,kernel_size=3,padding = 1),
#                                    nn.BatchNorm2d(in_channels),
#                                    nn.ReLU())
#     def forward(self, x):  
#         short_cut = x
#         ### 注意力分支
#         # 通道注意力
#         x_cam = self.cam(x)
#         short_cut_cam = self.basic1(x_cam)
#         ci = self.channel_interaction(short_cut_cam)
        
#         # 空间注意力 
#         x_pam = self.pam(x)
#         short_cut_pam = self.basic2(x_pam)
#         si = self.spatial_interaction(short_cut_pam)

#         ### DWConv分支
#         feat = torch.cat([x_cam,x_pam],dim=1)
#         dw = self.dwconv(feat)
#         dw_ci = self.channel_interaction(dw)
#         dw_si = self.spatial_interaction(dw)

#         x_cam_m = ci * dw_si * short_cut_cam
#         x_pam_m = si * dw_ci * short_cut_pam

#         atten = self.atten_conv(x_cam_m + x_pam_m)
#         out = self.final(atten + short_cut)
#         return out

class DAF(nn.Module):
    def __init__(self, in_channels):
        super(DAF, self).__init__()
        self.pam = PAM_Module(in_channels)
        self.cam = CAM_Module(in_channels)
        self.basic1 = basicBlock_dual(in_channels=in_channels, out_channels=in_channels)   # conv11 + relu + conv33 + relu
        self.basic2 = basicBlock_dual(in_channels=in_channels, out_channels=in_channels)
        # 定义通道交互模块
        self.channel_interaction = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(in_channels, in_channels // 8, kernel_size=1),
            nn.BatchNorm2d(in_channels // 8),
            nn.GELU(),
            nn.Conv2d(in_channels // 8, in_channels, kernel_size=1),
            nn.Sigmoid()
        )
        # 定义空间交互模块
        self.spatial_interaction = nn.Sequential(
            nn.Conv2d(in_channels, in_channels // 16, kernel_size=1),
            nn.BatchNorm2d(in_channels // 16),
            nn.GELU(),
            nn.Conv2d(in_channels // 16, 1, kernel_size=1),
            nn.Sigmoid()
        )
        # 深度卷积分支
        self.dwconv = nn.Sequential(
            nn.Conv2d(2 * in_channels , in_channels ,1),
            nn.Conv2d(in_channels, in_channels, kernel_size=3, stride=1, padding=1,groups=in_channels ),
            nn.BatchNorm2d(in_channels),
            nn.GELU()
        )

        self.atten_conv = nn.Sequential(nn.Conv2d(in_channels,in_channels,3,1,1),
                                        nn.BatchNorm2d(in_channels),
                                        nn.ReLU())

        self.final = nn.Sequential(nn.Conv2d(in_channels=in_channels,out_channels=in_channels,kernel_size=3,padding = 1),
                                   nn.BatchNorm2d(in_channels),
                                   nn.ReLU())
    def forward(self, x):  
        short_cut = x
        ### 注意力分支
        # 通道注意力
        x_cam = self.cam(x)
        ci = self.basic1(x_cam)
        
        # 空间注意力 
        x_pam = self.pam(x)
        si = self.basic2(x_pam)

        ### DWConv分支
        feat = torch.cat([x_cam,x_pam],dim=1)
        dw = self.dwconv(feat)
        dw_ci = self.channel_interaction(dw)
        dw_si = self.spatial_interaction(dw)

        x_cam_m = ci * dw_si 
        x_pam_m = si * dw_ci 

        atten = self.atten_conv(x_cam_m + x_pam_m)
        out = self.final(atten + short_cut)
        return out



# --------------------------------------------MSCF------------------------------
class SELayer(nn.Module):
    def __init__(self, channel, reduction=4):
        super(SELayer, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(channel, channel // reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(channel // reduction, channel, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x):
        b, c, _, _ = x.size()
        y = self.avg_pool(x).view(b, c)
        y = self.fc(y).view(b, c, 1, 1)
        return x * y.expand_as(x)

class DWConv_P(nn.Module):
    def __init__(self, in_channels, kernel_size):
        super(DWConv_P, self).__init__()
        self.depthwise = nn.Conv2d(in_channels, in_channels, kernel_size, padding=kernel_size//2, groups=in_channels, bias=False)
        self.pointwise = nn.Conv2d(in_channels, in_channels, 1, bias=False)
        self.bn = nn.BatchNorm2d(in_channels)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        x = self.depthwise(x)
        x = self.pointwise(x)
        x = self.bn(x)
        x = self.relu(x)
        return x



# class MSDP(nn.Module):
#     def __init__(self, in_channels):
#         super(MSDP, self).__init__()
#         self.in_channels = in_channels
        
#         self.convs = nn.ModuleList([
#             DWConv_P(in_channels=in_channels, kernel_size=k) for k in (1, 3, 5, 7)
#         ])
        
#         self.attention = SELayer(in_channels * 4)
        
#         self.bottleneck = nn.Sequential(
#             nn.Conv2d(in_channels * 4, in_channels, 1, bias=False),
#             nn.BatchNorm2d(in_channels),
#             nn.ReLU(inplace=True)
#         )
        
#     def forward(self, x):
#         short_cut = x
#         size = x.size()[2:]
#         feats = []

#         pooled_1x1 = F.avg_pool2d(x, kernel_size=1, stride=1, padding=0)
#         pooled_3x3 = F.avg_pool2d(x, kernel_size=3, stride=1, padding=0)
#         pooled_5x5 = F.avg_pool2d(x, kernel_size=5, stride=1, padding=0)
#         pooled_7x7 = F.avg_pool2d(x, kernel_size=7, stride=1, padding=0)
        

#         pools = [pooled_1x1, pooled_3x3, pooled_5x5, pooled_7x7]

#         for pool, conv in zip(pools, self.convs):
#             feat = conv(pool)
#             feat = F.interpolate(feat, size, mode='bilinear', align_corners=False)
#             feats.append(feat)
        
#         out = torch.cat(feats, dim=1)
#         out = self.attention(out)
#         out = self.bottleneck(out) + short_cut

        
#         return out


class MSDP(nn.Module):
    def __init__(self, in_channels):
        super(MSDP, self).__init__()
        self.in_channels = in_channels
        
        self.convs = nn.ModuleList([
            DWConv_P(in_channels=in_channels, kernel_size=k) for k in (1, 3, 5, 7)
        ])
        
        
        self.bottleneck = nn.Sequential(
            nn.Conv2d(in_channels * 4, in_channels, 1, bias=False),
            nn.BatchNorm2d(in_channels),
            nn.ReLU(inplace=True)
        )
        
    def forward(self, x):
        short_cut = x
        size = x.size()[2:]
        feats = []
        pooled_1x1 = F.avg_pool2d(x, kernel_size=1, stride=1, padding=0)
        pooled_3x3 = F.avg_pool2d(x, kernel_size=3, stride=1, padding=0)
        pooled_5x5 = F.avg_pool2d(x, kernel_size=5, stride=1, padding=0)
        pooled_7x7 = F.avg_pool2d(x, kernel_size=7, stride=1, padding=0)

        pools = [pooled_1x1, pooled_3x3, pooled_5x5, pooled_7x7]

        for pool, conv in zip(pools, self.convs):
            feat = conv(pool)
            feat = F.interpolate(feat, size, mode='bilinear', align_corners=False)
            feats.append(feat)
        
        out = torch.cat(feats, dim=1)
        out = self.bottleneck(out)
        return out


    


class MSCFModule(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(MSCFModule, self).__init__()
        self.conv1x1_1 = nn.Conv2d(in_channels, in_channels // 4, kernel_size=1)
        self.bn1 = nn.BatchNorm2d(in_channels // 4)
        self.pyramid_pooling = MSDP(in_channels // 4)
        self.anisotropic_strip_pooling = DAF(in_channels // 4)
        self.conv1x1_2 = nn.Conv2d(in_channels // 4, in_channels, kernel_size=1)
        self.bn2 = nn.BatchNorm2d(in_channels)
        self.final_conv = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1)
        self.bn3 = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        residual = x
        x = self.relu(self.bn1(self.conv1x1_1(x)))
        pp_out = self.pyramid_pooling(x)
        asp_out = self.anisotropic_strip_pooling(pp_out)
        combined = asp_out
        x = self.relu(residual + self.bn2(self.conv1x1_2(combined)))
        x = self.relu(self.bn3(self.final_conv(x)))

        return x



class Block(nn.Module):

    def __init__(self,
                 dim,
                 mlp_ratio=4.,
                 drop=0.,
                 drop_path=0.,
                 act_layer=nn.GELU,
                 norm_cfg=dict(type='BN')):
        super().__init__()
        self.norm1 = nn.BatchNorm2d(dim)
        self.attn = MSCFModule(in_channels=dim , out_channels = dim)
        self.drop_path = DropPath(
            drop_path) if drop_path > 0. else nn.Identity()
        self.norm2 = nn.BatchNorm2d(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = Mlp(in_features=dim,
                       hidden_features=mlp_hidden_dim,
                       act_layer=act_layer,
                       drop=drop)
        layer_scale_init_value = 1e-2
        # self.layer_scale_1 = layer_scale_init_value * jt.ones((dim))
        # self.layer_scale_2 = layer_scale_init_value * jt.ones((dim))
        self.layer_scale_1 = nn.Parameter(layer_scale_init_value * torch.ones(dim))
        self.layer_scale_2 = nn.Parameter(layer_scale_init_value * torch.ones(dim))

    def forward(self, x, H, W):
        B, N, C = x.shape
        x = x.permute(0, 2, 1).view(B, C, H, W)
        x = x + self.drop_path(self.layer_scale_1.unsqueeze(-1).unsqueeze(-1) *self.attn(self.norm1(x)))
        x = x + self.drop_path(self.layer_scale_2.unsqueeze(-1).unsqueeze(-1) *self.mlp(self.norm2(x)))
        x = x.view(B, C, N).permute(0, 2, 1)
        return x




class DWConv(nn.Module):

    def __init__(self, dim=768):
        super(DWConv, self).__init__()
        self.dwconv = nn.Conv2d(dim, dim, 3, 1, 1, bias=True, groups=dim)

    def forward(self, x):
        x = self.dwconv(x)
        return x



class Feature_Incentive_Block(nn.Module):
    def __init__(self, img_size=224, patch_size=3, stride=1, in_chans=3, embed_dim=768):
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


    def forward(self, x):
        x = self.proj(x)
        _, _, H, W = x.shape
        x = x.flatten(2).transpose(1, 2)
        x = self.act(x)
        x = self.norm(x)
        return x, H, W



class MSHA(nn.Module):
    def __init__(self,
                 in_chans=3,
                #  embed_dims=[64, 128, 256, 512],
                 embed_dims=[768],
                #  mlp_ratios=[4, 4, 4, 4],
                 mlp_ratios=[4],
                 drop_rate=0.,
                 drop_path_rate=0.2,
                #  depths=[3, 4, 6, 3],
                 depths=[1],
                 num_stages=1,
                 norm_cfg=dict(type='BN')):
        super(MSHA, self).__init__()

        self.depths = depths
        self.num_stages = num_stages

        # dpr = [x.item() for x in jt.linspace(0, drop_path_rate, sum(depths))]  # stochastic depth decay rule
        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, sum(depths))]
        cur = 0

        # self.patch_embed = Feature_Incentive_Block(in_chans=in_chans,embed_dim=embed_dims[0])

        for i in range(num_stages):
            block = nn.ModuleList([
                Block(dim=in_chans,
                    #   dim=embed_dims[i],
                      mlp_ratio=mlp_ratios[i],
                      drop=drop_rate,
                      drop_path=dpr[cur + j],
                      norm_cfg=norm_cfg) for j in range(depths[i])
            ])
            norm = nn.LayerNorm(in_chans)
            cur += depths[i]

            setattr(self, f"block{i + 1}", block)
            setattr(self, f"norm{i + 1}", norm)

        self.conv1x1 = nn.Conv2d(in_channels=in_chans, out_channels=in_chans, kernel_size=1, stride=1)

        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)


    def forward(self, x,H,W):
        # B,C,H,W = x.shape
        # x, H, W = self.patch_embed(x)

        for i in range(self.num_stages):
            block = getattr(self, f"block{i + 1}")
            norm = getattr(self, f"norm{i + 1}")
            for blk in block:
                x = blk(x, H, W)
            # x = norm(x)
            # x = x.reshape(B, H, W, -1).permute(0, 3, 1, 2)

            # x = self.conv1x1(x)
        return x







