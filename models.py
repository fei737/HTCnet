import torch
import torch.nn as nn
import torch.nn.functional as F


# 1. 差分引导模块（DGM）
class DGM(nn.Module):
    def __init__(self, in_channels):
        super(DGM, self).__init__()
        self.conv = nn.Conv2d(in_channels, in_channels, 1, 1, 0)  # 1×1卷积
        self.sigmoid = nn.Sigmoid()

    def forward(self, f_r, f_m):
        # f_r: RGB分支特征, f_m: RGB-D分支特征（C×H×W）
        f_n = f_r - f_m  # 元素相减（公式2）
        # GAP + GMP
        gap = torch.mean(f_n, dim=[2, 3], keepdim=True)
        gmp = torch.max(f_n, dim=[2, 3], keepdim=True)[0]
        # 生成通道注意力
        att = self.sigmoid(self.conv(torch.cat([gap, gmp], dim=1)))
        # 特征加权 + 残差连接
        f_r_enhanced = f_r * att + f_r
        return f_r_enhanced


# 2. 融合感知模块（FPM）
class FPM(nn.Module):
    def __init__(self, in_channels):
        super(FPM, self).__init__()
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, f_m, f_d):
        # f_m: RGB-D分支特征, f_d: 深度分支特征（C×H×W）
        B, C, H, W = f_d.shape
        # 元素相加融合
        f_fuse = f_m + f_d
        # Reshape生成空间注意力
        f_c = f_fuse.view(B, C, -1)  # B×C×HW
        att_spatial = self.softmax(torch.matmul(f_c.transpose(1, 2), f_c))  # B×HW×HW
        # 注意力加权
        f_d_reshaped = f_d.view(B, C, -1)  # B×C×HW
        f_d_att = torch.matmul(f_d_reshaped, att_spatial)  # B×C×HW
        f_d_att = f_d_att.view(B, C, H, W)
        # 残差连接
        f_d_enhanced = f_d_att + f_d
        return f_d_enhanced


# 3. 跨模态特征细化模块（CFRM）
class CFRM(nn.Module):
    def __init__(self, in_channels):
        super(CFRM, self).__init__()
        # 空间注意力（RGB分支）
        self.spatial_att = nn.Sequential(
            nn.Conv2d(in_channels, 1, 3, 1, 1),
            nn.Sigmoid()
        )
        # 通道注意力（深度分支）
        self.channel_att = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(in_channels, in_channels, 1, 1, 0),
            nn.Sigmoid()
        )
        # 融合分支：1×1卷积 + 空洞卷积（r=2,4）
        self.conv1x1 = nn.Conv2d(in_channels, in_channels, 1, 1, 0)
        self.dilated_conv2 = nn.Conv2d(in_channels, in_channels, 3, 1, 2, dilation=2)
        self.dilated_conv4 = nn.Conv2d(in_channels, in_channels, 3, 1, 4, dilation=4)
        self.conv_fuse = nn.Conv2d(in_channels * 3, in_channels, 1, 1, 0)

    def forward(self, f_r, f_d, f_m):
        # RGB分支：空间注意力
        spatial_att = self.spatial_att(f_r)
        f_r_spatial = f_r * spatial_att
        # 深度分支：通道注意力
        channel_att = self.channel_att(f_d)
        f_d_channel = f_d * channel_att
        # 跨模态引导
        f_d_enhanced = f_d_channel + f_r_spatial
        f_r_enhanced = f_r_spatial + f_d_channel

        # 融合分支：多尺度空洞卷积
        f_m_1x1 = self.conv1x1(f_m)
        f_m_d2 = self.dilated_conv2(f_m)
        f_m_d4 = self.dilated_conv4(f_m)
        f_m_multi = self.conv_fuse(torch.cat([f_m_1x1, f_m_d2, f_m_d4], dim=1))
        # 空间注意力增强
        gap_m = torch.mean(f_m_multi, dim=[2, 3], keepdim=True)
        gmp_m = torch.max(f_m_multi, dim=[2, 3], keepdim=True)[0]
        att_m = torch.sigmoid(torch.cat([gap_m, gmp_m], dim=1))
        f_m_enhanced = f_m_multi * att_m

        return f_r_enhanced, f_d_enhanced, f_m_enhanced


# 4. 语义引导模块（SGM）
class SGM(nn.Module):
    def __init__(self, in_channels):
        super(SGM, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, in_channels, 1, 1, 0)
        self.conv2 = nn.Conv2d(in_channels, in_channels, 1, 1, 0)
        self.bn = nn.BatchNorm2d(in_channels)

    def forward(self, f_curr, f_prev):
        # f_curr: 当前层特征（i层）, f_prev: 前一层特征（i-1层）
        # 生成通道注意力
        gap = torch.mean(f_curr, dim=[2, 3], keepdim=True)
        att_channel = self.conv2(self.conv1(gap))
        att_channel = torch.sigmoid(att_channel)
        # 特征加权 + 上采样
        f_curr_att = f_curr * att_channel
        f_curr_up = F.interpolate(f_curr_att, size=f_prev.shape[2:], mode='bilinear', align_corners=False)
        # 引导前一层特征
        f_prev_enhanced = f_prev * f_curr_up
        return f_prev_enhanced