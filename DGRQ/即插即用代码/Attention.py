import torch
import torch.nn as nn

# SE（Squeeze-and-Excitation）块 https://arxiv.org/pdf/1709.01507
class SEBlock(nn.Module):
    def __init__(self, in_channel, reduction=16):
        super(SEBlock, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(in_channel, in_channel // reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(in_channel // reduction, in_channel, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x):
        b, c, _, _ = x.size()
        y = self.avg_pool(x).view(b, c)
        y = self.fc(y).view(b, c, 1, 1)
        return x * y.expand_as(x)


# CBAM（Convolutional Block Attention Module）https://openaccess.thecvf.com/content_ECCV_2018/papers/Sanghyun_Woo_Convolutional_Block_Attention_ECCV_2018_paper.pdf
class CBAMBlock(nn.Module):
    def __init__(self, in_channels, reduction=16):
        super(CBAMBlock, self).__init__()
        self.channel_attention = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(in_channels, in_channels // reduction, kernel_size=1, bias=False),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels // reduction, in_channels, kernel_size=1, bias=False),
            nn.Sigmoid()
        )
        self.spatial_attention = nn.Sequential(
            nn.Conv2d(in_channels, 1, kernel_size=7, padding=3, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x):
        channel_attention = self.channel_attention(x)
        x = x * channel_attention
        spatial_attention = self.spatial_attention(x)
        x = x * spatial_attention
        return x

# Non-Local Block, https://arxiv.org/abs/1711.07971
class NonLocalBlock(nn.Module):
    def __init__(self, in_channels):
        super(NonLocalBlock, self).__init__()
        self.theta = nn.Conv2d(in_channels, in_channels // 2, kernel_size=1)
        self.phi = nn.Conv2d(in_channels, in_channels // 2, kernel_size=1)
        self.g = nn.Conv2d(in_channels, in_channels // 2, kernel_size=1)
        self.W = nn.Conv2d(in_channels // 2, in_channels, kernel_size=1)

    def forward(self, x):
        batch_size, C, H, W = x.size()
        theta_x = self.theta(x).view(batch_size, -1, H * W)
        phi_x = self.phi(x).view(batch_size, -1, H * W)
        g_x = self.g(x).view(batch_size, -1, H * W)
        theta_phi = torch.bmm(theta_x.permute(0, 2, 1), phi_x)
        theta_phi = torch.softmax(theta_phi, dim=-1)
        out = torch.bmm(g_x, theta_phi)
        out = out.view(batch_size, C // 2, H, W)
        out = self.W(out)
        return out + x
