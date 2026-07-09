import torch
import torch.nn as nn
# MFmamba:AMulti-function Network for Panchromatic Image Resolution Restoration Based on State-Space Model[AAAI 2026]
# https://arxiv.org/pdf/2511.18888
# https://github.com/QianqianWang1325/MFmamba.git
# 请以官方代码为准，代码仅供参考
def default_conv(ch_in, ch_out, kernel_size, bias=True):
    return nn.Conv2d(
        ch_in, ch_out, kernel_size,
        padding=(kernel_size//2), bias=bias)

class MHCB(nn.Module):
    def __init__(self, channels, bias=True, activation=nn.ReLU(inplace=True)):
        super(MHCB, self).__init__()

        # === 第一层多尺度卷积（Eq.1）===
        self.conv3_1 = default_conv(channels, channels, kernel_size=3, bias=bias)
        self.conv5_1 = default_conv(channels, channels, kernel_size=5, bias=bias)

        # === 中间 1×1 融合（Eq.2）===
        self.fuse_3 = nn.Conv2d(channels * 3, channels, kernel_size=1, bias=bias)
        self.fuse_5 = nn.Conv2d(channels * 3, channels, kernel_size=1, bias=bias)

        # === 第二层跨尺度卷积（Eq.3）===
        self.conv3_2 = default_conv(channels, channels, kernel_size=3, bias=bias)
        self.conv5_2 = default_conv(channels, channels, kernel_size=5, bias=bias)

        # === 最终 bottleneck 融合（Eq.4）===
        self.fuse_out = nn.Conv2d(channels * 5, channels, kernel_size=1, bias=bias)

        self.activation = activation

    def forward(self, x):
        # -------- Eq.(1): 第一层多尺度残差 --------
        x3_1 = self.activation(self.conv3_1(x)) + x
        x5_1 = self.activation(self.conv5_1(x)) + x

        # -------- Eq.(2): 特征拼接 + 1×1 融合 --------
        x_cat1 = torch.cat([x, x3_1, x5_1], dim=1)
        x3 = self.fuse_3(x_cat1)
        x5 = self.fuse_5(x_cat1)

        # -------- Eq.(3): 第二层跨尺度卷积 --------
        x3_2 = self.activation(self.conv3_2(x3))
        x5_2 = self.activation(self.conv5_2(x5))

        # -------- Eq.(4): Dense + Cross 融合 --------
        x_cat2 = torch.cat([x, x3_1, x5_1, x3_2, x5_2], dim=1)
        out = self.fuse_out(x_cat2)

        return out


if __name__ == '__main__':
    x = torch.randn(2, 64, 128, 128)
    mhcb = MHCB(64)
    y = mhcb(x)
    print(y.shape)
