import torch
import torch.nn as nn
# Partial Decoder Attention Network with Contour Weighted Loss Function for Data-Imbalance Medical Image Segmentation
# https://arxiv.org/pdf/2601.14338
# https://github.com/huangzyong/Contour-weighted-Loss-Seg
# 代码仅供参考，请以官方代码为准

class BasicBlock2D(nn.Module):
    def __init__(self, in_ch, out_ch, kernel_size=3, padding=1):
        super().__init__()
        self.conv1 = nn.Conv2d(in_ch, out_ch, kernel_size, padding=padding, bias=False)
        self.bn1 = nn.InstanceNorm2d(out_ch)
        self.relu = nn.ReLU(inplace=True)

        self.conv2 = nn.Conv2d(out_ch, out_ch, kernel_size, padding=padding, bias=False)
        self.bn2 = nn.InstanceNorm2d(out_ch)

        self.shortcut = nn.Identity()
        if in_ch != out_ch:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_ch, out_ch, kernel_size=1, bias=False),
                nn.InstanceNorm2d(out_ch)
            )

    def forward(self, x):
        identity = x
        out = self.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out = out + self.shortcut(identity)
        return self.relu(out)

class ChannelAttention2D(nn.Module):
    def __init__(self, channel, reduction=4):
        super().__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)

        self.fc = nn.Sequential(
            nn.Linear(channel, channel // reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(channel // reduction, channel, bias=False)
        )
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        b, c, _, _ = x.size()
        avg = self.fc(self.avg_pool(x).view(b, c))
        max_ = self.fc(self.max_pool(x).view(b, c))
        out = self.sigmoid(avg + max_).view(b, c, 1, 1)
        return out

class CWCA2D(nn.Module):
    """
    Channel-Wise Attention Module (CWCA) for 2D images
    Input : (B, C, H, W) × 2
    Output: (B, 2C, H, W)
    """

    def __init__(self, channel, reduction=4):
        super().__init__()

        self.fusion_conv = nn.Sequential(
            BasicBlock2D(2 * channel, 2 * channel),
            BasicBlock2D(2 * channel, 2 * channel)
        )

        self.weight_conv = nn.Sequential(
            nn.Conv2d(2 * channel, 2, kernel_size=3, padding=1),
            nn.Softmax(dim=1)
        )

        self.channel_attention = ChannelAttention2D(2 * channel, reduction)

    def forward(self, feat_enc, feat_dec):
        """
        feat_enc, feat_dec: (B, C, H, W)
        """
        assert feat_enc.shape == feat_dec.shape, "Shape mismatch"

        # 1. 拼接
        fusion = torch.cat([feat_enc, feat_dec], dim=1)  # (B, 2C, H, W)

        # 2. 空间权重学习（辅助）
        fusion = self.fusion_conv(fusion)
        weight_map = self.weight_conv(fusion)            # (B, 2, H, W)

        # 3. encoder / decoder 加权
        feat_enc_w = feat_enc * weight_map[:, 0:1, ...]
        feat_dec_w = feat_dec * weight_map[:, 1:2, ...]

        out = torch.cat([feat_enc_w, feat_dec_w], dim=1)

        # 4. 通道注意力（核心）
        channel_weight = self.channel_attention(out)
        out = out * channel_weight

        return out


if __name__ == "__main__":
    B, C, H, W = 2, 64, 256, 256

    feat_enc = torch.randn(B, C, H, W)
    feat_dec = torch.randn(B, C, H, W)

    cwca = CWCA2D(channel=C, reduction=4)
    out = cwca(feat_enc, feat_dec)

    print("Encoder shape :", feat_enc.shape)
    print("Decoder shape :", feat_dec.shape)
    print("CWCA output   :", out.shape)
