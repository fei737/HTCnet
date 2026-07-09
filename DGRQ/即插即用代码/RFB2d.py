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

class RFB2D(nn.Module):
    """
    Receptive Field Block (RFB) for 2D feature maps
    Input / Output: (B, C, H, W)
    """

    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.relu = nn.ReLU(inplace=True)

        # branch 0: standard conv
        self.branch0 = BasicBlock2D(in_ch, out_ch)

        # branch 1: small receptive field
        self.branch1 = nn.Sequential(
            BasicBlock2D(in_ch, out_ch, kernel_size=1, padding=0),
            BasicBlock2D(out_ch, out_ch, kernel_size=3, padding=1),
            BasicBlock2D(out_ch, out_ch, kernel_size=1, padding=0),
        )

        # branch 2: medium receptive field
        self.branch2 = nn.Sequential(
            BasicBlock2D(in_ch, out_ch, kernel_size=1, padding=0),
            BasicBlock2D(out_ch, out_ch, kernel_size=5, padding=2),
            BasicBlock2D(out_ch, out_ch, kernel_size=1, padding=0),
        )

        # branch 3: large receptive field
        self.branch3 = nn.Sequential(
            BasicBlock2D(in_ch, out_ch, kernel_size=1, padding=0),
            BasicBlock2D(out_ch, out_ch, kernel_size=7, padding=3),
            BasicBlock2D(out_ch, out_ch, kernel_size=1, padding=0),
        )

        self.conv_cat = BasicBlock2D(4 * out_ch, out_ch)
        self.conv_res = BasicBlock2D(in_ch, out_ch)

    def forward(self, x):
        x0 = self.branch0(x)
        x1 = self.branch1(x)
        x2 = self.branch2(x)
        x3 = self.branch3(x)

        x_cat = self.conv_cat(torch.cat([x0, x1, x2, x3], dim=1))
        out = self.relu(x_cat + self.conv_res(x))
        return out

if __name__ == "__main__":
    x = torch.randn(2, 64, 256, 256)
    rfb = RFB2D(64, 64)
    y = rfb(x)
    print("RFB-2D output shape:", y.shape)
