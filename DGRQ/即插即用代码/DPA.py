import torch
import torch.nn as nn
# MFmamba:AMulti-function Network for Panchromatic Image Resolution Restoration Based on State-Space Model[AAAI 2026]
# https://arxiv.org/pdf/2511.18888
# https://github.com/QianqianWang1325/MFmamba.git
# 请以官方代码为准，代码仅供参考
class DPA(nn.Module):
    """
    Dual Pool Attention (DPA)
    Paper-aligned implementation
    """
    def __init__(self, channels, reduction=16):
        super(DPA, self).__init__()

        # Dual Pool
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)

        # Shared MLP
        self.mlp_avg = nn.Sequential(
            nn.Linear(channels, channels // reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(channels // reduction, channels, bias=False),
            nn.Sigmoid()
        )

        self.mlp_max = nn.Sequential(
            nn.Linear(channels, channels // reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(channels // reduction, channels, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x):
        b, c, _, _ = x.size()

        # ===== Eq.(5): Dual Pool =====
        avg_feat = self.avg_pool(x).view(b, c)
        max_feat = self.max_pool(x).view(b, c)

        # ===== Eq.(6): Channel Excitation =====
        attn_avg = self.mlp_avg(avg_feat).view(b, c, 1, 1)
        attn_max = self.mlp_max(max_feat).view(b, c, 1, 1)

        # ===== Eq.(7): Dual Attention Fusion =====
        out = x * attn_avg + x * attn_max + x

        return out

if __name__ == '__main__':
    x = torch.randn(2, 64, 128, 128)
    dpa = DPA(64)
    y = dpa(x)
    print(y.shape)
