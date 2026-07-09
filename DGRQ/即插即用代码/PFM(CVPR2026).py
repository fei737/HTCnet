import torch
import torch.nn as nn
# It Takes Two: A Duet of Periodicity and Directionality for Burst Flicker Removal, CVPR 2026.
# https://arxiv.org/pdf/2603.22794
# https://github.com/qulishen/Flickerformer
# 本代码仅供参考，请以官方代码为准
class PFM(nn.Module):
    """
    Phase-aware Fusion Module (Plug-and-Play)
    输入: (B, 3C, H, W)
    输出: (B, C, H, W)
    """

    def __init__(self, dim, expansion=2.66, bias=False):
        super().__init__()

        hidden_dim = int(dim * expansion)

        self.net12 = nn.Sequential(
            nn.Conv2d(dim, hidden_dim, 3, padding=1, bias=bias),
            nn.ReLU(inplace=True),
            nn.Conv2d(hidden_dim, hidden_dim, 3, padding=1, bias=bias),
            nn.ReLU(inplace=True),
            nn.Conv2d(hidden_dim, dim, 1, bias=bias),
            nn.Sigmoid()
        )

        self.net23 = nn.Sequential(
            nn.Conv2d(dim, hidden_dim, 3, padding=1, bias=bias),
            nn.ReLU(inplace=True),
            nn.Conv2d(hidden_dim, hidden_dim, 3, padding=1, bias=bias),
            nn.ReLU(inplace=True),
            nn.Conv2d(hidden_dim, dim, 1, bias=bias),
            nn.Sigmoid()
        )

        self.fusion = nn.Conv2d(dim * 3, dim, 3, padding=1)

        self.eps = 1e-8

    def forward(self, x):
        x1, x2, x3 = x.chunk(3, dim=1)

        # FFT
        f1 = torch.fft.rfft2(x1)
        f2 = torch.fft.rfft2(x2)
        f3 = torch.fft.rfft2(x3)

        # phase
        phase1 = f1 / (torch.abs(f1) + self.eps)
        phase2 = f2 / (torch.abs(f2) + self.eps)
        phase3 = f3 / (torch.abs(f3) + self.eps)

        # correlation
        C12 = torch.abs(phase1 * torch.conj(phase2))
        C23 = torch.abs(phase3 * torch.conj(phase2))

        # learnable weighting
        C12 = self.net12(C12)
        C23 = self.net23(C23)

        # filtering
        f1 = C12 * f1
        f3 = C23 * f3

        # iFFT
        x1 = torch.fft.irfft2(f1)
        x3 = torch.fft.irfft2(f3)

        out = torch.cat([x1, x2, x3], dim=1)
        out = self.fusion(out)

        return out


if __name__ == "__main__":
    x = torch.randn(1, 96, 128, 128)  # 3C
    model = PFM(dim=32)

    y = model(x)
    print("PFM input :", x.shape)
    print("PFM output:", y.shape)