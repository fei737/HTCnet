import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange
# It Takes Two: A Duet of Periodicity and Directionality for Burst Flicker Removal, CVPR 2026.
# https://arxiv.org/pdf/2603.22794
# https://github.com/qulishen/Flickerformer
# 本代码仅供参考，请以官方代码为准
class AFFN(nn.Module):
    def __init__(self, dim, expansion_factor=2.66, bias=False, patch_size=8):
        super().__init__()

        hidden_dim = int(dim * expansion_factor)
        self.patch_size = patch_size

        self.project_in = nn.Conv2d(dim, hidden_dim * 2, 1, bias=bias)

        self.dwconv = nn.Conv2d(
            hidden_dim * 2, hidden_dim * 2,
            kernel_size=3, padding=1,
            groups=hidden_dim * 2, bias=bias
        )

        self.project_out = nn.Conv2d(hidden_dim, dim, 1, bias=bias)

        # 频域权重
        self.fft_weight = nn.Parameter(
            torch.ones(hidden_dim * 2, 1, 1, patch_size, patch_size // 2 + 1)
        )

        # 自适应参数（核心）
        self.alpha = nn.Parameter(torch.tensor(0.5))
        self.beta = nn.Parameter(torch.tensor(0.5))

    def forward(self, x):
        B, C, H, W = x.shape

        x = self.project_in(x)

        # patch化
        x_patch = rearrange(
            x, 'b c (h ph) (w pw) -> b c h w ph pw',
            ph=self.patch_size, pw=self.patch_size
        )

        # FFT
        Xf = torch.fft.rfft2(x_patch.float())
        Xf = Xf * self.fft_weight

        # 自相关
        power = Xf * torch.conj(Xf)
        R = torch.fft.irfft2(power, s=(self.patch_size, self.patch_size))

        # 融合
        Xf_new = Xf + self.alpha * power
        x_patch = torch.fft.irfft2(Xf_new, s=(self.patch_size, self.patch_size))
        x_patch = x_patch + self.beta * R

        # reshape回来
        x = rearrange(
            x_patch, 'b c h w ph pw -> b c (h ph) (w pw)',
            ph=self.patch_size, pw=self.patch_size
        )

        x1, x2 = self.dwconv(x).chunk(2, dim=1)
        x = F.gelu(x1) * x2

        x = self.project_out(x)

        return x


if __name__ == "__main__":
    x = torch.randn(1, 32, 128, 128)
    model = AFFN(dim=32)
    y = model(x)
    print("AFFN input :", x.shape)
    print("AFFN output:", y.shape)