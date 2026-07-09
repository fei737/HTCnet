import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange
# 本代码仅供参考，请以论文官方代码为准
# Leveraging Multispectral Sensors for Color Correction in Mobile Cameras (CVPR 2026)
# https://arxiv.org/pdf/2512.08441
# https://github.com/LucaCogo/Mobile-Spectral-CC
class TrueMCA(nn.Module):
    def __init__(self, dim, num_heads=4):
        super().__init__()
        self.num_heads = num_heads

        self.q = nn.Conv2d(dim, dim, 3, padding=1)
        self.k = nn.Conv2d(dim, dim, 3, padding=1)
        self.v = nn.Conv2d(dim, dim, 3, padding=1)

        self.a = nn.Sequential(
            nn.Conv2d(dim, dim, 3, padding=1),
            nn.Conv2d(dim, dim, 1)
        )

        self.proj = nn.Conv2d(dim, dim, 1)

    def forward(self, x):
        b, c, h, w = x.shape

        q = rearrange(self.q(x), 'b c h w -> b c (h w)')
        k = rearrange(self.k(x), 'b c h w -> b c (h w)')
        v = rearrange(self.v(x), 'b c h w -> b c (h w)')
        a = rearrange(self.a(x), 'b c h w -> b c (h w)')

        q = F.normalize(q, dim=-1)
        k = F.normalize(k, dim=-1)
        a = F.normalize(a, dim=-1)

        attn1 = q @ a.transpose(-2, -1)
        attn2 = a @ k.transpose(-2, -1)

        out = attn1 @ (attn2 @ v)

        out = rearrange(out, 'b c (h w) -> b c h w', h=h, w=w)

        return x + self.proj(out)

if __name__ == "__main__":
    model = TrueMCA(dim=32)

    x = torch.randn(1, 32, 64, 64)  # [B, C, H, W]
    out = model(x)
    print(out.shape)
