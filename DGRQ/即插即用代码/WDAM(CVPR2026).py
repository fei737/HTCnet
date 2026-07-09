import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange
# It Takes Two: A Duet of Periodicity and Directionality for Burst Flicker Removal, CVPR 2026.
# https://arxiv.org/pdf/2603.22794
# https://github.com/qulishen/Flickerformer
# 本代码仅供参考，请以官方代码为准
# ==================== Haar DWT ====================
def dwt_haar(x):
    x01 = x[:, :, 0::2, :] / 2
    x02 = x[:, :, 1::2, :] / 2

    x1 = x01[:, :, :, 0::2]
    x2 = x02[:, :, :, 0::2]
    x3 = x01[:, :, :, 1::2]
    x4 = x02[:, :, :, 1::2]

    LL = x1 + x2 + x3 + x4
    HL = -x1 - x2 + x3 + x4
    LH = -x1 + x2 - x3 + x4
    HH = x1 - x2 - x3 + x4

    return LL, LH, HL, HH


# ==================== Haar IDWT ====================
def idwt_haar(LL, LH, HL, HH):
    B, C, H, W = LL.shape

    out = torch.zeros(B, C, H*2, W*2, device=LL.device)

    x1 = LL - HL - LH + HH
    x2 = LL - HL + LH - HH
    x3 = LL + HL - LH - HH
    x4 = LL + HL + LH + HH

    out[:, :, 0::2, 0::2] = x1
    out[:, :, 1::2, 0::2] = x2
    out[:, :, 0::2, 1::2] = x3
    out[:, :, 1::2, 1::2] = x4

    return out


# ==================== WDAM ====================
class WDAM(nn.Module):
    def __init__(self, dim, num_heads=4, bias=False):
        super().__init__()

        self.num_heads = num_heads

        # 高频引导
        self.high_conv = nn.Sequential(
            nn.Conv2d(dim*2, dim*2, 3, padding=1, groups=2, bias=bias),
            nn.ReLU(inplace=True),
            nn.Conv2d(dim*2, dim, 1, bias=bias),
            nn.ReLU(inplace=True)
        )

        self.high_out = nn.Sequential(
            nn.Conv2d(dim*3, dim*3, 3, padding=1, groups=3, bias=bias),
            nn.ReLU(inplace=True)
        )

        # Attention
        self.qkv = nn.Conv2d(dim, dim*3, 1, bias=bias)
        self.qkv_dw = nn.Conv2d(dim*3, dim*3, 3, padding=1, groups=dim*3, bias=bias)
        self.proj = nn.Conv2d(dim, dim, 1, bias=bias)

        self.temperature = nn.Parameter(torch.ones(num_heads, 1, 1))

    def forward(self, x):
        B, C, H, W = x.shape

        # ================= DWT =================
        LL, LH, HL, HH = dwt_haar(x)

        # 高频引导
        filter_hv = self.high_conv(torch.cat([LH, HL], dim=1))

        # ================= Attention =================
        qkv = self.qkv_dw(self.qkv(LL))
        q, k, v = qkv.chunk(3, dim=1)

        v = v * filter_hv + v

        # reshape
        q = rearrange(q, 'b (h c) h1 w1 -> b h c (h1 w1)', h=self.num_heads)
        k = rearrange(k, 'b (h c) h1 w1 -> b h c (h1 w1)', h=self.num_heads)
        v = rearrange(v, 'b (h c) h1 w1 -> b h c (h1 w1)', h=self.num_heads)

        q = F.normalize(q, dim=-1)
        k = F.normalize(k, dim=-1)

        attn = (q @ k.transpose(-2, -1)) * self.temperature
        attn = attn.softmax(dim=-1)

        out = (attn @ v)

        out = rearrange(out, 'b h c (hw) -> b (h c) hw', h=self.num_heads, hw=(H//2)*(W//2))
        out = out.view(B, C, H//2, W//2)

        out = self.proj(out)

        # ================= 高频重建 =================
        Yh = self.high_out(torch.cat([LH, HL, HH], dim=1))
        LH, HL, HH = Yh.chunk(3, dim=1)

        x = idwt_haar(out, LH, HL, HH)

        return x


if __name__ == "__main__":
    x = torch.randn(1, 32, 128, 128)
    model = WDAM(dim=32)

    y = model(x)

    print("input :", x.shape)
    print("output:", y.shape)