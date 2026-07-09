import torch
import torch.nn as nn
# BinaryAttention: One-Bit QK-Attention for Vision and Diffusion Transformers[CVPR 2026]
# https://arxiv.org/pdf/2603.09582v1
# https://github.com/EdwardChasel/BinaryAttention
# 本代码仅供参考，请以官方代码为准
# ====== 基础量化函数 ======
def binarize(x):
    return x.sign()

def round_ste(x):
    return (x - x.detach()) + x.detach().round()

def symquantize(x, clip_val, bits=8, dequantize=True):
    qmin = -(2 ** (bits - 1))
    qmax = (2 ** (bits - 1)) - 1

    scale = (clip_val[1] - clip_val[0]) / (qmax - qmin)
    x_clipped = torch.clamp(x, clip_val[0], clip_val[1])
    x_q = round_ste(x_clipped / scale).clamp(qmin, qmax)

    if dequantize:
        return x_q * scale
    return x_q


# ====== Binary Attention 模块 ======
class BinaryAttention(nn.Module):
    def __init__(self, dim, num_heads=8, attn_quant=True, pv_quant=True):
        super().__init__()
        self.num_heads = num_heads
        self.scale = (dim // num_heads) ** -0.5

        self.qkv = nn.Linear(dim, dim * 3)
        self.proj = nn.Linear(dim, dim)

        self.attn_quant = attn_quant
        self.pv_quant = pv_quant

    # ===== 量化函数 =====
    @staticmethod
    def _quantize(x):
        s = x.abs().mean(dim=-2, keepdim=True).mean(dim=-1, keepdim=True)
        sign = binarize(x)
        return s * sign

    @staticmethod
    def _quantize_p(x):
        qmax = 255
        s = 1.0 / qmax
        q = round_ste(x / s).clamp(0, qmax)
        return s * q

    @staticmethod
    def _quantize_v(x, bits=8):
        act_clip_val = torch.tensor([-2.0, 2.0], device=x.device)
        return symquantize(x, act_clip_val, bits, True)

    # ===== forward =====
    def forward(self, x):
        """
        x: [B, N, C]
        """
        print(f"[Input] x shape: {x.shape}")

        B, N, C = x.shape

        # ===== QKV =====
        qkv = self.qkv(x)  # [B, N, 3C]
        qkv = qkv.reshape(B, N, 3, self.num_heads, C // self.num_heads)
        qkv = qkv.permute(2, 0, 3, 1, 4)  # [3, B, heads, N, dim]

        q, k, v = qkv[0], qkv[1], qkv[2]

        print(f"[Q] shape: {q.shape}")
        print(f"[K] shape: {k.shape}")
        print(f"[V] shape: {v.shape}")

        # ===== Binary Attention =====
        if self.attn_quant:
            q = self._quantize(q)
            k = self._quantize(k)

        attn = (q @ k.transpose(-2, -1)) * self.scale
        print(f"[Attn logits] shape: {attn.shape}")

        attn = attn.softmax(dim=-1)

        # ===== 量化 P/V =====
        if self.pv_quant:
            attn = self._quantize_p(attn)
            v = self._quantize_v(v)

        # ===== 输出 =====
        out = attn @ v  # [B, heads, N, dim]
        print(f"[Attn @ V] shape: {out.shape}")

        out = out.transpose(1, 2).reshape(B, N, C)
        out = self.proj(out)

        print(f"[Output] shape: {out.shape}")

        return out


if __name__ == "__main__":
    x = torch.randn(1, 196, 768)  # 模拟 ViT token

    attn = BinaryAttention(dim=768, num_heads=12)
    out = attn(x)