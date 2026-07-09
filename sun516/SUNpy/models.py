import math
import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import segmentation_models_pytorch as smp
from GSA import Decomposed_GSA, GeoPriorGen


def _is_main_process():
    return int(os.environ.get("RANK", "0")) == 0


class ConvBNAct(nn.Sequential):
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, padding=None,
                 groups=1, bias=False, activation=nn.ReLU):
        if padding is None:
            padding = kernel_size // 2
        layers = [
            nn.Conv2d(
                in_channels,
                out_channels,
                kernel_size=kernel_size,
                stride=stride,
                padding=padding,
                groups=groups,
                bias=bias
            ),
            nn.BatchNorm2d(out_channels)
        ]
        if activation is not None:
            try:
                layers.append(activation(inplace=True))
            except TypeError:
                layers.append(activation())
        super().__init__(*layers)


class ConvGNAct(nn.Sequential):
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, padding=None,
                 groups=1, bias=False, activation=nn.ReLU, num_groups=32):
        if padding is None:
            padding = kernel_size // 2
        gn_groups = min(num_groups, out_channels)
        while out_channels % gn_groups != 0:
            gn_groups -= 1
        layers = [
            nn.Conv2d(
                in_channels,
                out_channels,
                kernel_size=kernel_size,
                stride=stride,
                padding=padding,
                groups=groups,
                bias=bias
            ),
            nn.GroupNorm(gn_groups, out_channels)
        ]
        if activation is not None:
            try:
                layers.append(activation(inplace=True))
            except TypeError:
                layers.append(activation())
        super().__init__(*layers)


class HDDifferentialEnhancer(nn.Module):
    def __init__(self, in_channels=2):
        super().__init__()
        self.diff_conv = nn.Conv2d(
            in_channels,
            in_channels,
            kernel_size=3,
            padding=1,
            groups=in_channels,
            bias=False
        )
        laplacian = torch.tensor(
            [[-1.0, -1.0, -1.0], [-1.0, 8.0, -1.0], [-1.0, -1.0, -1.0]],
            dtype=torch.float32
        )
        self.diff_conv.weight.requires_grad = False
        self.register_buffer("laplacian", laplacian.view(1, 1, 3, 3).repeat(in_channels, 1, 1, 1))
        self.gate_generator = nn.Sequential(
            nn.Conv2d(in_channels * 2, max(in_channels * 8, 16), kernel_size=3, padding=1, bias=False),
            nn.GroupNorm(num_groups=4, num_channels=max(in_channels * 8, 16)),
            nn.GELU(),
            nn.Conv2d(max(in_channels * 8, 16), in_channels, kernel_size=1),
            nn.Sigmoid()
        )
        self.diff_mix = nn.Sequential(
            nn.Conv2d(in_channels, max(in_channels * 8, 16), kernel_size=3, padding=1, bias=False),
            nn.GroupNorm(num_groups=4, num_channels=max(in_channels * 8, 16)),
            nn.GELU(),
            nn.Conv2d(max(in_channels * 8, 16), in_channels, kernel_size=1)
        )
        self.enhance_scale = nn.Parameter(torch.tensor(0.25))

    def forward(self, hd):
        weight = self.laplacian.to(device=hd.device, dtype=hd.dtype)
        hd_diff = torch.abs(F.conv2d(hd, weight, padding=1, groups=hd.shape[1]))
        normalizer = hd_diff.flatten(2).amax(dim=2).view(hd.shape[0], hd.shape[1], 1, 1).clamp_min(1e-6)
        hd_diff = hd_diff / normalizer
        gate = self.gate_generator(torch.cat([hd, hd_diff], dim=1))
        diff_residual = self.diff_mix(hd_diff)
        scale = torch.clamp(self.enhance_scale, 0.0, 1.0)
        return hd * (1.0 + scale * gate) + scale * gate * diff_residual


class FixedBoundaryExtractor(nn.Module):
    def __init__(self, out_channels):
        super().__init__()
        hidden_channels = max(out_channels // 2, 16)
        self.smooth = nn.AvgPool2d(kernel_size=3, stride=1, padding=1)
        self.proj = nn.Sequential(
            ConvBNAct(3, hidden_channels, kernel_size=3, activation=nn.GELU),
            ConvBNAct(hidden_channels, out_channels, kernel_size=1, activation=nn.GELU)
        )
        sobel_x = torch.tensor([[-1.0, 0.0, 1.0], [-2.0, 0.0, 2.0], [-1.0, 0.0, 1.0]], dtype=torch.float32)
        sobel_y = torch.tensor([[-1.0, -2.0, -1.0], [0.0, 0.0, 0.0], [1.0, 2.0, 1.0]], dtype=torch.float32)
        laplacian = torch.tensor([[0.0, -1.0, 0.0], [-1.0, 4.0, -1.0], [0.0, -1.0, 0.0]], dtype=torch.float32)
        self.register_buffer("sobel_x", sobel_x.view(1, 1, 3, 3))
        self.register_buffer("sobel_y", sobel_y.view(1, 1, 3, 3))
        self.register_buffer("laplacian", laplacian.view(1, 1, 3, 3))

    def forward(self, x):
        x_mean = self.smooth(x.mean(dim=1, keepdim=True))
        grad_x = torch.abs(F.conv2d(x_mean, self.sobel_x, padding=1))
        grad_y = torch.abs(F.conv2d(x_mean, self.sobel_y, padding=1))
        lap = torch.abs(F.conv2d(x_mean, self.laplacian, padding=1))
        boundary = torch.cat([grad_x, grad_y, lap], dim=1)
        normalizer = boundary.flatten(2).amax(dim=2, keepdim=True).unsqueeze(-1).clamp_min(1e-6)
        boundary = boundary / normalizer
        return self.proj(boundary)


class AngleGradientExtractor(nn.Module):
    def __init__(self):
        super().__init__()
        self.smooth = nn.AvgPool2d(kernel_size=3, stride=1, padding=1)
        kernel_0 = [[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]]
        kernel_45 = [[-2, -1, 0], [-1, 0, 1], [0, 1, 2]]
        kernel_90 = [[-1, -2, -1], [0, 0, 0], [1, 2, 1]]
        kernel_135 = [[0, 1, 2], [-1, 0, 1], [-2, -1, 0]]

        kernels = torch.tensor([kernel_0, kernel_45, kernel_90, kernel_135], dtype=torch.float32)
        self.register_buffer("fixed_kernels", kernels.view(4, 1, 3, 3))
        self.direction_gate = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(4, 4, kernel_size=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(4, 4, kernel_size=1),
            nn.Sigmoid()
        )
        self.refine = nn.Sequential(
            nn.Conv2d(4, 8, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(8),
            nn.GELU(),
            nn.Conv2d(8, 1, kernel_size=1, bias=False)
        )

    def forward(self, hha):
        angle_channel = self.smooth(hha[:, 2:3, :, :])
        kernels = self.fixed_kernels.to(hha.device)
        angle_edges = torch.abs(F.conv2d(angle_channel, kernels, padding=1))
        angle_edges = angle_edges * self.direction_gate(angle_edges)
        return self.refine(angle_edges)


class AngleSideGuidePyramid(nn.Module):
    def __init__(self, out_channels_list):
        super().__init__()
        self.proj_layers = nn.ModuleList([
            nn.Sequential(
                ConvBNAct(1, max(ch // 4, 16), kernel_size=3, activation=nn.GELU),
                ConvBNAct(max(ch // 4, 16), ch, kernel_size=1, activation=nn.GELU)
            )
            for ch in out_channels_list
        ])

    def forward(self, angle_grad, target_sizes):
        guides = []
        for proj, size in zip(self.proj_layers, target_sizes):
            guide = F.interpolate(angle_grad, size=size, mode="bilinear", align_corners=False)
            guides.append(proj(guide))
        return guides


class CMFR_Module(nn.Module):
    def __init__(self, channels=64):
        super().__init__()
        hidden_channels = max(channels // 2, 16)
        self.rgb_proj = ConvBNAct(channels, channels, kernel_size=1)
        self.hha_proj = ConvBNAct(channels, channels, kernel_size=1)
        self.geo_context = nn.Sequential(
            ConvBNAct(channels * 2, channels, kernel_size=1),
            ConvBNAct(channels, channels, kernel_size=3)
        )
        self.confidence = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(channels, hidden_channels, kernel_size=1, bias=False),
            nn.ReLU(inplace=True),
            nn.Conv2d(hidden_channels, channels, kernel_size=1),
            nn.Sigmoid()
        )
        self.rgb_gate = nn.Sequential(
            nn.Conv2d(channels * 2, hidden_channels, kernel_size=1, bias=False),
            nn.BatchNorm2d(hidden_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(hidden_channels, channels, kernel_size=3, padding=1),
            nn.Sigmoid()
        )
        self.hha_gate = nn.Sequential(
            nn.Conv2d(channels * 2, hidden_channels, kernel_size=1, bias=False),
            nn.BatchNorm2d(hidden_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(hidden_channels, channels, kernel_size=3, padding=1),
            nn.Sigmoid()
        )
        self.out_proj = nn.Sequential(
            ConvBNAct(channels * 2, channels, kernel_size=1),
            ConvBNAct(channels, channels, kernel_size=3, activation=nn.GELU),
            nn.Conv2d(channels, channels, kernel_size=1, bias=False),
            nn.BatchNorm2d(channels)
        )
        self.layer_scale = nn.Parameter(torch.ones(1, channels, 1, 1) * 1e-4)

    def forward(self, rgb_f, hha_f):
        rgb_feat = self.rgb_proj(rgb_f)
        hha_feat = self.hha_proj(hha_f)
        context = self.geo_context(torch.cat([rgb_feat, hha_feat], dim=1))
        confidence = self.confidence(context)

        rgb_enhanced = rgb_feat + confidence * self.hha_gate(torch.cat([hha_feat, context], dim=1)) * hha_feat
        hha_enhanced = hha_feat + confidence * self.rgb_gate(torch.cat([rgb_feat, context], dim=1)) * rgb_feat

        fused = self.out_proj(torch.cat([rgb_enhanced, hha_enhanced], dim=1))
        return rgb_f + self.layer_scale * fused


class HDStructureFilter(nn.Module):
    def __init__(self, channels):
        super().__init__()
        hidden_channels = max(channels // 4, 16)
        self.smooth = nn.AvgPool2d(kernel_size=3, stride=1, padding=1)
        self.low_proj = ConvBNAct(channels, channels, kernel_size=3, activation=nn.GELU)
        self.guide_proj = nn.Sequential(
            ConvBNAct(channels, hidden_channels, kernel_size=1, activation=nn.GELU),
            nn.Conv2d(hidden_channels, 1, kernel_size=3, padding=1),
            nn.Sigmoid()
        )
        self.texture_gate = nn.Sequential(
            ConvBNAct(channels + 1, hidden_channels, kernel_size=1, activation=nn.GELU),
            nn.Conv2d(hidden_channels, channels, kernel_size=1),
            nn.Sigmoid()
        )
        self.out_proj = nn.Sequential(
            ConvBNAct(channels, channels, kernel_size=3, activation=nn.GELU),
            nn.Conv2d(channels, channels, kernel_size=1, bias=False),
            nn.BatchNorm2d(channels)
        )
        self.layer_scale = nn.Parameter(torch.ones(1, channels, 1, 1) * 1e-4)

    def forward(self, hd_feat, guide_feat):
        low_freq = self.smooth(hd_feat)
        structure_base = self.low_proj(low_freq)
        high_freq = hd_feat - low_freq
        guide_map = guide_feat.mean(dim=1, keepdim=True)
        guide_map = self.guide_proj(guide_feat) * (0.5 + 0.5 * guide_map.sigmoid())
        texture_keep = self.texture_gate(torch.cat([structure_base, guide_map], dim=1))
        filtered = structure_base + high_freq * texture_keep * guide_map
        return hd_feat + self.layer_scale * self.out_proj(filtered)


def _projection_block(in_channels, out_channels):
    if in_channels == out_channels:
        return nn.Identity()
    return nn.Sequential(
        nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=False),
        nn.BatchNorm2d(out_channels),
        nn.ReLU(inplace=True)
    )


class MultiHeadTopologicalAttention(nn.Module):
    def __init__(self, dim, num_heads=8, init_value=2, heads_range=4, layer_init_values=1e-5):
        super().__init__()
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = head_dim ** -0.5

        self.qkv = nn.Linear(dim, dim * 3, bias=False)
        self.proj = nn.Linear(dim, dim)
        decay = torch.log(
            1 - 2 ** (-init_value - heads_range * torch.arange(num_heads, dtype=torch.float) / num_heads)
        )
        self.register_buffer("decay", decay)
        self.gamma = nn.Parameter(torch.ones(1, num_heads, 1, 1) * 2.0)
        self.lepe = nn.Conv2d(dim, dim, kernel_size=5, stride=1, padding=2, groups=dim)
        self.layer_scale = nn.Parameter(layer_init_values * torch.ones(1, dim, 1, 1))

    def forward(self, rgb_feat, hha_feat, be_feat):
        b, c, h, w = rgb_feat.shape
        n = h * w
        x = rgb_feat + hha_feat
        x_flat = x.flatten(2).transpose(1, 2)

        be_spatial = torch.mean(be_feat, dim=1, keepdim=True)
        be_flat = be_spatial.flatten(2)
        topo_diff = torch.abs(be_flat.transpose(1, 2) - be_flat)
        multi_head_topo = topo_diff.unsqueeze(1) * torch.abs(self.decay.view(1, self.num_heads, 1, 1))

        qkv = self.qkv(x_flat).reshape(b, n, 3, self.num_heads, c // self.num_heads).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]

        v_2d = v.transpose(1, 2).reshape(b, n, c).transpose(1, 2).reshape(b, c, h, w).contiguous()
        lepe_out = self.lepe(v_2d)

        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = (attn - self.gamma * multi_head_topo).softmax(dim=-1)

        out = (attn @ v).transpose(1, 2).reshape(b, n, c)
        out = out.transpose(1, 2).reshape(b, c, h, w)
        out = out + lepe_out
        out = self.proj(out.flatten(2).transpose(1, 2))
        out = out.transpose(1, 2).reshape(b, c, h, w)
        return x + out * self.layer_scale, attn


class AngleGuidedGSA(nn.Module):
    def __init__(self, dim, num_heads=4, layer_init_values=1e-5):
        super().__init__()
        self.norm = nn.LayerNorm(dim)
        self.geo_prior = GeoPriorGen(embed_dim=dim, num_heads=num_heads, initial_value=3, heads_range=2)
        self.gsa = Decomposed_GSA(embed_dim=dim, num_heads=num_heads, value_factor=1)
        self.layer_scale = nn.Parameter(layer_init_values * torch.ones(1, dim, 1, 1))

    @staticmethod
    def _cast_rel_pos(rel_pos, dtype):
        (sin, cos), (mask_h, mask_w) = rel_pos
        return ((sin.to(dtype=dtype), cos.to(dtype=dtype)), (mask_h.to(dtype=dtype), mask_w.to(dtype=dtype)))

    def forward(self, x, guide_feat):
        b, c, h, w = x.shape
        guide_prior = guide_feat if guide_feat.shape[1] == 1 else guide_feat.mean(dim=1, keepdim=True)
        rel_pos = self.geo_prior((h, w), guide_prior, split_or_not=True)
        rel_pos = self._cast_rel_pos(rel_pos, x.dtype)
        x_bhwc = x.permute(0, 2, 3, 1).contiguous()
        x_norm = self.norm(x_bhwc)
        gsa_out = self.gsa(x_norm, rel_pos).permute(0, 3, 1, 2).contiguous()
        return x + self.layer_scale * gsa_out


class LowLevelGuidedFusion(nn.Module):
    def __init__(self, rgb_channels, hd_channels, aux_channels, out_channels):
        super().__init__()
        self.rgb_proj = _projection_block(rgb_channels, out_channels)
        self.hd_proj = _projection_block(hd_channels, out_channels)
        self.aux_proj = _projection_block(aux_channels, out_channels)
        mid_channels = max(out_channels // 2, 1)

        self.condition_proj = nn.Sequential(
            ConvBNAct(out_channels * 2, mid_channels, kernel_size=3, activation=nn.GELU),
            ConvBNAct(mid_channels, mid_channels, kernel_size=3, activation=nn.GELU),
        )
        self.sft_gamma = nn.Conv2d(mid_channels, out_channels, kernel_size=1)
        self.sft_beta = nn.Conv2d(mid_channels, out_channels, kernel_size=1)
        self.offset_head = nn.Sequential(
            nn.Conv2d(mid_channels, mid_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(mid_channels),
            nn.GELU(),
            nn.Conv2d(mid_channels, 2, kernel_size=3, padding=1),
        )
        self.common_enhance = nn.Sequential(
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.GELU(),
        )
        self.texture_filter = nn.Sequential(
            nn.Conv2d(out_channels * 2 + 1, mid_channels, kernel_size=1, bias=False),
            nn.BatchNorm2d(mid_channels),
            nn.GELU(),
            nn.Conv2d(mid_channels, out_channels, kernel_size=1),
            nn.Sigmoid(),
        )
        self.fusion = nn.Sequential(
            nn.Conv2d(out_channels * 4 + 1, out_channels, kernel_size=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.GELU(),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.GELU(),
            nn.Conv2d(out_channels, out_channels, kernel_size=1, bias=False),
            nn.BatchNorm2d(out_channels)
        )
        self.layer_scale = nn.Parameter(torch.ones(1, out_channels, 1, 1) * 1e-4)

    @staticmethod
    def _warp_with_offsets(x, offsets):
        b, _, h, w = x.shape
        dtype = x.dtype
        device = x.device
        yy, xx = torch.meshgrid(
            torch.linspace(-1.0, 1.0, h, device=device, dtype=dtype),
            torch.linspace(-1.0, 1.0, w, device=device, dtype=dtype),
            indexing="ij",
        )
        base_grid = torch.stack([xx, yy], dim=-1).unsqueeze(0).expand(b, h, w, 2)
        norm_x = max(w - 1, 1) / 2.0
        norm_y = max(h - 1, 1) / 2.0
        offsets = torch.tanh(offsets)
        offsets = torch.stack(
            [offsets[:, 0] / norm_x, offsets[:, 1] / norm_y],
            dim=-1,
        )
        return F.grid_sample(
            x,
            base_grid + offsets,
            mode="bilinear",
            padding_mode="border",
            align_corners=True,
        )

    def forward(self, rgb_feat, hd_feat, aux_feat):
        rgb_feat = self.rgb_proj(rgb_feat)
        hd_feat = self.hd_proj(hd_feat)
        aux_feat = self.aux_proj(aux_feat)

        condition = self.condition_proj(torch.cat([hd_feat, aux_feat], dim=1))
        offsets = self.offset_head(condition)
        rgb_aligned = self._warp_with_offsets(rgb_feat, offsets)

        gamma = 0.25 * torch.tanh(self.sft_gamma(condition))
        beta = 0.25 * torch.tanh(self.sft_beta(condition))
        rgb_modulated = rgb_aligned * (1.0 + gamma) + beta

        hd_norm = F.normalize(hd_feat, dim=1, eps=1e-6)
        rgb_parallel = (rgb_modulated * hd_norm).sum(dim=1, keepdim=True) * hd_norm
        rgb_orthogonal = rgb_modulated - rgb_parallel
        common_structure = self.common_enhance(rgb_parallel + hd_feat)

        guide_map = aux_feat.mean(dim=1, keepdim=True).sigmoid()
        texture_gate = self.texture_filter(torch.cat([rgb_orthogonal, hd_feat, guide_map], dim=1))
        filtered_texture = rgb_orthogonal * texture_gate

        fused = self.fusion(torch.cat([
            common_structure,
            filtered_texture,
            hd_feat,
            aux_feat,
            guide_map,
        ], dim=1))
        return rgb_feat + self.layer_scale * fused


class TransformerGuidedFusion(nn.Module):
    def __init__(self, rgb_channels, hd_channels, aux_channels, out_channels):
        super().__init__()
        self.rgb_proj = _projection_block(rgb_channels, out_channels)
        self.hd_proj = _projection_block(hd_channels, out_channels)
        self.aux_proj = _projection_block(aux_channels, out_channels)
        self.transformer = MultiHeadTopologicalAttention(dim=out_channels)
        self.post_fusion = nn.Sequential(
            nn.Conv2d(out_channels * 2, out_channels, kernel_size=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, rgb_feat, hd_feat, aux_feat):
        rgb_feat = self.rgb_proj(rgb_feat)
        hd_feat = self.hd_proj(hd_feat)
        aux_feat = self.aux_proj(aux_feat)
        transformer_feat, _ = self.transformer(rgb_feat, hd_feat, aux_feat)
        return self.post_fusion(torch.cat([transformer_feat, aux_feat], dim=1)) + rgb_feat


class AdvancedGlobalTransformerFusion(nn.Module):
    def __init__(self, rgb_channels, hd_channels, aux_channels, out_channels, safe_mode=False):
        super().__init__()
        self.safe_mode = safe_mode
        self.rgb_proj = _projection_block(rgb_channels, out_channels)
        self.hd_proj = _projection_block(hd_channels, out_channels)
        self.aux_proj = _projection_block(aux_channels, out_channels)
        self.layout_encoder = nn.Sequential(
            ConvBNAct(out_channels, out_channels, kernel_size=3, groups=out_channels, activation=nn.GELU),
            ConvBNAct(out_channels, out_channels, kernel_size=1, activation=nn.GELU),
        )
        self.layout_pool = nn.AdaptiveAvgPool2d((8, 8))
        self.layout_norm = nn.LayerNorm(out_channels)
        self.semantic_norm = nn.LayerNorm(out_channels)
        self.layout_cross_attn = nn.MultiheadAttention(out_channels, num_heads=8, batch_first=True)
        self.scene_gate = nn.Sequential(
            nn.Conv2d(out_channels, max(out_channels // 4, 16), kernel_size=1),
            nn.GELU(),
            nn.Conv2d(max(out_channels // 4, 16), out_channels, kernel_size=1),
            nn.Sigmoid()
        )
        self.asym_fusion = nn.Sequential(
            nn.Conv2d(out_channels * 3, out_channels, kernel_size=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.GELU(),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.GELU(),
        )
        self.transformer = AngleGuidedGSA(dim=out_channels, num_heads=8) if not safe_mode else None
        self.safe_fusion = nn.Sequential(
            nn.Conv2d(out_channels * 3, out_channels, kernel_size=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        ) if safe_mode else None
        self.ffn = nn.Sequential(
            nn.Conv2d(out_channels, out_channels * 2, kernel_size=1, bias=False),
            nn.BatchNorm2d(out_channels * 2),
            nn.GELU(),
            nn.Conv2d(
                out_channels * 2,
                out_channels * 2,
                kernel_size=3,
                padding=2,
                dilation=2,
                groups=out_channels * 2,
                bias=False
            ),
            nn.BatchNorm2d(out_channels * 2),
            nn.GELU(),
            nn.Conv2d(out_channels * 2, out_channels, kernel_size=1, bias=False),
            nn.BatchNorm2d(out_channels)
        )
        self.layer_scale = nn.Parameter(1e-5 * torch.ones(1, out_channels, 1, 1))

    def forward(self, rgb_feat, hd_feat, aux_feat):
        rgb_feat = self.rgb_proj(rgb_feat)
        hd_feat = self.hd_proj(hd_feat)
        aux_feat = self.aux_proj(aux_feat)

        hd_layout = self.layout_encoder(hd_feat)
        layout_tokens = self.layout_pool(hd_layout).flatten(2).transpose(1, 2)
        semantic_tokens = (rgb_feat + 0.25 * aux_feat).flatten(2).transpose(1, 2)
        layout_context, _ = self.layout_cross_attn(
            self.layout_norm(layout_tokens),
            self.semantic_norm(semantic_tokens),
            self.semantic_norm(semantic_tokens),
        )
        scene_token = layout_context.mean(dim=1).view(rgb_feat.size(0), rgb_feat.size(1), 1, 1)
        scene_gate = self.scene_gate(scene_token)
        asym_feat = self.asym_fusion(torch.cat([
            rgb_feat * (1.0 + scene_gate),
            hd_layout,
            aux_feat,
        ], dim=1))

        if self.safe_mode:
            attn_feat = self.safe_fusion(torch.cat([asym_feat, hd_layout, aux_feat], dim=1))
        else:
            guide_map = aux_feat.mean(dim=1, keepdim=True)
            attn_feat = self.transformer(asym_feat + hd_layout, guide_map)
        out = attn_feat + self.layer_scale * self.ffn(attn_feat)
        return out + rgb_feat


class UnifiedGeometryFusion(nn.Module):
    def __init__(
        self,
        rgb_channels,
        hd_channels,
        aux_channels,
        out_channels,
        scale_index,
        num_scales,
        safe_mode=False,
        max_attention_tokens=1024,
    ):
        super().__init__()
        self.scale_index = scale_index
        self.num_scales = max(num_scales, 1)
        self.max_attention_tokens = max_attention_tokens
        self.rgb_proj = _projection_block(rgb_channels, out_channels)
        self.hd_proj = _projection_block(hd_channels, out_channels)
        self.aux_proj = _projection_block(aux_channels, out_channels)
        mid_channels = max(out_channels // 2, 16)

        self.condition_proj = nn.Sequential(
            ConvBNAct(out_channels * 2, mid_channels, kernel_size=3, activation=nn.GELU),
            ConvBNAct(mid_channels, mid_channels, kernel_size=3, activation=nn.GELU),
        )
        self.sft_gamma = nn.Conv2d(mid_channels, out_channels, kernel_size=1)
        self.sft_beta = nn.Conv2d(mid_channels, out_channels, kernel_size=1)
        self.offset_head = nn.Sequential(
            nn.Conv2d(mid_channels, mid_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(mid_channels),
            nn.GELU(),
            nn.Conv2d(mid_channels, 2, kernel_size=3, padding=1),
        )
        self.common_enhance = nn.Sequential(
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.GELU(),
        )
        self.texture_filter = nn.Sequential(
            nn.Conv2d(out_channels * 2 + 1, mid_channels, kernel_size=1, bias=False),
            nn.BatchNorm2d(mid_channels),
            nn.GELU(),
            nn.Conv2d(mid_channels, out_channels, kernel_size=1),
            nn.Sigmoid(),
        )
        self.local_fusion = nn.Sequential(
            nn.Conv2d(out_channels * 4 + 1, out_channels, kernel_size=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.GELU(),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.GELU(),
            nn.Conv2d(out_channels, out_channels, kernel_size=1, bias=False),
            nn.BatchNorm2d(out_channels)
        )

        attn_heads = 8 if out_channels % 8 == 0 else 4 if out_channels % 4 == 0 else 1
        self.semantic_attn = MultiHeadTopologicalAttention(dim=out_channels, num_heads=attn_heads)
        self.semantic_post = nn.Sequential(
            nn.Conv2d(out_channels * 2, out_channels, kernel_size=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.GELU(),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
        )
        self.deep_gsa = None
        if not safe_mode and scale_index >= max(num_scales - 2, 1):
            self.deep_gsa = AngleGuidedGSA(dim=out_channels, num_heads=attn_heads)

        self.scale_embed = nn.Parameter(torch.zeros(1, out_channels, 1, 1))
        self.route_gate = nn.Sequential(
            nn.Conv2d(out_channels * 4, mid_channels, kernel_size=1, bias=False),
            nn.GELU(),
            nn.Conv2d(mid_channels, 2, kernel_size=1)
        )
        with torch.no_grad():
            depth_ratio = scale_index / float(max(num_scales - 1, 1))
            self.route_gate[-1].bias.copy_(torch.tensor([1.0 - depth_ratio, depth_ratio]))

        self.local_scale = nn.Parameter(torch.ones(1, out_channels, 1, 1) * 1e-4)
        self.semantic_scale = nn.Parameter(torch.ones(1, out_channels, 1, 1) * 1e-4)

    @staticmethod
    def _warp_with_offsets(x, offsets):
        return LowLevelGuidedFusion._warp_with_offsets(x, offsets)

    def _local_branch(self, rgb_feat, hd_feat, aux_feat):
        condition = self.condition_proj(torch.cat([hd_feat, aux_feat], dim=1))
        rgb_aligned = self._warp_with_offsets(rgb_feat, self.offset_head(condition))
        gamma = 0.25 * torch.tanh(self.sft_gamma(condition))
        beta = 0.25 * torch.tanh(self.sft_beta(condition))
        rgb_modulated = rgb_aligned * (1.0 + gamma) + beta

        hd_norm = F.normalize(hd_feat, dim=1, eps=1e-6)
        rgb_parallel = (rgb_modulated * hd_norm).sum(dim=1, keepdim=True) * hd_norm
        rgb_orthogonal = rgb_modulated - rgb_parallel
        common_structure = self.common_enhance(rgb_parallel + hd_feat)

        guide_map = aux_feat.mean(dim=1, keepdim=True).sigmoid()
        texture_gate = self.texture_filter(torch.cat([rgb_orthogonal, hd_feat, guide_map], dim=1))
        filtered_texture = rgb_orthogonal * texture_gate
        fused = self.local_fusion(torch.cat([
            common_structure,
            filtered_texture,
            hd_feat,
            aux_feat,
            guide_map,
        ], dim=1))
        return rgb_feat + self.local_scale * fused

    def _attention_size(self, h, w):
        tokens = h * w
        if tokens <= self.max_attention_tokens:
            return h, w
        scale = math.sqrt(self.max_attention_tokens / float(tokens))
        return max(1, int(round(h * scale))), max(1, int(round(w * scale)))

    def _semantic_branch(self, rgb_feat, hd_feat, aux_feat):
        h, w = rgb_feat.shape[2:]
        attn_h, attn_w = self._attention_size(h, w)
        if (attn_h, attn_w) != (h, w):
            rgb_attn = F.interpolate(rgb_feat, size=(attn_h, attn_w), mode="bilinear", align_corners=False)
            hd_attn = F.interpolate(hd_feat, size=(attn_h, attn_w), mode="bilinear", align_corners=False)
            aux_attn = F.interpolate(aux_feat, size=(attn_h, attn_w), mode="bilinear", align_corners=False)
        else:
            rgb_attn, hd_attn, aux_attn = rgb_feat, hd_feat, aux_feat

        attn_feat, _ = self.semantic_attn(rgb_attn, hd_attn, aux_attn)
        if self.deep_gsa is not None:
            attn_feat = self.deep_gsa(attn_feat, aux_attn.mean(dim=1, keepdim=True))
        if attn_feat.shape[2:] != (h, w):
            attn_feat = F.interpolate(attn_feat, size=(h, w), mode="bilinear", align_corners=False)
        fused = self.semantic_post(torch.cat([attn_feat, aux_feat], dim=1))
        return rgb_feat + self.semantic_scale * fused

    def forward(self, rgb_feat, hd_feat, aux_feat):
        rgb_feat = self.rgb_proj(rgb_feat)
        hd_feat = self.hd_proj(hd_feat)
        aux_feat = self.aux_proj(aux_feat)

        local_feat = self._local_branch(rgb_feat, hd_feat, aux_feat)
        semantic_feat = self._semantic_branch(rgb_feat, hd_feat, aux_feat)

        route_context = torch.cat([
            F.adaptive_avg_pool2d(rgb_feat, 1),
            F.adaptive_avg_pool2d(hd_feat, 1),
            F.adaptive_avg_pool2d(aux_feat, 1),
            self.scale_embed.expand(rgb_feat.size(0), -1, -1, -1),
        ], dim=1)
        route = torch.softmax(self.route_gate(route_context), dim=1)
        local_delta = local_feat - rgb_feat
        semantic_delta = semantic_feat - rgb_feat
        return rgb_feat + route[:, 0:1] * local_delta + route[:, 1:2] * semantic_delta


class CrossScaleFlowAttention(nn.Module):
    def __init__(self, shallow_ch, deep_ch):
        super().__init__()
        self.align = nn.Sequential(
            nn.Conv2d(deep_ch, shallow_ch, kernel_size=1, bias=False),
            nn.BatchNorm2d(shallow_ch),
            nn.ReLU(inplace=True)
        )
        self.channel_gate = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(shallow_ch, max(shallow_ch // 2, 1), kernel_size=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(max(shallow_ch // 2, 1), shallow_ch, kernel_size=1),
            nn.Sigmoid()
        )
        self.spatial_gate = nn.Sequential(
            nn.Conv2d(shallow_ch * 2, max(shallow_ch // 2, 1), kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(max(shallow_ch // 2, 1), 1, kernel_size=1),
            nn.Sigmoid()
        )

    def forward(self, shallow_feat, deep_feat):
        deep_up = F.interpolate(deep_feat, size=shallow_feat.shape[2:], mode="bilinear", align_corners=False)
        deep_aligned = self.align(deep_up)
        c_gate = self.channel_gate(deep_aligned)
        s_gate = self.spatial_gate(torch.cat([shallow_feat, deep_aligned], dim=1))
        return shallow_feat * s_gate * c_gate + deep_aligned


class GatedEdgeInteraction(nn.Module):
    def __init__(self, n_classes, boundary_channels):
        super().__init__()
        self.semantic_proj = ConvBNAct(n_classes, n_classes, kernel_size=3, activation=nn.GELU)
        self.edge_proj = nn.Sequential(
            ConvBNAct(2, max(n_classes, 16), kernel_size=3, activation=nn.GELU),
            nn.Conv2d(max(n_classes, 16), n_classes, kernel_size=1),
            nn.Sigmoid()
        )
        self.boundary_proj = nn.Sequential(
            ConvBNAct(boundary_channels, n_classes, kernel_size=1, activation=nn.GELU),
            nn.Conv2d(n_classes, n_classes, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(n_classes)
        )
        self.fusion = nn.Sequential(
            ConvBNAct(n_classes * 3, n_classes, kernel_size=1, activation=nn.GELU),
            nn.Dropout2d(0.1),
            nn.Conv2d(n_classes, n_classes, kernel_size=1)
        )

    @staticmethod
    def _seg_boundary_map(seg_logits):
        probs = F.softmax(seg_logits, dim=1)
        c = probs.shape[1]
        kernel_x = torch.tensor(
            [[-1.0, 0.0, 1.0], [-2.0, 0.0, 2.0], [-1.0, 0.0, 1.0]],
            dtype=probs.dtype,
            device=probs.device
        ).view(1, 1, 3, 3).repeat(c, 1, 1, 1)
        kernel_y = torch.tensor(
            [[-1.0, -2.0, -1.0], [0.0, 0.0, 0.0], [1.0, 2.0, 1.0]],
            dtype=probs.dtype,
            device=probs.device
        ).view(1, 1, 3, 3).repeat(c, 1, 1, 1)
        grad_x = F.conv2d(probs, kernel_x, padding=1, groups=c)
        grad_y = F.conv2d(probs, kernel_y, padding=1, groups=c)
        boundary_map = (torch.abs(grad_x) + torch.abs(grad_y)).mean(dim=1, keepdim=True)
        normalizer = boundary_map.flatten(1).amax(dim=1).view(-1, 1, 1, 1).clamp_min(1e-6)
        return boundary_map / normalizer

    def forward(self, seg_base, edge_prob, boundary_feat):
        seg_boundary = self._seg_boundary_map(seg_base)
        edge_gate = self.edge_proj(torch.cat([edge_prob, seg_boundary], dim=1))
        seg_feat = self.semantic_proj(seg_base)
        boundary_bias = self.boundary_proj(boundary_feat)
        edge_activated_seg = (seg_feat + boundary_bias) * (1.0 + edge_gate)
        out = self.fusion(torch.cat([seg_base, edge_activated_seg, boundary_bias], dim=1))
        return out + seg_base


class LightweightUpsampleRefine(nn.Module):
    def __init__(self, in_channels, out_channels, scale_factor=4):
        super().__init__()
        self.scale_factor = scale_factor
        self.proj = nn.Conv2d(in_channels, out_channels, kernel_size=1) if in_channels != out_channels else nn.Identity()
        self.refine = nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1, groups=out_channels)

    def forward(self, x):
        x = self.proj(x)
        # Refine at low resolution first, then upsample, to avoid full-resolution
        # depthwise convolution on segmentation logits that can easily trigger OOM.
        x = x + self.refine(x)
        return F.interpolate(x, scale_factor=self.scale_factor, mode="bilinear", align_corners=False)


class DetailEnhancedEdgeHead(nn.Module):
    def __init__(self, detail_channels, shallow_channels, guide_channels, n_classes):
        super().__init__()
        mid_channels = max(detail_channels // 4, 32)
        self.n_classes = n_classes
        self.shallow_proj = nn.Sequential(
            ConvBNAct(shallow_channels, mid_channels, kernel_size=1, activation=nn.GELU),
            ConvBNAct(mid_channels, mid_channels, kernel_size=3, activation=nn.GELU)
        )
        self.guide_proj = nn.Sequential(
            ConvBNAct(guide_channels, mid_channels, kernel_size=1, activation=nn.GELU),
            ConvBNAct(mid_channels, mid_channels, kernel_size=3, activation=nn.GELU)
        )
        self.detail_proj = ConvBNAct(detail_channels, mid_channels, kernel_size=1, activation=nn.GELU)
        self.semantic_proj = nn.Sequential(
            ConvBNAct(1, mid_channels, kernel_size=3, activation=nn.GELU),
            ConvBNAct(mid_channels, mid_channels, kernel_size=3, activation=nn.GELU)
        )
        self.shallow_boundary = FixedBoundaryExtractor(mid_channels)
        self.detail_boundary = FixedBoundaryExtractor(mid_channels)
        self.boundary_fuse = nn.Sequential(
            ConvBNAct(mid_channels * 5, detail_channels, kernel_size=3, activation=nn.GELU),
            ConvBNAct(detail_channels, detail_channels, kernel_size=3, groups=detail_channels, activation=nn.GELU),
            nn.Conv2d(detail_channels, detail_channels, kernel_size=1, bias=False),
            nn.BatchNorm2d(detail_channels)
        )
        self.boundary_gate = nn.Sequential(
            nn.Conv2d(detail_channels, mid_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(mid_channels),
            nn.GELU(),
            nn.Conv2d(mid_channels, 1, kernel_size=1),
            nn.Sigmoid()
        )
        self.detail_logit_proj = nn.Conv2d(detail_channels, n_classes, kernel_size=1, bias=False)
        self.edge_head = nn.Sequential(
            ConvBNAct(detail_channels * 2 + 1, 128, kernel_size=3, activation=nn.GELU),
            ConvBNAct(128, 64, kernel_size=3, activation=nn.GELU),
            nn.Conv2d(64, 1, kernel_size=1)
        )

    @staticmethod
    def _seg_boundary_map(seg_logits):
        probs = F.softmax(seg_logits, dim=1)
        c = probs.shape[1]
        kernel_x = torch.tensor(
            [[-1.0, 0.0, 1.0], [-2.0, 0.0, 2.0], [-1.0, 0.0, 1.0]],
            dtype=probs.dtype,
            device=probs.device
        ).view(1, 1, 3, 3).repeat(c, 1, 1, 1)
        kernel_y = torch.tensor(
            [[-1.0, -2.0, -1.0], [0.0, 0.0, 0.0], [1.0, 2.0, 1.0]],
            dtype=probs.dtype,
            device=probs.device
        ).view(1, 1, 3, 3).repeat(c, 1, 1, 1)
        grad_x = F.conv2d(probs, kernel_x, padding=1, groups=c)
        grad_y = F.conv2d(probs, kernel_y, padding=1, groups=c)
        boundary_map = (torch.abs(grad_x) + torch.abs(grad_y)).mean(dim=1, keepdim=True)
        normalizer = boundary_map.flatten(1).amax(dim=1).view(-1, 1, 1, 1).clamp_min(1e-6)
        return boundary_map / normalizer

    def forward(self, detail_feat, shallow_feat, guide_feat, seg_logits):
        shallow_feat = self.shallow_proj(shallow_feat)
        guide_feat = self.guide_proj(guide_feat)
        if shallow_feat.shape[2:] != detail_feat.shape[2:]:
            shallow_feat = F.interpolate(shallow_feat, size=detail_feat.shape[2:], mode="bilinear", align_corners=False)
        if guide_feat.shape[2:] != detail_feat.shape[2:]:
            guide_feat = F.interpolate(guide_feat, size=detail_feat.shape[2:], mode="bilinear", align_corners=False)

        detail_context = self.detail_proj(detail_feat)
        semantic_boundary = self._seg_boundary_map(seg_logits.detach())
        if semantic_boundary.shape[2:] != detail_feat.shape[2:]:
            semantic_boundary = F.interpolate(semantic_boundary, size=detail_feat.shape[2:], mode="bilinear", align_corners=False)
        semantic_feat = self.semantic_proj(semantic_boundary)
        shallow_boundary = self.shallow_boundary(shallow_feat) * (0.2 + semantic_boundary)
        detail_boundary = self.detail_boundary(detail_context) * (0.4 + semantic_boundary)
        guide_feat = guide_feat * (0.3 + semantic_boundary)
        boundary_feat = self.boundary_fuse(torch.cat([
            detail_context,
            shallow_boundary,
            detail_boundary,
            guide_feat,
            semantic_feat
        ], dim=1))
        boundary_gate = self.boundary_gate(boundary_feat)
        detail_feat = detail_feat + boundary_feat * (0.15 + 0.85 * boundary_gate) * (0.25 + semantic_boundary)
        detail_logits = self.detail_logit_proj(detail_feat)
        semantic_logit = torch.logit(semantic_boundary.clamp(1e-4, 1.0 - 1e-4))
        edge_residual = self.edge_head(torch.cat([detail_feat, boundary_feat, semantic_boundary], dim=1))
        edge_logits = semantic_logit + 0.35 * edge_residual
        return detail_feat, edge_logits, detail_logits, boundary_feat


class StripPooling(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.pool1 = nn.AdaptiveAvgPool2d((1, None))
        self.pool2 = nn.AdaptiveAvgPool2d((None, 1))
        self.conv1 = nn.Conv2d(in_channels, out_channels, 1, bias=False)
        self.conv2 = nn.Conv2d(in_channels, out_channels, 1, bias=False)
        self.fusion = nn.Sequential(
            nn.Conv2d(out_channels, out_channels, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.Sigmoid()
        )

    def forward(self, x):
        _, _, h, w = x.size()
        x1 = F.interpolate(self.conv1(self.pool1(x)), size=(h, w), mode="bilinear", align_corners=False)
        x2 = F.interpolate(self.conv2(self.pool2(x)), size=(h, w), mode="bilinear", align_corners=False)
        return x * self.fusion(x1 + x2)


class SP_ASPP(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.branch1 = nn.Conv2d(in_channels, out_channels, 1, bias=False)
        self.branch2 = nn.Conv2d(in_channels, out_channels, 3, padding=6, dilation=6, bias=False)
        self.branch3 = nn.Conv2d(in_channels, out_channels, 3, padding=12, dilation=12, bias=False)
        self.strip_pool = StripPooling(in_channels, out_channels)
        self.project = nn.Sequential(
            nn.Conv2d(out_channels * 4, out_channels, 1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        feat1 = self.branch1(x)
        feat2 = self.branch2(x)
        feat3 = self.branch3(x)
        feat_strip = self.strip_pool(x)
        return self.project(torch.cat([feat1, feat2, feat3, feat_strip], dim=1))


class MultiScaleDeformableQueryAttention(nn.Module):
    def __init__(self, dim, num_levels, num_points=4):
        super().__init__()
        self.dim = dim
        self.num_levels = num_levels
        self.num_points = num_points
        self.level_embed = nn.Parameter(torch.zeros(num_levels, dim))
        self.reference_mlp = nn.Sequential(
            nn.Linear(dim, dim),
            nn.GELU(),
            nn.Linear(dim, num_levels * 2)
        )
        self.offset_mlp = nn.Sequential(
            nn.Linear(dim, dim),
            nn.GELU(),
            nn.Linear(dim, num_levels * num_points * 2)
        )
        self.weight_mlp = nn.Linear(dim, num_levels * num_points)
        self.out_proj = nn.Linear(dim, dim)
        nn.init.normal_(self.level_embed, std=0.02)
        nn.init.zeros_(self.offset_mlp[-1].weight)
        nn.init.zeros_(self.offset_mlp[-1].bias)

    @staticmethod
    def _sine_position(dim, h, w, device, dtype):
        quarter_dim = max(dim // 4, 1)
        y = torch.linspace(0.0, 1.0, h, device=device, dtype=dtype)
        x = torch.linspace(0.0, 1.0, w, device=device, dtype=dtype)
        yy, xx = torch.meshgrid(y, x, indexing="ij")
        omega = torch.arange(quarter_dim, device=device, dtype=dtype)
        omega = 1.0 / (10000 ** (omega / max(quarter_dim - 1, 1)))
        pos = torch.cat([
            torch.sin(xx[..., None] * omega),
            torch.cos(xx[..., None] * omega),
            torch.sin(yy[..., None] * omega),
            torch.cos(yy[..., None] * omega),
        ], dim=-1)
        if pos.shape[-1] < dim:
            pos = F.pad(pos, (0, dim - pos.shape[-1]))
        elif pos.shape[-1] > dim:
            pos = pos[..., :dim]
        return pos.permute(2, 0, 1).unsqueeze(0)

    def forward(self, queries, features):
        b, q, c = queries.shape
        references = self.reference_mlp(queries).view(b, q, self.num_levels, 1, 2).sigmoid()
        offsets = self.offset_mlp(queries).view(b, q, self.num_levels, self.num_points, 2)
        offsets = 0.5 * torch.tanh(offsets)
        weights = self.weight_mlp(queries).view(b, q, self.num_levels, self.num_points)
        weights = torch.softmax(weights.flatten(2), dim=-1).view(b, q, self.num_levels, self.num_points)

        sampled_per_level = []
        for level, feat in enumerate(features):
            _, _, h, w = feat.shape
            level_bias = self.level_embed[level].view(1, c, 1, 1).to(dtype=feat.dtype, device=feat.device)
            pos = self._sine_position(c, h, w, feat.device, feat.dtype)
            feat = feat + level_bias + pos
            coords = (references[:, :, level] + offsets[:, :, level]).clamp(0.0, 1.0)
            grid = coords.mul(2.0).sub(1.0).view(b, q * self.num_points, 1, 2)
            sampled = F.grid_sample(feat, grid, mode="bilinear", padding_mode="border", align_corners=False)
            sampled = sampled.squeeze(-1).transpose(1, 2).view(b, q, self.num_points, c)
            sampled_per_level.append(sampled)

        sampled = torch.stack(sampled_per_level, dim=2)
        context = (sampled * weights.unsqueeze(-1)).sum(dim=(2, 3))
        return self.out_proj(context)


class QueryFusionDecoder(nn.Module):
    def __init__(self, in_channels_list, n_classes, decoder_channels=256, query_heads=8, ppm_scales=(1, 2, 3, 6)):
        super().__init__()
        self.n_classes = n_classes
        self.decoder_channels = decoder_channels
        self.lateral_convs = nn.ModuleList([
            ConvBNAct(in_channels, decoder_channels, kernel_size=1)
            for in_channels in in_channels_list
        ])
        self.ppm_stages = nn.ModuleList([
            nn.Sequential(
                nn.AdaptiveAvgPool2d(scale),
                ConvGNAct(decoder_channels, decoder_channels, kernel_size=1)
            )
            for scale in ppm_scales
        ])
        self.ppm_bottleneck = ConvBNAct(
            decoder_channels * (len(ppm_scales) + 1),
            decoder_channels,
            kernel_size=3
        )
        self.fpn_blocks = nn.ModuleList([
            ConvBNAct(decoder_channels, decoder_channels, kernel_size=3)
            for _ in range(max(len(in_channels_list) - 1, 0))
        ])
        self.fusion = ConvBNAct(decoder_channels * len(in_channels_list), decoder_channels, kernel_size=1)
        self.detail_refine = nn.Sequential(
            ConvBNAct(decoder_channels, decoder_channels, kernel_size=3, groups=decoder_channels, activation=nn.GELU),
            ConvBNAct(decoder_channels, decoder_channels, kernel_size=1, activation=nn.GELU)
        )
        self.memory_projs = nn.ModuleList([
            ConvBNAct(decoder_channels, decoder_channels, kernel_size=1)
            for _ in in_channels_list
        ])
        self.query_embed = nn.Embedding(n_classes, decoder_channels)
        self.dynamic_query_mlp = nn.Sequential(
            nn.Linear(decoder_channels, decoder_channels * 2),
            nn.GELU(),
            nn.Linear(decoder_channels * 2, n_classes * decoder_channels)
        )
        self.query_norm = nn.LayerNorm(decoder_channels)
        self.multiscale_query_attn = MultiScaleDeformableQueryAttention(
            decoder_channels,
            num_levels=len(in_channels_list),
            num_points=4
        )
        self.query_ffn = nn.Sequential(
            nn.Linear(decoder_channels, decoder_channels * 4),
            nn.GELU(),
            nn.Linear(decoder_channels * 4, decoder_channels)
        )
        self.mask_proj = nn.Conv2d(decoder_channels, decoder_channels, kernel_size=1, bias=False)
        self.semantic_stream = nn.Sequential(
            ConvBNAct(decoder_channels, decoder_channels, kernel_size=3, activation=nn.GELU),
            ConvBNAct(decoder_channels, decoder_channels, kernel_size=1, activation=nn.GELU),
        )
        self.geometry_stream = nn.Sequential(
            ConvBNAct(decoder_channels, decoder_channels, kernel_size=3, groups=decoder_channels, activation=nn.GELU),
            ConvBNAct(decoder_channels, decoder_channels, kernel_size=1, activation=nn.GELU),
        )
        self.classifier = nn.Conv2d(decoder_channels, n_classes, kernel_size=1)
        self.geometry_classifier = nn.Conv2d(decoder_channels, n_classes, kernel_size=1)
        self.reconstruction_gate = nn.Sequential(
            nn.Conv2d(decoder_channels * 2, decoder_channels, kernel_size=1, bias=False),
            nn.BatchNorm2d(decoder_channels),
            nn.GELU(),
            nn.Conv2d(decoder_channels, n_classes, kernel_size=1),
            nn.Sigmoid()
        )
        self.aux_classifier = nn.Conv2d(decoder_channels, n_classes, kernel_size=1)
        self.query_scale = nn.Parameter(torch.tensor(0.05))
        self.geometry_scale = nn.Parameter(torch.tensor(0.1))

    def _ppm_forward(self, x):
        ppm_outs = [x]
        for stage in self.ppm_stages:
            pooled = stage(x)
            pooled = F.interpolate(pooled, size=x.shape[2:], mode="bilinear", align_corners=False)
            ppm_outs.append(pooled)
        return self.ppm_bottleneck(torch.cat(ppm_outs, dim=1))

    def forward(self, features):
        laterals = [proj(feat) for proj, feat in zip(self.lateral_convs, features)]
        laterals[-1] = self._ppm_forward(laterals[-1])

        for idx in range(len(laterals) - 2, -1, -1):
            laterals[idx] = laterals[idx] + F.interpolate(
                laterals[idx + 1],
                size=laterals[idx].shape[2:],
                mode="bilinear",
                align_corners=False
            )
            laterals[idx] = self.fpn_blocks[idx](laterals[idx])

        target_size = laterals[0].shape[2:]
        pyramid = torch.cat([
            feat if feat.shape[2:] == target_size
            else F.interpolate(feat, size=target_size, mode="bilinear", align_corners=False)
            for feat in laterals
        ], dim=1)
        detail_feat = self.detail_refine(self.fusion(pyramid))
        semantic_feat = self.semantic_stream(detail_feat)
        geometry_feat = self.geometry_stream(detail_feat)

        scene_vector = F.adaptive_avg_pool2d(semantic_feat + geometry_feat, 1).flatten(1)
        global_token = scene_vector.unsqueeze(1)
        dynamic_queries = self.dynamic_query_mlp(scene_vector).view(
            detail_feat.size(0),
            self.n_classes,
            self.decoder_channels,
        )
        static_queries = self.query_embed.weight.unsqueeze(0).expand(detail_feat.size(0), -1, -1)
        queries = static_queries + dynamic_queries + global_token

        memory_features = [proj(feat) for proj, feat in zip(self.memory_projs, laterals)]
        attn_out = self.multiscale_query_attn(self.query_norm(queries), memory_features)
        queries = queries + attn_out
        queries = queries + self.query_ffn(self.query_norm(queries))

        pixel_feat = F.normalize(self.mask_proj(semantic_feat), dim=1)
        query_feat = F.normalize(queries, dim=-1)
        mask_logits = torch.einsum("bqc,bchw->bqhw", query_feat, pixel_feat)

        semantic_logits = self.classifier(semantic_feat) + self.query_scale * mask_logits
        geometry_logits = self.geometry_classifier(geometry_feat)
        geometry_gate = self.reconstruction_gate(torch.cat([semantic_feat, geometry_feat], dim=1))
        seg_logits = semantic_logits * (1.0 + self.geometry_scale * geometry_gate) + self.geometry_scale * geometry_logits
        aux_source = laterals[1] if len(laterals) > 1 else laterals[0]
        aux_logits = self.aux_classifier(aux_source)
        return seg_logits, detail_feat, aux_logits


class PFNet(nn.Module):
    def __init__(self, n_classes=14, pretrained_path=None, return_aux=False, encoder_name="mit_b2", safe_mode=False):
        super().__init__()
        self.return_aux = return_aux
        self.safe_mode = safe_mode

        encoder_weights = "imagenet" if pretrained_path == "imagenet" else None
        self.rgb_encoder = smp.encoders.get_encoder(encoder_name, in_channels=3, weights=encoder_weights)
        self.hd_encoder = smp.encoders.get_encoder(encoder_name, in_channels=2, weights=encoder_weights)
        self.hd_enhancer = HDDifferentialEnhancer(in_channels=2)

        encoder_channels = self.rgb_encoder.out_channels
        self.valid_indices = [i for i, ch in enumerate(encoder_channels) if ch > 0 and i > 0]
        self.cmfr = CMFR_Module(channels=encoder_channels[2])
        self.angle_grad = AngleGradientExtractor()
        self.side_guide = AngleSideGuidePyramid([encoder_channels[i] for i in self.valid_indices])
        self.hd_structure_filters = nn.ModuleDict({
            str(i): HDStructureFilter(self.hd_encoder.out_channels[i])
            for i in self.valid_indices[:2]
        })

        if pretrained_path and pretrained_path != "imagenet":
            if not os.path.exists(pretrained_path):
                raise FileNotFoundError(f"Pretrained encoder checkpoint not found: {pretrained_path}")
            state_dict = torch.load(pretrained_path, map_location="cpu")
            state_dict = state_dict.get("state_dict", state_dict)
            torch.nn.Module.load_state_dict(self.rgb_encoder, state_dict, strict=False)
            target_key_w = "patch_embed1.proj.weight"
            if target_key_w in state_dict:
                weight_2ch = state_dict[target_key_w].mean(dim=1, keepdim=True).repeat(1, 2, 1, 1) * 1.5
                new_state_dict = state_dict.copy()
                new_state_dict[target_key_w] = weight_2ch
                torch.nn.Module.load_state_dict(self.hd_encoder, new_state_dict, strict=False)
            if _is_main_process():
                print(f"Loaded pretrained encoder weights from {pretrained_path}")

        self.fusion_layers = nn.ModuleList()
        self.inter_layer_refiners = nn.ModuleList()

        hd_channels = self.hd_encoder.out_channels

        num_scales = len(self.valid_indices)
        for order, i in enumerate(self.valid_indices):
            ch = encoder_channels[i]
            self.fusion_layers.append(UnifiedGeometryFusion(
                rgb_channels=ch,
                hd_channels=hd_channels[i],
                aux_channels=ch,
                out_channels=ch,
                scale_index=order,
                num_scales=num_scales,
                safe_mode=safe_mode,
            ))

        for shallow_idx, deep_idx in zip(self.valid_indices[:-1], self.valid_indices[1:]):
            self.inter_layer_refiners.append(
                CrossScaleFlowAttention(encoder_channels[shallow_idx], encoder_channels[deep_idx])
            )

        deepest_ch = encoder_channels[-1]
        self.aspp = SP_ASPP(deepest_ch, deepest_ch)

        decoder_channels = 256
        decoder_in_channels = [encoder_channels[i] for i in self.valid_indices]
        self.decoder = QueryFusionDecoder(
            in_channels_list=decoder_in_channels,
            n_classes=n_classes,
            decoder_channels=decoder_channels
        )
        self.detail_edge_head = DetailEnhancedEdgeHead(
            detail_channels=decoder_channels,
            shallow_channels=decoder_in_channels[0],
            guide_channels=decoder_in_channels[0],
            n_classes=n_classes
        )
        self.rgb_high_freq_guide = FixedBoundaryExtractor(decoder_channels)

        self.gated_edge = GatedEdgeInteraction(n_classes, boundary_channels=decoder_channels)
        self.carafe_seg = LightweightUpsampleRefine(in_channels=n_classes, out_channels=n_classes, scale_factor=4)
        self.carafe_edge = LightweightUpsampleRefine(in_channels=1, out_channels=1, scale_factor=4)

    def forward(self, rgb, hha):
        hd = hha[:, 0:2, :, :]
        angle_grad = self.angle_grad(hha)
        hd = self.hd_enhancer(hd)

        feats_rgb = self.rgb_encoder(rgb)
        feats_hd = self.hd_encoder(hd)
        side_guides = self.side_guide(
            angle_grad,
            [feats_rgb[i].shape[2:] for i in self.valid_indices]
        )
        for guide_idx, feat_idx in enumerate(self.valid_indices[:2]):
            feats_hd[feat_idx] = self.hd_structure_filters[str(feat_idx)](feats_hd[feat_idx], side_guides[guide_idx])
        feats_rgb[2] = self.cmfr(feats_rgb[2], feats_hd[2])

        fused_list = []
        fusion_ptr = 0
        for i in range(len(feats_rgb)):
            if i in self.valid_indices:
                fused_layer = self.fusion_layers[fusion_ptr](feats_rgb[i], feats_hd[i], side_guides[fusion_ptr])
                fused_list.append(fused_layer)
                fusion_ptr += 1

        fused_list[-1] = self.aspp(fused_list[-1])
        for idx in range(len(fused_list) - 2, -1, -1):
            fused_list[idx] = self.inter_layer_refiners[idx](fused_list[idx], fused_list[idx + 1])

        seg_base, detail_feat, aux_seg_base = self.decoder([f.contiguous() for f in fused_list])
        detail_feat, edge_feat, detail_logits, boundary_feat = self.detail_edge_head(
            detail_feat,
            fused_list[0],
            side_guides[0],
            seg_base
        )
        detail_logits = F.interpolate(detail_logits, size=seg_base.shape[2:], mode="bilinear", align_corners=False)
        edge_feat_120 = F.interpolate(edge_feat, size=seg_base.shape[2:], mode="bilinear", align_corners=False)
        edge_feat_120 = torch.nan_to_num(edge_feat_120, nan=0.0, posinf=30.0, neginf=-30.0).clamp(-30.0, 30.0)
        edge_prob_120 = torch.sigmoid(edge_feat_120)
        seg_base = seg_base + 0.3 * detail_logits * (1.0 + edge_prob_120)
        if boundary_feat.shape[2:] != seg_base.shape[2:]:
            boundary_feat = F.interpolate(boundary_feat, size=seg_base.shape[2:], mode="bilinear", align_corners=False)
        rgb_high_freq = self.rgb_high_freq_guide(feats_rgb[self.valid_indices[0]])
        if rgb_high_freq.shape[2:] != boundary_feat.shape[2:]:
            rgb_high_freq = F.interpolate(rgb_high_freq, size=boundary_feat.shape[2:], mode="bilinear", align_corners=False)
        boundary_feat = boundary_feat + 0.1 * rgb_high_freq

        seg_refined = self.gated_edge(seg_base, edge_prob_120, boundary_feat).contiguous()
        seg_logits = self.carafe_seg(seg_refined)
        edge_logits = self.carafe_edge(edge_feat_120.contiguous())

        if self.return_aux:
            aux_seg_logits = F.interpolate(aux_seg_base, size=rgb.shape[2:], mode="bilinear", align_corners=False)
            return seg_logits, edge_logits, aux_seg_logits
        return seg_logits
