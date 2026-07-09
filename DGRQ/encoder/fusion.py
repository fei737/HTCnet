import math

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.checkpoint as cp

from GSA import Decomposed_GSA, GeoPriorGen

from .common import ConvBNAct, LearnableBlend, LearnableGate


def projection_block(in_channels, out_channels):
    if in_channels == out_channels:
        return nn.Identity()
    return nn.Sequential(
        nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=False),
        nn.BatchNorm2d(out_channels),
        nn.ReLU(inplace=True),
    )


def warp_with_offsets(x, offsets):
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
    offsets = torch.stack([offsets[:, 0] / norm_x, offsets[:, 1] / norm_y], dim=-1)
    return F.grid_sample(
        x,
        base_grid + offsets,
        mode="bilinear",
        padding_mode="border",
        align_corners=True,
    )


class GeometryAwareContextAttention(nn.Module):
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


class AngularGuidedGlobalAttention(nn.Module):
    def __init__(self, dim, num_heads=4, layer_init_values=1e-5):
        super().__init__()
        self.norm = nn.LayerNorm(dim)
        self.geo_prior = GeoPriorGen(embed_dim=dim, num_heads=num_heads, initial_value=3, heads_range=2)
        self.gsa = Decomposed_GSA(embed_dim=dim, num_heads=num_heads, value_factor=1)
        self.layer_scale = nn.Parameter(layer_init_values * torch.ones(1, dim, 1, 1))

    def forward(self, x, guide_feat):
        _, c, h, w = x.shape
        x_safe = torch.nan_to_num(x, nan=0.0, posinf=1e4, neginf=-1e4)
        guide_prior = guide_feat if guide_feat.shape[1] == 1 else guide_feat.mean(dim=1, keepdim=True)
        guide_prior = torch.nan_to_num(guide_prior.float(), nan=0.0, posinf=1e4, neginf=-1e4)
        rel_pos = self.geo_prior((h, w), guide_prior, split_or_not=True)
        x_bhwc = x_safe.permute(0, 2, 3, 1).contiguous()
        x_norm = F.layer_norm(
            x_bhwc.float(),
            (c,),
            self.norm.weight.float(),
            self.norm.bias.float(),
            self.norm.eps,
        ).to(dtype=x.dtype)
        gsa_out = self.gsa(x_norm, rel_pos).permute(0, 3, 1, 2).contiguous()
        gsa_out = torch.nan_to_num(gsa_out, nan=0.0, posinf=1e4, neginf=-1e4)
        return x + self.layer_scale * gsa_out


class LocalSemanticFusionBlock(nn.Module):
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
        local_scale_init=1e-4,
        semantic_scale_init=1e-4,
        routing_mode="legacy",
        reliability_strength=0.30,
        fusion_branch_mode="both",
        use_grad_checkpoint=False,
    ):
        super().__init__()
        self.scale_index = scale_index
        self.num_scales = max(num_scales, 1)
        self.max_attention_tokens = max_attention_tokens
        self.use_grad_checkpoint = use_grad_checkpoint
        self.routing_mode = routing_mode
        self.reliability_strength = float(reliability_strength)
        if fusion_branch_mode not in {"both", "local", "semantic", "rgb"}:
            raise ValueError(f"Unknown fusion_branch_mode: {fusion_branch_mode!r}")
        self.fusion_branch_mode = fusion_branch_mode
        self.rgb_proj = projection_block(rgb_channels, out_channels)
        self.hd_proj = projection_block(hd_channels, out_channels)
        self.aux_proj = projection_block(aux_channels, out_channels)
        mid_channels = max(out_channels // 2, 16)
        self.hd_confidence = nn.Sequential(
            nn.Conv2d(out_channels * 3, mid_channels, kernel_size=1, bias=False),
            nn.BatchNorm2d(mid_channels),
            nn.GELU(),
            nn.Conv2d(mid_channels, mid_channels, kernel_size=3, padding=1, groups=mid_channels, bias=False),
            nn.BatchNorm2d(mid_channels),
            nn.GELU(),
            nn.Conv2d(mid_channels, 1, kernel_size=1),
            nn.Sigmoid(),
        )

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
            nn.BatchNorm2d(out_channels),
        )

        attn_heads = 8 if out_channels % 8 == 0 else 4 if out_channels % 4 == 0 else 1
        self.semantic_attn = GeometryAwareContextAttention(dim=out_channels, num_heads=attn_heads)
        self.semantic_post = nn.Sequential(
            nn.Conv2d(out_channels * 2, out_channels, kernel_size=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.GELU(),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
        )
        self.deep_gsa = None
        if not safe_mode and scale_index >= max(num_scales - 2, 1):
            self.deep_gsa = AngularGuidedGlobalAttention(dim=out_channels, num_heads=attn_heads)

        self.scale_embed = nn.Parameter(torch.zeros(1, out_channels, 1, 1))
        self.route_gate = nn.Sequential(
            nn.Conv2d(out_channels * 4 + 1, mid_channels, kernel_size=1, bias=False),
            nn.GELU(),
            nn.Conv2d(mid_channels, 2, kernel_size=1),
        )
        self.reliability_router = None
        if routing_mode == "rcfr":
            self.reliability_router = nn.Sequential(
                nn.Conv2d(out_channels * 3 + 3, mid_channels, kernel_size=1, bias=False),
                nn.GELU(),
                nn.Conv2d(mid_channels, 2, kernel_size=1),
                nn.Sigmoid(),
            )
        with torch.no_grad():
            depth_ratio = scale_index / float(max(num_scales - 1, 1))
            self.route_gate[-1].bias.copy_(torch.tensor([1.0 - depth_ratio, depth_ratio]))

        self.local_scale = nn.Parameter(torch.ones(1, out_channels, 1, 1) * float(local_scale_init))
        self.semantic_scale = nn.Parameter(torch.ones(1, out_channels, 1, 1) * float(semantic_scale_init))
        self.delta_reliability_scale = nn.Parameter(torch.tensor(0.0))

        # Learnable replacements for the fixed confidence/reliability affines.
        self.sft_gamma_scale = nn.Parameter(torch.tensor(0.25))
        self.sft_beta_scale = nn.Parameter(torch.tensor(0.25))
        self.hd_conf_gate = LearnableGate(init_lo=0.5, init_span=0.5)
        self.rel_blend = LearnableBlend(2, ctx_channels=2, init_bias=(0.65, 0.35))
        self.rel_conf_gate = LearnableGate(init_lo=0.5, init_span=0.5)
        self.aux_conf_gate = LearnableGate(init_lo=0.25, init_span=0.75)
        self.semantic_rel_gate = LearnableGate(init_lo=0.25, init_span=0.75)
        # reliability_support: base + span * confidence_mean.
        self.rel_support_gate = LearnableGate(init_lo=0.35, init_span=0.45 + self.reliability_strength)
        self.local_router_gain = nn.Parameter(torch.tensor(0.30))
        self.semantic_router_gain = nn.Parameter(torch.tensor(0.30))

    def _local_branch(self, rgb_feat, hd_feat, aux_feat):
        condition = self.condition_proj(torch.cat([hd_feat, aux_feat], dim=1))
        rgb_aligned = warp_with_offsets(rgb_feat, self.offset_head(condition))
        gamma = self.sft_gamma_scale * torch.tanh(self.sft_gamma(condition))
        beta = self.sft_beta_scale * torch.tanh(self.sft_beta(condition))
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

        def _attn(rgb_a, hd_a, aux_a):
            feat, _ = self.semantic_attn(rgb_a, hd_a, aux_a)
            if self.deep_gsa is not None:
                feat = self.deep_gsa(feat, aux_a.mean(dim=1, keepdim=True))
            return feat

        if self.use_grad_checkpoint and self.training and torch.is_grad_enabled():
            attn_feat = cp.checkpoint(_attn, rgb_attn, hd_attn, aux_attn, use_reentrant=False)
        else:
            attn_feat = _attn(rgb_attn, hd_attn, aux_attn)
        if attn_feat.shape[2:] != (h, w):
            attn_feat = F.interpolate(attn_feat, size=(h, w), mode="bilinear", align_corners=False)
        fused = self.semantic_post(torch.cat([attn_feat, aux_feat], dim=1))
        return rgb_feat + self.semantic_scale * fused

    def forward(self, rgb_feat, hd_feat, aux_feat, geometry_reliability=None):
        rgb_feat = self.rgb_proj(rgb_feat)
        hd_feat = self.hd_proj(hd_feat)
        aux_feat = self.aux_proj(aux_feat)
        hd_confidence = self.hd_conf_gate(self.hd_confidence(torch.cat([rgb_feat, hd_feat, aux_feat], dim=1)))
        rel = None
        if geometry_reliability is not None:
            rel = F.interpolate(
                geometry_reliability,
                size=hd_confidence.shape[2:],
                mode="bilinear",
                align_corners=False,
            ).clamp(0.0, 1.0)
            if self.routing_mode == "rcfr":
                hd_confidence = self.rel_blend(
                    [hd_confidence, rel],
                    torch.cat([hd_confidence, rel], dim=1),
                ).clamp(0.0, 1.0)
            else:
                hd_confidence = hd_confidence * self.rel_conf_gate(rel)
        hd_feat = hd_feat * hd_confidence
        aux_feat = aux_feat * self.aux_conf_gate(hd_confidence)

        if self.fusion_branch_mode == "rgb":
            return rgb_feat

        local_feat = self._local_branch(rgb_feat, hd_feat, aux_feat)
        semantic_feat = self._semantic_branch(rgb_feat, hd_feat, aux_feat)

        route_context = torch.cat([
            F.adaptive_avg_pool2d(rgb_feat, 1),
            F.adaptive_avg_pool2d(hd_feat, 1),
            F.adaptive_avg_pool2d(aux_feat, 1),
            self.scale_embed.expand(rgb_feat.size(0), -1, -1, -1),
            F.adaptive_avg_pool2d(hd_confidence, 1),
        ], dim=1)
        route = torch.softmax(self.route_gate(route_context), dim=1)
        if self.fusion_branch_mode == "local":
            route = torch.cat([torch.ones_like(route[:, 0:1]), torch.zeros_like(route[:, 1:2])], dim=1)
        elif self.fusion_branch_mode == "semantic":
            route = torch.cat([torch.zeros_like(route[:, 0:1]), torch.ones_like(route[:, 1:2])], dim=1)
        local_delta = local_feat - rgb_feat
        semantic_delta = semantic_feat - rgb_feat
        confidence_mean = F.adaptive_avg_pool2d(hd_confidence, 1)
        delta_gain = 1.0 + torch.tanh(self.delta_reliability_scale) * (confidence_mean - 0.5)
        delta_gain = delta_gain.clamp(0.5, 1.5)
        local_delta = local_delta * delta_gain
        semantic_delta = semantic_delta * delta_gain
        if self.reliability_router is not None:
            scale_value = self.scale_index / float(max(self.num_scales - 1, 1))
            scale_map = torch.full(
                (rgb_feat.size(0), 1, 1, 1),
                scale_value,
                device=rgb_feat.device,
                dtype=rgb_feat.dtype,
            )
            reliability_context = torch.cat([
                F.adaptive_avg_pool2d(rgb_feat, 1),
                F.adaptive_avg_pool2d(hd_feat, 1),
                F.adaptive_avg_pool2d(aux_feat, 1),
                F.adaptive_avg_pool2d(hd_confidence, 1),
                F.adaptive_avg_pool2d(1.0 - hd_confidence, 1),
                scale_map,
            ], dim=1)
            reliability_gate = self.reliability_router(reliability_context)
            reliability_support = self.rel_support_gate(confidence_mean)
            local_delta = local_delta * (reliability_support + self.local_router_gain * reliability_gate[:, 0:1])
            semantic_delta = semantic_delta * (reliability_support + self.semantic_router_gain * reliability_gate[:, 1:2])
        elif geometry_reliability is not None:
            semantic_reliability = F.adaptive_avg_pool2d(
                F.interpolate(
                    geometry_reliability,
                    size=rgb_feat.shape[2:],
                    mode="bilinear",
                    align_corners=False,
                ).clamp(0.0, 1.0),
                1,
            )
            semantic_delta = semantic_delta * self.semantic_rel_gate(semantic_reliability)
        return rgb_feat + route[:, 0:1] * local_delta + route[:, 1:2] * semantic_delta


class CrossScaleFeatureAligner(nn.Module):
    def __init__(self, shallow_ch, deep_ch):
        super().__init__()
        self.align = nn.Sequential(
            nn.Conv2d(deep_ch, shallow_ch, kernel_size=1, bias=False),
            nn.BatchNorm2d(shallow_ch),
            nn.ReLU(inplace=True),
        )
        self.channel_gate = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(shallow_ch, max(shallow_ch // 2, 1), kernel_size=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(max(shallow_ch // 2, 1), shallow_ch, kernel_size=1),
            nn.Sigmoid(),
        )
        self.spatial_gate = nn.Sequential(
            nn.Conv2d(shallow_ch * 2, max(shallow_ch // 2, 1), kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(max(shallow_ch // 2, 1), 1, kernel_size=1),
            nn.Sigmoid(),
        )

    def forward(self, shallow_feat, deep_feat):
        deep_up = F.interpolate(deep_feat, size=shallow_feat.shape[2:], mode="bilinear", align_corners=False)
        deep_aligned = self.align(deep_up)
        c_gate = self.channel_gate(deep_aligned)
        s_gate = self.spatial_gate(torch.cat([shallow_feat, deep_aligned], dim=1))
        return shallow_feat * s_gate * c_gate + deep_aligned


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
            nn.Sigmoid(),
        )

    def forward(self, x):
        _, _, h, w = x.size()
        x1 = F.interpolate(self.conv1(self.pool1(x)), size=(h, w), mode="bilinear", align_corners=False)
        x2 = F.interpolate(self.conv2(self.pool2(x)), size=(h, w), mode="bilinear", align_corners=False)
        return x * self.fusion(x1 + x2)


class StripPyramidASPP(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.branch1 = nn.Conv2d(in_channels, out_channels, 1, bias=False)
        self.branch2 = nn.Conv2d(in_channels, out_channels, 3, padding=6, dilation=6, bias=False)
        self.branch3 = nn.Conv2d(in_channels, out_channels, 3, padding=12, dilation=12, bias=False)
        self.strip_pool = StripPooling(in_channels, out_channels)
        self.project = nn.Sequential(
            nn.Conv2d(out_channels * 4, out_channels, 1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        feat1 = self.branch1(x)
        feat2 = self.branch2(x)
        feat3 = self.branch3(x)
        feat_strip = self.strip_pool(x)
        return self.project(torch.cat([feat1, feat2, feat3, feat_strip], dim=1))


# Backward-compatible aliases for existing checkpoints, scripts, and notes.
MultiHeadTopologicalAttention = GeometryAwareContextAttention
AngleGuidedGSA = AngularGuidedGlobalAttention
DynamicGatedLocalSemanticRoutingModule = LocalSemanticFusionBlock
CrossScaleDualGatedAlignmentModule = CrossScaleFeatureAligner
SP_ASPP = StripPyramidASPP
