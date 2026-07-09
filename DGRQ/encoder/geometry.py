import torch
import torch.nn as nn
import torch.nn.functional as F

from mamba_layout import GeometryStateSpaceScan

from .common import ConvBNAct, ConvGNAct, make_group_count


class StructuralContrastEnhancer(nn.Module):
    def __init__(self, in_channels=2):
        super().__init__()
        hidden_channels = max(in_channels * 8, 16)
        laplacian = torch.tensor(
            [[-1.0, -1.0, -1.0], [-1.0, 8.0, -1.0], [-1.0, -1.0, -1.0]],
            dtype=torch.float32,
        )
        self.register_buffer("laplacian", laplacian.view(1, 1, 3, 3).repeat(in_channels, 1, 1, 1))
        self.gate_generator = nn.Sequential(
            nn.Conv2d(in_channels * 2, hidden_channels, kernel_size=3, padding=1, bias=False),
            nn.GroupNorm(num_groups=4, num_channels=hidden_channels),
            nn.GELU(),
            nn.Conv2d(hidden_channels, in_channels, kernel_size=1),
            nn.Sigmoid(),
        )
        self.diff_mix = nn.Sequential(
            nn.Conv2d(in_channels, hidden_channels, kernel_size=3, padding=1, bias=False),
            nn.GroupNorm(num_groups=4, num_channels=hidden_channels),
            nn.GELU(),
            nn.Conv2d(hidden_channels, in_channels, kernel_size=1),
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


class AngularBoundaryDescriptor(nn.Module):
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
            nn.Sigmoid(),
        )
        self.refine = nn.Sequential(
            nn.Conv2d(4, 8, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(8),
            nn.GELU(),
            nn.Conv2d(8, 1, kernel_size=1, bias=False),
        )

    def forward(self, hha):
        angle_channel = self.smooth(hha[:, 2:3, :, :])
        kernels = self.fixed_kernels.to(device=hha.device, dtype=hha.dtype)
        angle_edges = torch.abs(F.conv2d(angle_channel, kernels, padding=1))
        angle_edges = angle_edges * self.direction_gate(angle_edges)
        return self.refine(angle_edges)


class AngularGuidancePyramid(nn.Module):
    def __init__(self, out_channels_list):
        super().__init__()
        self.proj_layers = nn.ModuleList([
            nn.Sequential(
                ConvBNAct(1, max(ch // 4, 16), kernel_size=3, activation=nn.GELU),
                ConvBNAct(max(ch // 4, 16), ch, kernel_size=1, activation=nn.GELU),
            )
            for ch in out_channels_list
        ])

    def forward(self, angle_grad, target_sizes):
        guides = []
        for proj, size in zip(self.proj_layers, target_sizes):
            guide = F.interpolate(angle_grad, size=size, mode="bilinear", align_corners=False)
            guides.append(proj(guide))
        return guides


class GeometryReliabilityEstimator(nn.Module):
    def __init__(self, in_channels=3):
        super().__init__()
        hidden_channels = 24
        self.encoder = nn.Sequential(
            ConvBNAct(in_channels + 2, hidden_channels, kernel_size=3, activation=nn.GELU),
            ConvBNAct(hidden_channels, hidden_channels, kernel_size=3, activation=nn.GELU),
            nn.Conv2d(hidden_channels, 1, kernel_size=1),
            nn.Sigmoid(),
        )

    def forward(self, hha):
        local_mean = F.avg_pool2d(hha, kernel_size=3, stride=1, padding=1)
        local_var = F.avg_pool2d((hha - local_mean).pow(2), kernel_size=3, stride=1, padding=1)
        noise_level = local_var.mean(dim=1, keepdim=True).clamp_min(1e-9).sqrt()
        invalid_hint = (hha.abs().mean(dim=1, keepdim=True) < 1e-4).to(dtype=hha.dtype)
        return self.encoder(torch.cat([hha, noise_level, invalid_hint], dim=1))


class GeometryPromptRecovery(nn.Module):
    def __init__(self, in_channels=3, max_residual=0.25):
        super().__init__()
        hidden_channels = 24
        self.max_residual = float(max_residual)
        self.encoder = nn.Sequential(
            ConvBNAct(in_channels + 3, hidden_channels, kernel_size=3, activation=nn.GELU),
            nn.Conv2d(hidden_channels, hidden_channels, kernel_size=3, padding=1, groups=hidden_channels, bias=False),
            nn.BatchNorm2d(hidden_channels),
            nn.GELU(),
            ConvBNAct(hidden_channels, hidden_channels, kernel_size=1, activation=nn.GELU),
            nn.Conv2d(hidden_channels, in_channels, kernel_size=3, padding=1),
        )
        nn.init.zeros_(self.encoder[-1].weight)
        nn.init.zeros_(self.encoder[-1].bias)

    def forward(self, hha, confidence):
        local_mean = F.avg_pool2d(hha, kernel_size=3, stride=1, padding=1)
        local_var = F.avg_pool2d((hha - local_mean).pow(2), kernel_size=3, stride=1, padding=1)
        noise_level = local_var.mean(dim=1, keepdim=True).clamp_min(1e-9).sqrt()
        invalid_hint = (hha.abs().mean(dim=1, keepdim=True) < 1e-4).to(dtype=hha.dtype)
        residual = torch.tanh(self.encoder(torch.cat([hha, confidence, noise_level, invalid_hint], dim=1)))
        correction_gate = (1.0 - confidence).clamp(0.0, 1.0)
        recovered = hha + self.max_residual * correction_gate * residual
        return torch.nan_to_num(recovered, nan=0.0, posinf=3.0, neginf=-3.0).clamp(-3.0, 3.0)


class FixedSobelEdge(nn.Module):
    def __init__(self):
        super().__init__()
        kernel_x = torch.tensor(
            [[-1.0, 0.0, 1.0], [-2.0, 0.0, 2.0], [-1.0, 0.0, 1.0]],
            dtype=torch.float32,
        )
        kernel_y = torch.tensor(
            [[-1.0, -2.0, -1.0], [0.0, 0.0, 0.0], [1.0, 2.0, 1.0]],
            dtype=torch.float32,
        )
        self.register_buffer("kernel_x", kernel_x.view(1, 1, 3, 3))
        self.register_buffer("kernel_y", kernel_y.view(1, 1, 3, 3))

    def forward(self, x):
        if x.shape[1] > 1:
            x = x.mean(dim=1, keepdim=True)
        x = F.avg_pool2d(x, kernel_size=3, stride=1, padding=1)
        kernel_x = self.kernel_x.to(device=x.device, dtype=x.dtype)
        kernel_y = self.kernel_y.to(device=x.device, dtype=x.dtype)
        grad_x = F.conv2d(x, kernel_x, padding=1)
        grad_y = F.conv2d(x, kernel_y, padding=1)
        edge = torch.sqrt(grad_x.pow(2) + grad_y.pow(2) + 1e-6)
        normalizer = edge.flatten(1).amax(dim=1).view(x.shape[0], 1, 1, 1).clamp_min(1e-6)
        return (edge / normalizer).clamp(0.0, 1.0)


class RGBGuidedGeometryRecovery(nn.Module):
    def __init__(self, in_channels=3, max_residual=0.25, rgb_context_channels=8):
        super().__init__()
        hidden_channels = 32
        self.max_residual = float(max_residual)
        self.rgb_edge = FixedSobelEdge()
        self.rgb_context = nn.Sequential(
            ConvBNAct(4, 16, kernel_size=3, activation=nn.GELU),
            nn.Conv2d(16, 16, kernel_size=3, padding=1, groups=16, bias=False),
            nn.BatchNorm2d(16),
            nn.GELU(),
            ConvBNAct(16, rgb_context_channels, kernel_size=1, activation=nn.GELU),
        )
        self.encoder = nn.Sequential(
            ConvBNAct(in_channels + 3 + rgb_context_channels, hidden_channels, kernel_size=3, activation=nn.GELU),
            nn.Conv2d(hidden_channels, hidden_channels, kernel_size=3, padding=1, groups=hidden_channels, bias=False),
            nn.BatchNorm2d(hidden_channels),
            nn.GELU(),
            ConvBNAct(hidden_channels, hidden_channels, kernel_size=1, activation=nn.GELU),
            nn.Conv2d(hidden_channels, in_channels, kernel_size=3, padding=1),
        )
        nn.init.zeros_(self.encoder[-1].weight)
        nn.init.zeros_(self.encoder[-1].bias)

    def forward(self, hha, confidence, rgb=None):
        local_mean = F.avg_pool2d(hha, kernel_size=3, stride=1, padding=1)
        local_var = F.avg_pool2d((hha - local_mean).pow(2), kernel_size=3, stride=1, padding=1)
        noise_level = local_var.mean(dim=1, keepdim=True).clamp_min(1e-9).sqrt()
        invalid_hint = (hha.abs().mean(dim=1, keepdim=True) < 1e-4).to(dtype=hha.dtype)

        if rgb is None:
            rgb = hha.new_zeros(hha.shape[0], 3, hha.shape[2], hha.shape[3])
        elif rgb.shape[2:] != hha.shape[2:]:
            rgb = F.interpolate(rgb, size=hha.shape[2:], mode="bilinear", align_corners=False)
        rgb_edge = self.rgb_edge(rgb)
        rgb_context = self.rgb_context(torch.cat([rgb, rgb_edge], dim=1))

        residual = torch.tanh(
            self.encoder(torch.cat([hha, confidence, noise_level, invalid_hint, rgb_context], dim=1))
        )
        correction_gate = (1.0 - confidence).clamp(0.0, 1.0) * (0.60 + 0.40 * rgb_edge)
        recovered = hha + self.max_residual * correction_gate * residual
        return torch.nan_to_num(recovered, nan=0.0, posinf=3.0, neginf=-3.0).clamp(-3.0, 3.0)


class CrossModalReliabilityEstimator(nn.Module):
    def __init__(self, hidden_channels=24):
        super().__init__()
        self.edge = FixedSobelEdge()
        self.router = nn.Sequential(
            ConvBNAct(6, hidden_channels, kernel_size=3, activation=nn.GELU),
            ConvBNAct(hidden_channels, hidden_channels, kernel_size=3, activation=nn.GELU),
            nn.Conv2d(hidden_channels, 1, kernel_size=1),
        )
        nn.init.zeros_(self.router[-1].weight)
        nn.init.constant_(self.router[-1].bias, 1.0)

    def forward(self, rgb, hha, angle_grad, confidence):
        if rgb.shape[2:] != hha.shape[2:]:
            rgb = F.interpolate(rgb, size=hha.shape[2:], mode="bilinear", align_corners=False)
        confidence = confidence.clamp(0.0, 1.0)
        rgb_edge = self.edge(rgb)
        hha_edge = self.edge(hha)
        angle_edge = angle_grad.abs()
        angle_edge = angle_edge / angle_edge.flatten(1).amax(dim=1).view(angle_edge.shape[0], 1, 1, 1).clamp_min(1e-6)
        angle_edge = angle_edge.clamp(0.0, 1.0)
        edge_gap = torch.abs(rgb_edge - hha_edge)
        structural_agreement = torch.exp(-edge_gap)
        learned_agreement = torch.sigmoid(
            self.router(torch.cat([rgb_edge, hha_edge, angle_edge, edge_gap, confidence, structural_agreement], dim=1))
        )
        consistency = 0.5 * structural_agreement + 0.5 * learned_agreement
        return (0.65 * confidence + 0.35 * consistency).clamp(0.05, 1.0)


class StructuralPromptFilter(nn.Module):
    def __init__(self, channels):
        super().__init__()
        hidden_channels = max(channels // 4, 16)
        self.smooth = nn.AvgPool2d(kernel_size=3, stride=1, padding=1)
        self.low_proj = ConvBNAct(channels, channels, kernel_size=3, activation=nn.GELU)
        self.guide_proj = nn.Sequential(
            ConvBNAct(channels, hidden_channels, kernel_size=1, activation=nn.GELU),
            nn.Conv2d(hidden_channels, 1, kernel_size=3, padding=1),
            nn.Sigmoid(),
        )
        self.texture_gate = nn.Sequential(
            ConvBNAct(channels + 1, hidden_channels, kernel_size=1, activation=nn.GELU),
            nn.Conv2d(hidden_channels, channels, kernel_size=1),
            nn.Sigmoid(),
        )
        self.out_proj = nn.Sequential(
            ConvBNAct(channels, channels, kernel_size=3, activation=nn.GELU),
            nn.Conv2d(channels, channels, kernel_size=1, bias=False),
            nn.BatchNorm2d(channels),
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


class DirectionalEdgePromptRefiner(nn.Module):
    """DEGConv-inspired directional edge refinement for geometry prompts.

    The module uses horizontal and vertical depthwise filters to strengthen
    geometry boundaries, but the output projection is zero-initialized so the
    branch starts as an exact identity and can be safely ablated.
    """

    def __init__(self, channels, hidden_channels=None):
        super().__init__()
        hidden_channels = hidden_channels or min(max(channels, 16), 96)
        self.in_proj = ConvGNAct(channels + 2, hidden_channels, kernel_size=1, activation=nn.GELU, num_groups=8)
        self.horizontal_edge = nn.Conv2d(
            hidden_channels,
            hidden_channels,
            kernel_size=(1, 3),
            padding=(0, 1),
            groups=hidden_channels,
            bias=False,
        )
        self.vertical_edge = nn.Conv2d(
            hidden_channels,
            hidden_channels,
            kernel_size=(3, 1),
            padding=(1, 0),
            groups=hidden_channels,
            bias=False,
        )
        self.direction_gate = nn.Sequential(
            nn.Conv2d(hidden_channels * 2 + 2, hidden_channels, kernel_size=1, bias=False),
            nn.GroupNorm(make_group_count(hidden_channels, max_groups=8), hidden_channels),
            nn.GELU(),
            nn.Conv2d(hidden_channels, 2, kernel_size=1),
        )
        self.out_proj = nn.Sequential(
            nn.Conv2d(hidden_channels * 2, hidden_channels, kernel_size=1, bias=False),
            nn.GroupNorm(make_group_count(hidden_channels, max_groups=8), hidden_channels),
            nn.GELU(),
            nn.Conv2d(hidden_channels, channels, kernel_size=1),
        )
        self.layer_scale = nn.Parameter(torch.ones(1, channels, 1, 1) * 1e-4)
        nn.init.zeros_(self.out_proj[-1].weight)
        nn.init.zeros_(self.out_proj[-1].bias)

    @staticmethod
    def _match_single_channel_map(x, reference):
        if x.shape[2:] != reference.shape[2:]:
            x = F.interpolate(x, size=reference.shape[2:], mode="bilinear", align_corners=False)
        if x.shape[1] != 1:
            x = x.mean(dim=1, keepdim=True)
        return x.to(dtype=reference.dtype)

    def forward(self, boundary_prompt, angle_grad, reliability):
        angle_map = self._match_single_channel_map(angle_grad, boundary_prompt).abs()
        angle_norm = angle_map / angle_map.flatten(1).amax(dim=1).view(angle_map.shape[0], 1, 1, 1).clamp_min(1e-6)
        angle_norm = angle_norm.clamp(0.0, 1.0)

        reliability = self._match_single_channel_map(reliability, boundary_prompt).clamp(0.0, 1.0)
        x = self.in_proj(torch.cat([boundary_prompt, angle_norm, reliability], dim=1))
        h_edge = self.horizontal_edge(x)
        v_edge = self.vertical_edge(x)
        direction_gate = torch.softmax(
            self.direction_gate(torch.cat([h_edge, v_edge, angle_norm, reliability], dim=1)),
            dim=1,
        )
        directional = torch.cat([
            h_edge * direction_gate[:, 0:1],
            v_edge * direction_gate[:, 1:2],
        ], dim=1)
        residual = self.out_proj(directional)
        residual_gate = 0.25 + 0.75 * reliability
        return boundary_prompt + self.layer_scale * residual_gate * residual


class ReliabilityAwareAutocorrelationPromptMixer(nn.Module):
    """AFFN-inspired prompt mixer with safe patch autocorrelation.

    The branch is zero-initialized at the output so enabling it does not disturb
    an existing checkpoint at load time. It learns a bounded residual only where
    the geometry prompt is considered reliable.
    """

    def __init__(self, channels, patch_size=8, hidden_channels=None):
        super().__init__()
        self.patch_size = int(patch_size)
        hidden_channels = hidden_channels or min(max(channels // 2, 16), 128)
        self.in_proj = ConvGNAct(channels, hidden_channels, kernel_size=1, activation=nn.GELU, num_groups=8)
        self.freq_weight = nn.Parameter(
            torch.ones(1, hidden_channels, 1, 1, self.patch_size, self.patch_size // 2 + 1)
        )
        self.corr_scale = nn.Parameter(torch.tensor(0.10))
        self.dwconv = nn.Conv2d(
            hidden_channels,
            hidden_channels,
            kernel_size=3,
            padding=1,
            groups=hidden_channels,
            bias=False,
        )
        self.gate = nn.Sequential(
            nn.Conv2d(hidden_channels * 2 + 1, hidden_channels, kernel_size=1, bias=False),
            nn.GroupNorm(make_group_count(hidden_channels, max_groups=8), hidden_channels),
            nn.GELU(),
            nn.Conv2d(hidden_channels, hidden_channels, kernel_size=1),
            nn.Sigmoid(),
        )
        self.out_proj = nn.Conv2d(hidden_channels, channels, kernel_size=1)
        self.layer_scale = nn.Parameter(torch.ones(1, channels, 1, 1) * 1e-3)
        nn.init.zeros_(self.out_proj.weight)
        nn.init.zeros_(self.out_proj.bias)

    def _patch_autocorrelation(self, x):
        b, c, h, w = x.shape
        patch = self.patch_size
        pad_h = (-h) % patch
        pad_w = (-w) % patch
        if pad_h or pad_w:
            x_pad = F.pad(x, (0, pad_w, 0, pad_h), mode="replicate")
        else:
            x_pad = x

        hp, wp = x_pad.shape[2:]
        patches = x_pad.reshape(b, c, hp // patch, patch, wp // patch, patch)
        patches = patches.permute(0, 1, 2, 4, 3, 5).contiguous()
        patches_f = patches.float()

        spec = torch.fft.rfft2(patches_f, dim=(-2, -1), norm="ortho")
        freq_weight = self.freq_weight.to(device=x.device, dtype=spec.real.dtype)
        spec = spec * freq_weight
        power = spec * torch.conj(spec)
        corr = torch.fft.irfft2(power, s=(patch, patch), dim=(-2, -1), norm="ortho")
        corr = corr - corr.mean(dim=(-2, -1), keepdim=True)
        corr_std = corr.std(dim=(-2, -1), keepdim=True, unbiased=False).clamp_min(1e-5)
        corr = corr / corr_std

        corr = corr.permute(0, 1, 2, 4, 3, 5).contiguous().reshape(b, c, hp, wp)
        return corr[:, :, :h, :w].to(dtype=x.dtype)

    def forward(self, prompt_feat, reliability=None):
        hidden = self.in_proj(prompt_feat)
        corr = self._patch_autocorrelation(hidden)
        if reliability is None:
            reliability = hidden.new_ones(hidden.shape[0], 1, hidden.shape[2], hidden.shape[3])
        elif reliability.shape[2:] != hidden.shape[2:]:
            reliability = F.interpolate(reliability, size=hidden.shape[2:], mode="bilinear", align_corners=False)
        if reliability.shape[1] != 1:
            reliability = reliability.mean(dim=1, keepdim=True)
        reliability = reliability.clamp(0.0, 1.0).to(dtype=hidden.dtype)

        gate = self.gate(torch.cat([hidden, corr, reliability], dim=1))
        mixed = self.dwconv(hidden + torch.tanh(self.corr_scale) * corr)
        residual = self.out_proj(F.gelu(mixed) * gate)
        residual_gate = 0.25 + 0.75 * reliability
        return prompt_feat + self.layer_scale * residual_gate * residual


class FrequencyAwareGeometryPrompt(nn.Module):
    def __init__(
        self,
        out_channels_list,
        layout_mode="ssm",
        layout_state_dim=16,
        prompt_channels=32,
        routing_mode="legacy",
        reliability_strength=0.30,
        use_prompt_autocorr=True,
        use_directional_edge_refine=True,
        autocorr_patch_size=8,
    ):
        super().__init__()
        prompt_channels = int(prompt_channels)
        self.layout_mode = layout_mode
        self.routing_mode = routing_mode
        self.reliability_strength = float(reliability_strength)
        self.use_prompt_autocorr = bool(use_prompt_autocorr)
        self.use_directional_edge_refine = bool(use_directional_edge_refine)
        self.layout_in = ConvGNAct(3, prompt_channels, kernel_size=7, activation=nn.GELU, num_groups=4)
        if layout_mode == "ssm":
            self.layout_aggregator = GeometryStateSpaceScan(prompt_channels, d_state=layout_state_dim)
        elif layout_mode == "conv":
            self.layout_aggregator = nn.Sequential(
                ConvGNAct(prompt_channels, prompt_channels, kernel_size=3, activation=nn.GELU, num_groups=4),
                ConvGNAct(prompt_channels, prompt_channels, kernel_size=3, activation=nn.GELU, num_groups=4),
            )
        elif layout_mode == "avgpool":
            self.layout_aggregator = nn.Sequential(
                nn.AvgPool2d(kernel_size=9, stride=1, padding=4),
                ConvGNAct(prompt_channels, prompt_channels, kernel_size=3, activation=nn.GELU, num_groups=4),
            )
        else:
            raise ValueError(f"Unknown layout_mode: {layout_mode!r} (expected 'ssm', 'conv', or 'avgpool')")
        self.layout_stream = nn.Sequential(self.layout_in, self.layout_aggregator)
        self.boundary_stream = nn.Sequential(
            ConvGNAct(4, prompt_channels, kernel_size=3, activation=nn.GELU, num_groups=4),
            nn.Conv2d(prompt_channels, prompt_channels, kernel_size=3, padding=1, groups=prompt_channels, bias=False),
            nn.GroupNorm(4, prompt_channels),
            nn.GELU(),
            ConvGNAct(prompt_channels, prompt_channels, kernel_size=3, activation=nn.GELU, num_groups=4),
        )
        self.layout_refine = nn.Sequential(
            nn.Conv2d(prompt_channels, prompt_channels, kernel_size=3, padding=1, groups=prompt_channels, bias=False),
            nn.GroupNorm(4, prompt_channels),
            nn.GELU(),
            nn.Conv2d(prompt_channels, prompt_channels, kernel_size=1, bias=False),
            nn.GroupNorm(4, prompt_channels),
            nn.GELU(),
        )
        self.boundary_refine = nn.Sequential(
            nn.Conv2d(prompt_channels, prompt_channels, kernel_size=3, padding=1, groups=prompt_channels, bias=False),
            nn.GroupNorm(4, prompt_channels),
            nn.GELU(),
            nn.Conv2d(prompt_channels, prompt_channels, kernel_size=1, bias=False),
            nn.GroupNorm(4, prompt_channels),
            nn.GELU(),
        )
        self.directional_edge_refine = (
            DirectionalEdgePromptRefiner(prompt_channels) if self.use_directional_edge_refine else None
        )
        self.frequency_router = nn.Sequential(
            nn.Conv2d(prompt_channels * 2, prompt_channels, kernel_size=1, bias=False),
            nn.GroupNorm(4, prompt_channels),
            nn.GELU(),
            nn.Conv2d(prompt_channels, len(out_channels_list), kernel_size=1),
        )
        self.spatial_frequency_router = None
        if routing_mode == "rcfr":
            self.spatial_frequency_router = nn.Sequential(
                ConvGNAct(prompt_channels * 2 + 2, prompt_channels, kernel_size=1, activation=nn.GELU, num_groups=4),
                nn.Conv2d(prompt_channels, len(out_channels_list), kernel_size=1),
            )
        self.proj_layers = nn.ModuleList([
            nn.Sequential(
                ConvGNAct(prompt_channels, max(ch // 4, 16), kernel_size=1, activation=nn.GELU, num_groups=4),
                nn.Conv2d(max(ch // 4, 16), ch, kernel_size=1, bias=False),
                nn.GroupNorm(make_group_count(ch), ch),
            )
            for ch in out_channels_list
        ])
        self.prompt_scales = nn.ParameterList([
            nn.Parameter(torch.ones(1, ch, 1, 1) * 0.1)
            for ch in out_channels_list
        ])
        self.autocorr_layers = nn.ModuleList([
            ReliabilityAwareAutocorrelationPromptMixer(ch, patch_size=autocorr_patch_size)
            for ch in out_channels_list
        ]) if self.use_prompt_autocorr else None

    def forward(self, hd, angle_grad, confidence, target_sizes):
        hd_low = F.avg_pool2d(hd, kernel_size=9, stride=1, padding=4)
        hd_high = hd - hd_low
        confidence = confidence.clamp(0.0, 1.0)
        uncertainty = 1.0 - confidence
        layout_prompt = self.layout_refine(self.layout_stream(torch.cat([hd_low, confidence], dim=1)))
        boundary_prompt = self.boundary_refine(self.boundary_stream(torch.cat([hd_high, angle_grad, confidence], dim=1)))
        if self.directional_edge_refine is not None:
            boundary_prompt = self.directional_edge_refine(boundary_prompt, angle_grad, confidence)
        router_logits = self.frequency_router(torch.cat([layout_prompt, boundary_prompt], dim=1))
        router_logits = F.adaptive_avg_pool2d(router_logits, 1)
        if self.spatial_frequency_router is not None:
            spatial_logits = self.spatial_frequency_router(
                torch.cat([layout_prompt, boundary_prompt, confidence, uncertainty], dim=1)
            )
        else:
            spatial_logits = None

        prompts = []
        num_scales = max(1, len(target_sizes) - 1)
        for idx, (proj, scale, size) in enumerate(zip(self.proj_layers, self.prompt_scales, target_sizes)):
            depth_prior = float(idx) / float(num_scales)
            learned_layout_weight = torch.sigmoid(router_logits[:, idx:idx + 1])
            if self.routing_mode == "rcfr":
                spatial_layout_weight = torch.sigmoid(spatial_logits[:, idx:idx + 1])
                layout_weight = (
                    (0.20 + 0.45 * depth_prior)
                    + 0.15 * learned_layout_weight
                    + 0.20 * spatial_layout_weight
                    + self.reliability_strength * uncertainty
                ).clamp(0.10, 0.90)
            else:
                layout_weight = (0.25 + 0.50 * depth_prior) + 0.25 * learned_layout_weight
            boundary_weight = 1.0 - layout_weight
            prompt = layout_weight * layout_prompt + boundary_weight * boundary_prompt
            prompt_i = F.interpolate(prompt, size=size, mode="bilinear", align_corners=False)
            prompt_i = proj(prompt_i)
            if self.autocorr_layers is not None:
                reliability_i = F.interpolate(confidence, size=size, mode="bilinear", align_corners=False)
                prompt_i = self.autocorr_layers[idx](prompt_i, reliability_i)
            prompts.append(scale * prompt_i)
        return prompts


# Backward-compatible aliases for existing checkpoints, scripts, and notes.
BimodalStructuralDifferentialEnhancer = StructuralContrastEnhancer
AngularGeometryFieldDescriptor = AngularBoundaryDescriptor
MultiScaleAngularGuidancePyramid = AngularGuidancePyramid
ModalUncertaintyEstimationBlock = GeometryReliabilityEstimator
GeometryPromptRecoveryBlock = GeometryPromptRecovery
RGBGuidedGeometryPromptRecoveryBlock = RGBGuidedGeometryRecovery
GeometryAppearanceConsistencyRouting = CrossModalReliabilityEstimator
GuidanceDrivenStructuralTextureFilter = StructuralPromptFilter
DirectionalGeometryEdgePromptRefiner = DirectionalEdgePromptRefiner
ReliabilityAwareAutocorrelationPromptMixerBlock = ReliabilityAwareAutocorrelationPromptMixer
FrequencyDisentangledGeometricPromptField = FrequencyAwareGeometryPrompt
