import torch
import torch.nn as nn
import torch.nn.functional as F

from .layers import ConvBNAct, ConvGNAct, make_group_count


GEOMETRY_ENCODINGS = {
    "unified",
    "independent",
    "factorized_static",
    "factorized_routed",
    "factorized_swapped",
    "factorized_final",
    "factorized_final_soft",
    "factorized_channel_routed",
}


def normalized_hha_invalid_hint(hha, explicit_hint=None):
    """Build a dense invalid-pixel mask for normalized HHA input."""
    if explicit_hint is not None:
        if explicit_hint.shape[2:] != hha.shape[2:]:
            explicit_hint = F.interpolate(explicit_hint.float(), size=hha.shape[2:], mode="nearest")
        if explicit_hint.shape[1] != 1:
            explicit_hint = explicit_hint.amax(dim=1, keepdim=True)
        return explicit_hint.to(device=hha.device, dtype=hha.dtype).clamp(0.0, 1.0)
    neutral_code = hha.abs().amax(dim=1, keepdim=True) < 0.02
    zero_code = (hha + 1.0).abs().amax(dim=1, keepdim=True) < 0.02
    return (neutral_code | zero_code).to(dtype=hha.dtype)


def normalized_hha_angle_to_radians(angle):
    """Map normalized HHA angle codes from [-1, 1] to [0, pi/2]."""
    return (angle.clamp(-1.0, 1.0) + 1.0) * (0.25 * torch.pi)


class GeometryStructureEncoder(nn.Module):
    """Enhance local disparity and height contrast without a second backbone."""

    def __init__(self, in_channels=2):
        super().__init__()
        hidden_channels = max(in_channels * 8, 16)
        laplacian = torch.tensor(
            [[-1.0, -1.0, -1.0], [-1.0, 8.0, -1.0], [-1.0, -1.0, -1.0]],
            dtype=torch.float32,
        )
        self.register_buffer(
            "laplacian",
            laplacian.view(1, 1, 3, 3).repeat(in_channels, 1, 1, 1),
        )
        self.gate = nn.Sequential(
            nn.Conv2d(in_channels * 2, hidden_channels, kernel_size=3, padding=1, bias=False),
            nn.GroupNorm(4, hidden_channels),
            nn.GELU(),
            nn.Conv2d(hidden_channels, in_channels, kernel_size=1),
            nn.Sigmoid(),
        )
        self.projection = nn.Sequential(
            nn.Conv2d(in_channels, hidden_channels, kernel_size=3, padding=1, bias=False),
            nn.GroupNorm(4, hidden_channels),
            nn.GELU(),
            nn.Conv2d(hidden_channels, in_channels, kernel_size=1),
        )
        self.scale = nn.Parameter(torch.tensor(0.25))

    def forward(self, geometry):
        kernel = self.laplacian.to(device=geometry.device, dtype=geometry.dtype)
        contrast = torch.abs(F.conv2d(geometry, kernel, padding=1, groups=geometry.shape[1]))
        normalizer = contrast.flatten(2).amax(dim=2).view(*contrast.shape[:2], 1, 1).clamp_min(1e-6)
        contrast = contrast / normalizer
        gate = self.gate(torch.cat([geometry, contrast], dim=1))
        scale = self.scale.clamp(0.0, 1.0)
        return geometry * (1.0 + scale * gate) + scale * gate * self.projection(contrast)


class HHAEdgeDescriptor(nn.Module):
    """Extract four directional responses from the HHA angle channel."""

    def __init__(self):
        super().__init__()
        self.smooth = nn.AvgPool2d(kernel_size=3, stride=1, padding=1)
        kernels = torch.tensor(
            [
                [[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]],
                [[-2, -1, 0], [-1, 0, 1], [0, 1, 2]],
                [[-1, -2, -1], [0, 0, 0], [1, 2, 1]],
                [[0, 1, 2], [-1, 0, 1], [-2, -1, 0]],
            ],
            dtype=torch.float32,
        )
        self.register_buffer("kernels", kernels.view(4, 1, 3, 3))
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
        angle = self.smooth(hha[:, 2:3])
        responses = torch.abs(F.conv2d(angle, self.kernels.to(dtype=hha.dtype), padding=1))
        return self.refine(responses * self.direction_gate(responses))


class GeometryGuidePyramid(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.projections = nn.ModuleList(
            [
                nn.Sequential(
                    ConvBNAct(1, max(channel // 4, 16), kernel_size=3, activation=nn.GELU),
                    ConvBNAct(max(channel // 4, 16), channel, kernel_size=1, activation=nn.GELU),
                )
                for channel in channels
            ]
        )

    def forward(self, edge_map, target_sizes):
        return [
            projection(F.interpolate(edge_map, size=size, mode="bilinear", align_corners=False))
            for projection, size in zip(self.projections, target_sizes)
        ]


class GeometryReliabilityHead(nn.Module):
    """Estimate spatial HHA reliability and hard-mask known invalid pixels."""

    def __init__(self, in_channels=3):
        super().__init__()
        hidden_channels = 24
        self.encoder = nn.Sequential(
            ConvBNAct(in_channels + 2, hidden_channels, kernel_size=3, activation=nn.GELU),
            ConvBNAct(hidden_channels, hidden_channels, kernel_size=3, activation=nn.GELU),
            nn.Conv2d(hidden_channels, 1, kernel_size=1),
            nn.Sigmoid(),
        )
        nn.init.normal_(self.encoder[-2].weight, mean=0.0, std=1e-3)
        nn.init.constant_(self.encoder[-2].bias, float(torch.logit(torch.tensor(0.90))))

    def forward(self, hha, invalid_hint=None):
        local_mean = F.avg_pool2d(hha, kernel_size=3, stride=1, padding=1)
        local_variance = F.avg_pool2d((hha - local_mean).pow(2), kernel_size=3, stride=1, padding=1)
        noise = local_variance.mean(dim=1, keepdim=True).clamp_min(1e-9).sqrt()
        invalid = normalized_hha_invalid_hint(hha, invalid_hint)
        reliability = self.encoder(torch.cat([hha, noise, invalid], dim=1))
        return reliability * (1.0 - invalid)


class FactorizedGeometryReliabilityHead(nn.Module):
    """Estimate separate layout and boundary reliability maps."""

    def __init__(self):
        super().__init__()
        hidden_channels = 24
        self.encoder = nn.Sequential(
            ConvBNAct(7, hidden_channels, kernel_size=3, activation=nn.GELU),
            ConvBNAct(hidden_channels, hidden_channels, kernel_size=3, activation=nn.GELU),
            nn.Conv2d(hidden_channels, 2, kernel_size=1),
            nn.Sigmoid(),
        )
        nn.init.normal_(self.encoder[-2].weight, mean=0.0, std=1e-3)
        nn.init.constant_(self.encoder[-2].bias, float(torch.logit(torch.tensor(0.90))))

    def forward(self, hha, invalid_hint=None):
        local_mean = F.avg_pool2d(hha, kernel_size=3, stride=1, padding=1)
        local_variance = F.avg_pool2d((hha - local_mean).pow(2), kernel_size=3, stride=1, padding=1)
        channel_noise = local_variance.clamp_min(1e-9).sqrt()
        invalid = normalized_hha_invalid_hint(hha, invalid_hint)
        reliability = self.encoder(torch.cat([hha, channel_noise, invalid], dim=1))
        return reliability * (1.0 - invalid)


class GeometryPromptEncoder(nn.Module):
    """Encode low-frequency layout and high-frequency boundary HHA cues."""

    def __init__(self, out_channels, prompt_channels=32, use_output_scales=True):
        super().__init__()
        prompt_channels = int(prompt_channels)
        groups = make_group_count(prompt_channels, max_groups=4)
        self.layout_stream = nn.Sequential(
            ConvGNAct(3, prompt_channels, kernel_size=7, activation=nn.GELU, num_groups=4),
            ConvGNAct(prompt_channels, prompt_channels, kernel_size=3, activation=nn.GELU, num_groups=4),
            ConvGNAct(prompt_channels, prompt_channels, kernel_size=3, activation=nn.GELU, num_groups=4),
        )
        self.boundary_stream = nn.Sequential(
            ConvGNAct(4, prompt_channels, kernel_size=3, activation=nn.GELU, num_groups=4),
            nn.Conv2d(
                prompt_channels,
                prompt_channels,
                kernel_size=3,
                padding=1,
                groups=prompt_channels,
                bias=False,
            ),
            nn.GroupNorm(groups, prompt_channels),
            nn.GELU(),
            ConvGNAct(prompt_channels, prompt_channels, kernel_size=3, activation=nn.GELU, num_groups=4),
        )
        denominator = max(1, len(out_channels) - 1)
        initial_weights = [0.35 + 0.40 * (index / float(denominator)) for index in range(len(out_channels))]
        self.layout_mix_logits = nn.ParameterList(
            [
                nn.Parameter(torch.logit(torch.tensor(weight).clamp(1e-4, 1.0 - 1e-4)))
                for weight in initial_weights
            ]
        )
        self.projections = nn.ModuleList(
            [
                nn.Sequential(
                    ConvGNAct(prompt_channels, max(channel // 4, 16), kernel_size=1, activation=nn.GELU, num_groups=4),
                    nn.Conv2d(max(channel // 4, 16), channel, kernel_size=1, bias=False),
                    nn.GroupNorm(make_group_count(channel), channel),
                )
                for channel in out_channels
            ]
        )
        self.prompt_scales = (
            nn.ParameterList(
                [nn.Parameter(torch.full((1, channel, 1, 1), 0.1)) for channel in out_channels]
            )
            if use_output_scales
            else None
        )

    def forward(self, geometry, edge_map, reliability, target_sizes):
        low_frequency = F.avg_pool2d(geometry, kernel_size=9, stride=1, padding=4)
        high_frequency = geometry - low_frequency
        reliability = reliability.clamp(0.0, 1.0)
        layout = self.layout_stream(torch.cat([low_frequency, reliability], dim=1))
        boundary = self.boundary_stream(torch.cat([high_frequency, edge_map, reliability], dim=1))

        prompts = []
        for index, (projection, size) in enumerate(zip(self.projections, target_sizes)):
            layout_weight = torch.sigmoid(self.layout_mix_logits[index])
            prompt = layout_weight * layout + (1.0 - layout_weight) * boundary
            prompt = F.interpolate(prompt, size=size, mode="bilinear", align_corners=False)
            prompt = projection(prompt)
            if self.prompt_scales is not None:
                prompt = self.prompt_scales[index] * prompt
            prompts.append(prompt)
        return prompts


class _StageProjector(nn.Module):
    def __init__(self, prompt_channels, out_channels):
        super().__init__()
        self.projections = nn.ModuleList(
            [
                nn.Sequential(
                    ConvGNAct(
                        prompt_channels,
                        max(channel // 4, 16),
                        kernel_size=1,
                        activation=nn.GELU,
                        num_groups=4,
                    ),
                    nn.Conv2d(max(channel // 4, 16), channel, kernel_size=1, bias=False),
                    nn.GroupNorm(make_group_count(channel), channel),
                )
                for channel in out_channels
            ]
        )

    def forward(self, feature, target_sizes):
        return [
            projection(F.interpolate(feature, size=size, mode="bilinear", align_corners=False))
            for projection, size in zip(self.projections, target_sizes)
        ]


class UnifiedHHAEncoder(nn.Module):
    """Capacity-controlled baseline that mixes all HHA channels immediately."""

    def __init__(self, out_channels, prompt_channels=32):
        super().__init__()
        self.encoder = nn.Sequential(
            ConvGNAct(5, prompt_channels, kernel_size=5, activation=nn.GELU, num_groups=4),
            ConvGNAct(
                prompt_channels,
                prompt_channels,
                kernel_size=3,
                activation=nn.GELU,
                num_groups=4,
            ),
        )
        self.projector = _StageProjector(prompt_channels, out_channels)

    def forward(self, hha, reliability, target_sizes):
        feature = self.encoder(torch.cat([hha, reliability], dim=1))
        prompts = self.projector(feature, target_sizes)
        low = F.avg_pool2d(feature, kernel_size=5, stride=1, padding=2)
        boundary_map = (feature - low).abs().mean(dim=1, keepdim=True)
        return prompts, prompts, boundary_map


class IndependentHHAEncoder(nn.Module):
    """Encode channels independently, then average them without physical relations."""

    def __init__(self, out_channels, prompt_channels=32):
        super().__init__()
        self.stems = nn.ModuleList(
            [
                nn.Sequential(
                    ConvGNAct(1, prompt_channels, kernel_size=5, activation=nn.GELU, num_groups=4),
                    ConvGNAct(
                        prompt_channels,
                        prompt_channels,
                        kernel_size=3,
                        activation=nn.GELU,
                        num_groups=4,
                    ),
                )
                for _ in range(3)
            ]
        )
        self.reliability_projection = ConvGNAct(
            2,
            prompt_channels,
            kernel_size=1,
            activation=nn.GELU,
            num_groups=4,
        )
        self.projector = _StageProjector(prompt_channels, out_channels)

    def forward(self, hha, reliability, target_sizes):
        channel_features = [stem(hha[:, index:index + 1]) for index, stem in enumerate(self.stems)]
        feature = sum(channel_features) / float(len(channel_features))
        feature = feature + self.reliability_projection(reliability)
        prompts = self.projector(feature, target_sizes)
        low = F.avg_pool2d(feature, kernel_size=5, stride=1, padding=2)
        boundary_map = (feature - low).abs().mean(dim=1, keepdim=True)
        return prompts, prompts, boundary_map


class PhysicsFactorizedHHAEncoder(nn.Module):
    """Build D-H layout relations and reliability-conditioned D/H/A boundaries."""

    def __init__(self, out_channels, prompt_channels=32):
        super().__init__()
        self.disparity_stem = nn.Sequential(
            ConvGNAct(1, prompt_channels, kernel_size=5, activation=nn.GELU, num_groups=4),
            ConvGNAct(prompt_channels, prompt_channels, kernel_size=3, activation=nn.GELU, num_groups=4),
        )
        self.height_stem = nn.Sequential(
            ConvGNAct(1, prompt_channels, kernel_size=5, activation=nn.GELU, num_groups=4),
            ConvGNAct(prompt_channels, prompt_channels, kernel_size=3, activation=nn.GELU, num_groups=4),
        )
        self.angle_stem = nn.Sequential(
            ConvGNAct(2, prompt_channels, kernel_size=3, activation=nn.GELU, num_groups=4),
            ConvGNAct(prompt_channels, prompt_channels, kernel_size=3, activation=nn.GELU, num_groups=4),
        )
        self.layout_relation = nn.Sequential(
            ConvGNAct(prompt_channels, prompt_channels, kernel_size=1, activation=nn.GELU, num_groups=4),
            nn.Conv2d(prompt_channels, prompt_channels, kernel_size=3, padding=1, groups=prompt_channels, bias=False),
            nn.GroupNorm(make_group_count(prompt_channels, max_groups=4), prompt_channels),
            nn.GELU(),
        )
        self.layout_refine = ConvGNAct(
            prompt_channels,
            prompt_channels,
            kernel_size=3,
            activation=nn.GELU,
            num_groups=4,
        )
        self.boundary_projections = nn.ModuleList(
            [
                ConvGNAct(
                    prompt_channels,
                    prompt_channels,
                    kernel_size=3,
                    activation=nn.GELU,
                    num_groups=4,
                )
                for _ in range(3)
            ]
        )
        gate_channels = max(prompt_channels // 2, 8)
        self.boundary_router = nn.Sequential(
            ConvGNAct(5, gate_channels, kernel_size=3, activation=nn.GELU, num_groups=4),
            nn.Conv2d(gate_channels, 3, kernel_size=1),
        )
        self.layout_projector = _StageProjector(prompt_channels, out_channels)
        self.boundary_projector = _StageProjector(prompt_channels, out_channels)
        self.last_boundary_weights = None

    @staticmethod
    def _high_pass(feature, kernel_size):
        low = F.avg_pool2d(feature, kernel_size=kernel_size, stride=1, padding=kernel_size // 2)
        return feature - low

    def forward(self, hha, reliability, target_sizes):
        disparity = self.disparity_stem(hha[:, 0:1])
        height = self.height_stem(hha[:, 1:2])

        angle_radians = normalized_hha_angle_to_radians(hha[:, 2:3])
        angle_basis = torch.cat([torch.sin(angle_radians), torch.cos(angle_radians)], dim=1)
        angle = self.angle_stem(angle_basis)

        disparity_low = F.avg_pool2d(disparity, kernel_size=9, stride=1, padding=4)
        height_low = F.avg_pool2d(height, kernel_size=9, stride=1, padding=4)
        relation = self.layout_relation(disparity_low * height_low)
        layout = self.layout_refine(disparity_low + height_low + relation)

        high_frequency = (
            self._high_pass(disparity, 9),
            self._high_pass(height, 9),
            self._high_pass(angle, 5),
        )
        boundary_components = [
            projection(component)
            for projection, component in zip(self.boundary_projections, high_frequency)
        ]
        component_energy = [component.abs().mean(dim=1, keepdim=True) for component in boundary_components]
        router_input = torch.cat(component_energy + [reliability[:, 0:1], reliability[:, 1:2]], dim=1)
        boundary_weights = torch.softmax(self.boundary_router(router_input).float(), dim=1).to(hha.dtype)
        self.last_boundary_weights = boundary_weights.detach()
        boundary = sum(
            boundary_weights[:, index:index + 1] * component
            for index, component in enumerate(boundary_components)
        )

        layout_prompts = self.layout_projector(layout, target_sizes)
        boundary_prompts = self.boundary_projector(boundary, target_sizes)
        boundary_map = boundary.abs().mean(dim=1, keepdim=True)
        return layout_prompts, boundary_prompts, boundary_map


class HHAChannelRelationGate(nn.Module):
    """Learn local HHA channel weights and pairwise geometry relations.

    HHA channels are not interchangeable: disparity carries the strongest
    metric cue, height is useful for layout, and angle is mostly a boundary
    cue.  This block therefore predicts a spatial channel mixture from the
    raw values, high-frequency energy, and the two factorized reliability
    maps.  Pairwise products provide a compact interaction basis without
    replacing the physical D/H/A stems with a large early concatenation.
    """

    def __init__(self, prompt_channels=32):
        super().__init__()
        hidden = max(prompt_channels // 2, 16)
        self.channel_logits = nn.Sequential(
            ConvGNAct(8, hidden, kernel_size=3, activation=nn.GELU, num_groups=4),
            nn.Conv2d(hidden, 3, kernel_size=1),
        )
        nn.init.zeros_(self.channel_logits[-1].weight)
        nn.init.zeros_(self.channel_logits[-1].bias)
        self.relation_mixer = nn.Sequential(
            ConvGNAct(5, hidden, kernel_size=3, activation=nn.GELU, num_groups=4),
            nn.Conv2d(hidden, 3, kernel_size=1),
        )
        nn.init.zeros_(self.relation_mixer[-1].weight)
        nn.init.zeros_(self.relation_mixer[-1].bias)
        self.channel_scale = nn.Parameter(torch.tensor(0.25))
        self.relation_scale = nn.Parameter(torch.tensor(0.10))
        self.last_channel_weights = None

    @staticmethod
    def _high_frequency(hha):
        low = F.avg_pool2d(hha, kernel_size=5, stride=1, padding=2)
        return (hha - low).abs()

    def forward(self, hha, reliability):
        high_frequency = self._high_frequency(hha)
        gate_input = torch.cat([hha, high_frequency, reliability], dim=1)
        weights = torch.softmax(self.channel_logits(gate_input).float(), dim=1).to(hha.dtype)
        self.last_channel_weights = weights.detach()
        scale = self.channel_scale.clamp(0.0, 1.0)
        channel_gain = 1.0 + scale * (3.0 * weights - 1.0)
        pairwise = torch.cat(
            [
                hha[:, 0:1] * hha[:, 1:2],
                hha[:, 0:1] * hha[:, 2:3],
                hha[:, 1:2] * hha[:, 2:3],
                reliability,
            ],
            dim=1,
        )
        relation = self.relation_mixer(pairwise)
        relation_scale = self.relation_scale.clamp(0.0, 1.0)
        return hha * channel_gain + relation_scale * relation


class ChannelRoutedPhysicsFactorizedHHAEncoder(PhysicsFactorizedHHAEncoder):
    """Physics-factorized encoder with adaptive D/H/A relation routing."""

    def __init__(self, out_channels, prompt_channels=32):
        super().__init__(out_channels, prompt_channels=prompt_channels)
        self.channel_relation = HHAChannelRelationGate(prompt_channels=prompt_channels)

    def forward(self, hha, reliability, target_sizes):
        routed_hha = self.channel_relation(hha, reliability)
        return super().forward(routed_hha, reliability, target_sizes)


def build_factorized_hha_encoder(encoding, out_channels, prompt_channels=32):
    if encoding not in GEOMETRY_ENCODINGS:
        raise ValueError(f"Unknown geometry encoding: {encoding!r}")
    if encoding == "unified":
        return UnifiedHHAEncoder(out_channels, prompt_channels=prompt_channels)
    if encoding == "independent":
        return IndependentHHAEncoder(out_channels, prompt_channels=prompt_channels)
    if encoding == "factorized_channel_routed":
        return ChannelRoutedPhysicsFactorizedHHAEncoder(
            out_channels, prompt_channels=prompt_channels
        )
    return PhysicsFactorizedHHAEncoder(out_channels, prompt_channels=prompt_channels)
