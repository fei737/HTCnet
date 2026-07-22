import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from .layers import ConvGNAct, make_group_count


def _match_feature_map(tensor, reference):
    if tensor.shape[2:] != reference.shape[2:]:
        tensor = F.interpolate(tensor, size=reference.shape[2:], mode="bilinear", align_corners=False)
    return tensor.to(dtype=reference.dtype)


def _single_channel_map(tensor, reference, default=1.0):
    if tensor is None:
        return reference.new_full((reference.shape[0], 1, *reference.shape[2:]), float(default))
    tensor = _match_feature_map(tensor, reference)
    if tensor.shape[1] != 1:
        tensor = tensor.mean(dim=1, keepdim=True)
    return tensor.clamp(0.0, 1.0)


class RGBIdentityFusion(nn.Module):
    """Geometry-free path used by the RGB baseline."""

    def forward(self, rgb_feature, prompt=None, guide=None, reliability=None):
        return rgb_feature


class GeometryPromptAdapter(nn.Module):
    """Low-cost reliability-gated geometry prompt residual."""

    def __init__(self, channels, scale_init=0.05):
        super().__init__()
        groups = make_group_count(channels, max_groups=16)
        hidden = max(channels // 8, 16)
        self.prompt_projection = nn.Sequential(
            nn.Conv2d(channels, channels, kernel_size=3, padding=1, groups=channels, bias=False),
            nn.GroupNorm(groups, channels),
            nn.GELU(),
            nn.Conv2d(channels, channels, kernel_size=1, bias=False),
            nn.GroupNorm(groups, channels),
        )
        self.gate = nn.Sequential(
            nn.Conv2d(channels + 2, hidden, kernel_size=1, bias=False),
            nn.GroupNorm(make_group_count(hidden, max_groups=8), hidden),
            nn.GELU(),
            nn.Conv2d(hidden, 1, kernel_size=1),
            nn.Sigmoid(),
        )
        self.residual_scale = nn.Parameter(torch.full((1, channels, 1, 1), float(scale_init)))

    def forward(self, rgb_feature, prompt, guide, reliability=None):
        prompt = _match_feature_map(prompt, rgb_feature)
        guide_map = _single_channel_map(guide, rgb_feature, default=0.0)
        reliability = _single_channel_map(reliability, rgb_feature)
        prompt_delta = self.prompt_projection(prompt)
        gate = self.gate(torch.cat([prompt_delta, guide_map, reliability], dim=1))
        return rgb_feature + self.residual_scale * gate * reliability * prompt_delta


class ShallowGeometryFusion(nn.Module):
    """Boundary-aware local fusion for the two high-resolution stages."""

    def __init__(self, channels, scale_init=0.05):
        super().__init__()
        groups = make_group_count(channels, max_groups=16)
        hidden = max(channels // 2, 32)
        self.rgb_local = nn.Sequential(
            nn.Conv2d(channels, channels, kernel_size=3, padding=1, groups=channels, bias=False),
            nn.GroupNorm(groups, channels),
            nn.GELU(),
        )
        self.prompt_local = nn.Sequential(
            nn.Conv2d(channels, channels, kernel_size=3, padding=1, groups=channels, bias=False),
            nn.GroupNorm(groups, channels),
            nn.GELU(),
        )
        self.detail_gate = nn.Sequential(
            nn.Conv2d(channels + 2, hidden, kernel_size=1, bias=False),
            nn.GroupNorm(make_group_count(hidden, max_groups=8), hidden),
            nn.GELU(),
            nn.Conv2d(hidden, channels, kernel_size=1),
            nn.Sigmoid(),
        )
        self.fusion = nn.Sequential(
            nn.Conv2d(channels * 2 + 2, hidden, kernel_size=1, bias=False),
            nn.GroupNorm(make_group_count(hidden, max_groups=8), hidden),
            nn.GELU(),
            nn.Conv2d(hidden, hidden, kernel_size=3, padding=1, groups=hidden, bias=False),
            nn.GroupNorm(make_group_count(hidden, max_groups=8), hidden),
            nn.GELU(),
            nn.Conv2d(hidden, channels, kernel_size=1, bias=False),
            nn.GroupNorm(groups, channels),
        )
        self.local_scale = nn.Parameter(torch.full((1, channels, 1, 1), float(scale_init)))

    def forward(self, rgb_feature, prompt, guide, reliability=None):
        prompt = _match_feature_map(prompt, rgb_feature)
        guide_map = _single_channel_map(guide, rgb_feature, default=0.0)
        reliability = _single_channel_map(reliability, rgb_feature)
        rgb_local = self.rgb_local(rgb_feature)
        prompt_local = self.prompt_local(prompt)
        detail_gate = self.detail_gate(torch.cat([prompt_local, guide_map, reliability], dim=1))
        delta = self.fusion(torch.cat([rgb_local, prompt_local, guide_map, reliability], dim=1))
        return rgb_feature + self.local_scale * detail_gate * reliability * delta


class ReliabilityConditionedPixelFusion(nn.Module):
    """Linear-cost pixel fusion with explicit RGB-geometry agreement."""

    def __init__(self, channels, scale_init=0.10):
        super().__init__()
        groups = make_group_count(channels, max_groups=16)
        hidden = max(channels // 2, 32)
        self.rgb_local = nn.Sequential(
            nn.Conv2d(channels, channels, kernel_size=3, padding=1, groups=channels, bias=False),
            nn.GroupNorm(groups, channels),
            nn.GELU(),
        )
        self.geometry_local = nn.Sequential(
            nn.Conv2d(channels, channels, kernel_size=3, padding=1, groups=channels, bias=False),
            nn.GroupNorm(groups, channels),
            nn.GELU(),
        )
        gate_channels = max(hidden // 2, 16)
        self.pixel_gate = nn.Sequential(
            ConvGNAct(3, gate_channels, kernel_size=3, activation=nn.GELU, num_groups=4),
            nn.Conv2d(gate_channels, 1, kernel_size=1),
            nn.Sigmoid(),
        )
        self.fusion = nn.Sequential(
            ConvGNAct(
                channels * 2 + 3,
                hidden,
                kernel_size=1,
                activation=nn.GELU,
                num_groups=8,
            ),
            nn.Conv2d(hidden, hidden, kernel_size=3, padding=1, groups=hidden, bias=False),
            nn.GroupNorm(make_group_count(hidden, max_groups=8), hidden),
            nn.GELU(),
            nn.Conv2d(hidden, channels, kernel_size=1, bias=False),
            nn.GroupNorm(groups, channels),
        )
        self.local_scale = nn.Parameter(torch.full((1, channels, 1, 1), float(scale_init)))

    def forward(self, rgb_feature, prompt, guide, reliability=None):
        prompt = _match_feature_map(prompt, rgb_feature)
        guide_map = _single_channel_map(guide, rgb_feature, default=0.0)
        reliability = _single_channel_map(reliability, rgb_feature)
        rgb_local = self.rgb_local(rgb_feature)
        geometry_local = self.geometry_local(prompt)

        rgb_unit = F.normalize(rgb_local.float(), dim=1, eps=1e-6)
        geometry_unit = F.normalize(geometry_local.float(), dim=1, eps=1e-6)
        agreement = (rgb_unit * geometry_unit).sum(dim=1, keepdim=True).to(rgb_local.dtype)
        interaction = rgb_local * geometry_local
        gate = self.pixel_gate(torch.cat([agreement, guide_map, reliability], dim=1))
        delta = self.fusion(
            torch.cat(
                [geometry_local, interaction, guide_map, agreement, reliability],
                dim=1,
            )
        )
        return rgb_feature + self.local_scale * gate * reliability * delta


class GeometryGuidedSemanticAttention(nn.Module):
    """Bias deepest-stage RGB self-attention with reliable geometry affinity."""

    def __init__(self, channels, max_attention_tokens=1024, scale_init=0.05):
        super().__init__()
        self.channels = int(channels)
        self.num_heads = 8 if channels % 8 == 0 else 4 if channels % 4 == 0 else 1
        self.head_dim = channels // self.num_heads
        self.geometry_dim = 8
        self.max_attention_tokens = int(max_attention_tokens)
        self.norm = nn.LayerNorm(channels)
        self.query = nn.Linear(channels, channels, bias=False)
        self.key = nn.Linear(channels, channels, bias=False)
        self.value = nn.Linear(channels, channels, bias=False)
        self.output = nn.Linear(channels, channels)
        self.rgb_position = nn.Conv2d(
            channels, channels, kernel_size=3, padding=1, groups=channels, bias=False
        )
        self.geometry_embedding = ConvGNAct(
            channels * 2 + 1,
            self.num_heads * self.geometry_dim,
            kernel_size=1,
            activation=nn.GELU,
            num_groups=self.num_heads,
        )
        self.geometry_strength = nn.Parameter(torch.tensor(0.5))
        self.semantic_scale = nn.Parameter(torch.full((1, channels, 1, 1), float(scale_init)))

    def _attention_size(self, height, width):
        tokens = height * width
        if tokens <= self.max_attention_tokens:
            return height, width
        scale = math.sqrt(self.max_attention_tokens / float(tokens))
        return max(1, int(round(height * scale))), max(1, int(round(width * scale)))

    def forward(self, rgb_feature, prompt, guide, reliability=None):
        prompt = _match_feature_map(prompt, rgb_feature)
        guide = _match_feature_map(guide, rgb_feature)
        reliability = _single_channel_map(reliability, rgb_feature)
        height, width = rgb_feature.shape[2:]
        attention_size = self._attention_size(height, width)
        if attention_size != (height, width):
            rgb_attention = F.interpolate(
                rgb_feature, size=attention_size, mode="bilinear", align_corners=False
            )
            prompt_attention = F.interpolate(
                prompt, size=attention_size, mode="bilinear", align_corners=False
            )
            guide_attention = F.interpolate(
                guide, size=attention_size, mode="bilinear", align_corners=False
            )
            reliability_attention = F.interpolate(
                reliability, size=attention_size, mode="bilinear", align_corners=False
            )
        else:
            rgb_attention = rgb_feature
            prompt_attention = prompt
            guide_attention = guide
            reliability_attention = reliability

        batch, _, attention_height, attention_width = rgb_attention.shape
        tokens = attention_height * attention_width
        rgb_tokens = (rgb_attention + self.rgb_position(rgb_attention)).flatten(2).transpose(1, 2)
        rgb_tokens = self.norm(rgb_tokens.float()).to(dtype=rgb_attention.dtype)
        query = self.query(rgb_tokens).reshape(
            batch, tokens, self.num_heads, self.head_dim
        ).transpose(1, 2)
        key = self.key(rgb_tokens).reshape(
            batch, tokens, self.num_heads, self.head_dim
        ).transpose(1, 2)
        value = self.value(rgb_tokens).reshape(
            batch, tokens, self.num_heads, self.head_dim
        ).transpose(1, 2)
        attention = (query @ key.transpose(-2, -1)) * (self.head_dim ** -0.5)

        geometry = self.geometry_embedding(
            torch.cat([prompt_attention, guide_attention, reliability_attention], dim=1)
        )
        geometry = geometry.reshape(
            batch, self.num_heads, self.geometry_dim, tokens
        ).transpose(-2, -1)
        geometry = F.normalize(geometry.float(), dim=-1, eps=1e-6).to(dtype=attention.dtype)
        geometry_affinity = geometry @ geometry.transpose(-2, -1)
        reliability_tokens = reliability_attention.flatten(2).clamp(0.0, 1.0)
        pair_reliability = torch.sqrt(
            reliability_tokens.unsqueeze(-1) * reliability_tokens.unsqueeze(-2) + 1e-6
        )
        attention = attention + (
            torch.tanh(self.geometry_strength) * geometry_affinity * pair_reliability
        )
        attention = torch.softmax(attention.float(), dim=-1).to(dtype=value.dtype)
        semantic = (attention @ value).transpose(1, 2).reshape(batch, tokens, self.channels)
        semantic = self.output(semantic).transpose(1, 2).reshape(
            batch, self.channels, attention_height, attention_width
        )
        if semantic.shape[2:] != (height, width):
            semantic = F.interpolate(semantic, size=(height, width), mode="bilinear", align_corners=False)
        return rgb_feature + self.semantic_scale * semantic


class ReliabilityConditionedLinearAttention(nn.Module):
    """Geometry-conditioned RGB attention with linear token complexity."""

    def __init__(self, channels, max_attention_tokens=1024, scale_init=0.10):
        super().__init__()
        self.channels = int(channels)
        self.num_heads = 8 if channels % 8 == 0 else 4 if channels % 4 == 0 else 1
        self.head_dim = channels // self.num_heads
        self.max_attention_tokens = int(max_attention_tokens)
        self.norm = nn.LayerNorm(channels)
        self.qkv = nn.Linear(channels, channels * 3, bias=False)
        self.output = nn.Linear(channels, channels)
        self.rgb_position = nn.Conv2d(
            channels, channels, kernel_size=3, padding=1, groups=channels, bias=False
        )
        self.geometry_embedding = ConvGNAct(
            channels * 2 + 1,
            channels,
            kernel_size=1,
            activation=nn.GELU,
            num_groups=min(self.num_heads, 8),
        )
        self.geometry_key = nn.Linear(channels, channels)
        self.geometry_strength = nn.Parameter(torch.tensor(0.5))
        self.semantic_scale = nn.Parameter(torch.full((1, channels, 1, 1), float(scale_init)))

    def _attention_size(self, height, width):
        tokens = height * width
        if tokens <= self.max_attention_tokens:
            return height, width
        scale = math.sqrt(self.max_attention_tokens / float(tokens))
        return max(1, int(round(height * scale))), max(1, int(round(width * scale)))

    def forward(self, rgb_feature, prompt, guide, reliability=None):
        prompt = _match_feature_map(prompt, rgb_feature)
        guide = _match_feature_map(guide, rgb_feature)
        reliability = _single_channel_map(reliability, rgb_feature)
        height, width = rgb_feature.shape[2:]
        attention_size = self._attention_size(height, width)
        if attention_size != (height, width):
            rgb_attention = F.interpolate(
                rgb_feature, size=attention_size, mode="bilinear", align_corners=False
            )
            prompt_attention = F.interpolate(
                prompt, size=attention_size, mode="bilinear", align_corners=False
            )
            guide_attention = F.interpolate(
                guide, size=attention_size, mode="bilinear", align_corners=False
            )
            reliability_attention = F.interpolate(
                reliability, size=attention_size, mode="bilinear", align_corners=False
            )
        else:
            rgb_attention = rgb_feature
            prompt_attention = prompt
            guide_attention = guide
            reliability_attention = reliability

        batch, _, attention_height, attention_width = rgb_attention.shape
        tokens = attention_height * attention_width
        rgb_tokens = (rgb_attention + self.rgb_position(rgb_attention)).flatten(2).transpose(1, 2)
        rgb_tokens = self.norm(rgb_tokens.float()).to(dtype=rgb_attention.dtype)
        qkv = self.qkv(rgb_tokens).reshape(
            batch, tokens, 3, self.num_heads, self.head_dim
        ).permute(2, 0, 3, 1, 4)
        query, key, value = qkv.unbind(dim=0)

        geometry = self.geometry_embedding(
            torch.cat([prompt_attention, guide_attention, reliability_attention], dim=1)
        )
        geometry_tokens = geometry.flatten(2).transpose(1, 2)
        geometry_key = torch.sigmoid(self.geometry_key(geometry_tokens)).reshape(
            batch, tokens, self.num_heads, self.head_dim
        ).transpose(1, 2)
        reliability_tokens = reliability_attention.flatten(2).transpose(1, 2).unsqueeze(1)
        key_scale = 1.0 + (
            torch.tanh(self.geometry_strength) * geometry_key * reliability_tokens
        )

        query = F.elu(query.float()) + 1.0
        key = (F.elu(key.float()) + 1.0) * key_scale.float()
        value = value.float()
        context = torch.einsum("bhnd,bhne->bhde", key, value)
        key_sum = key.sum(dim=2)
        normalizer = torch.einsum("bhnd,bhd->bhn", query, key_sum).clamp_min(1e-6).reciprocal()
        semantic = torch.einsum("bhnd,bhde,bhn->bhne", query, context, normalizer)
        semantic = semantic.transpose(1, 2).reshape(batch, tokens, self.channels)
        semantic = self.output(semantic.to(dtype=rgb_attention.dtype)).transpose(1, 2).reshape(
            batch, self.channels, attention_height, attention_width
        )
        if semantic.shape[2:] != (height, width):
            semantic = F.interpolate(
                semantic, size=(height, width), mode="bilinear", align_corners=False
            )
        return rgb_feature + self.semantic_scale * reliability * semantic


class TopologyAwareLocalFusion(nn.Module):
    """Local RGB-geometry interaction with an explicit boundary prior.

    The block separates local appearance and geometry responses, then gates
    their interaction by agreement, boundary evidence, and reliability.  It
    starts as a near identity so pretrained RGB features remain stable.
    """

    def __init__(self, channels, scale_init=0.05):
        super().__init__()
        # FactorizedPromptRouter can request the reliability gate at its output.
        # Keeping this explicit prevents accidental double attenuation.
        self.external_reliability_gate = False
        groups = make_group_count(channels, max_groups=16)
        hidden = max(channels // 2, 32)
        self.rgb_local = nn.Sequential(
            nn.Conv2d(channels, channels, kernel_size=5, padding=2, groups=channels, bias=False),
            nn.GroupNorm(groups, channels),
            nn.GELU(),
        )
        self.geometry_local = nn.Sequential(
            nn.Conv2d(channels, channels, kernel_size=5, padding=2, groups=channels, bias=False),
            nn.GroupNorm(groups, channels),
            nn.GELU(),
        )
        self.interaction = nn.Sequential(
            ConvGNAct(channels * 2 + 3, hidden, kernel_size=1, activation=nn.GELU, num_groups=8),
            nn.Conv2d(hidden, hidden, kernel_size=3, padding=1, groups=hidden, bias=False),
            nn.GroupNorm(make_group_count(hidden, max_groups=8), hidden),
            nn.GELU(),
            nn.Conv2d(hidden, channels, kernel_size=1, bias=False),
            nn.GroupNorm(groups, channels),
        )
        self.gate = nn.Sequential(
            ConvGNAct(3, max(channels // 8, 16), kernel_size=3, activation=nn.GELU, num_groups=4),
            nn.Conv2d(max(channels // 8, 16), 1, kernel_size=1),
            nn.Sigmoid(),
        )
        self.layer_scale = nn.Parameter(torch.full((1, channels, 1, 1), float(scale_init)))

    def forward(self, rgb_feature, prompt, guide, reliability=None):
        prompt = _match_feature_map(prompt, rgb_feature)
        guide_map = _single_channel_map(guide, rgb_feature, default=0.0)
        reliability = _single_channel_map(reliability, rgb_feature)
        rgb_local = self.rgb_local(rgb_feature)
        geometry_local = self.geometry_local(prompt)
        agreement = F.cosine_similarity(rgb_local.float(), geometry_local.float(), dim=1).unsqueeze(1)
        delta = self.interaction(
            torch.cat([rgb_local, geometry_local, agreement, guide_map, reliability], dim=1)
        )
        gate = self.gate(torch.cat([agreement, guide_map, reliability], dim=1))
        residual_reliability = (
            rgb_feature.new_ones(rgb_feature.shape[0], 1, *rgb_feature.shape[2:])
            if self.external_reliability_gate
            else reliability
        )
        return rgb_feature + self.layer_scale * gate * residual_reliability * delta


class TopologyAwareSemanticAttention(nn.Module):
    """Topology-biased semantic attention with bounded token complexity.

    A pairwise boundary-distance bias encourages tokens with compatible
    geometry to exchange context.  It is computed only after reducing the
    deepest feature map to ``max_attention_tokens`` so memory use is bounded.
    """

    def __init__(
        self,
        channels,
        max_attention_tokens=512,
        scale_init=0.05,
        layer_scale_init=1e-3,
    ):
        super().__init__()
        self.external_reliability_gate = False
        self.channels = int(channels)
        self.num_heads = 8 if channels % 8 == 0 else 4 if channels % 4 == 0 else 1
        self.head_dim = channels // self.num_heads
        self.max_attention_tokens = int(max_attention_tokens)
        self.norm = nn.LayerNorm(channels)
        self.qkv = nn.Linear(channels, channels * 3, bias=False)
        self.proj = nn.Linear(channels, channels)
        self.local_position = nn.Conv2d(
            channels, channels, kernel_size=5, padding=2, groups=channels, bias=False
        )
        self.geometry_projection = ConvGNAct(
            channels + 2,
            self.num_heads,
            kernel_size=1,
            activation=nn.GELU,
            num_groups=self.num_heads,
        )
        self.topology_temperature = nn.Parameter(
            torch.full((1, self.num_heads, 1, 1), 0.5)
        )
        self.layer_scale = nn.Parameter(torch.full((1, channels, 1, 1), float(layer_scale_init)))
        self.semantic_scale = nn.Parameter(torch.full((1, channels, 1, 1), float(scale_init)))

    def _attention_size(self, height, width):
        tokens = height * width
        if tokens <= self.max_attention_tokens:
            return height, width
        scale = math.sqrt(self.max_attention_tokens / float(tokens))
        return max(1, int(round(height * scale))), max(1, int(round(width * scale)))

    def forward(self, rgb_feature, prompt, guide, reliability=None):
        prompt = _match_feature_map(prompt, rgb_feature)
        guide = _single_channel_map(guide, rgb_feature, default=0.0)
        reliability = _single_channel_map(reliability, rgb_feature)
        height, width = rgb_feature.shape[2:]
        attention_size = self._attention_size(height, width)
        if attention_size != (height, width):
            rgb_small = F.interpolate(rgb_feature, size=attention_size, mode="bilinear", align_corners=False)
            prompt_small = F.interpolate(prompt, size=attention_size, mode="bilinear", align_corners=False)
            guide_small = F.interpolate(guide, size=attention_size, mode="bilinear", align_corners=False)
            reliability_small = F.interpolate(reliability, size=attention_size, mode="bilinear", align_corners=False)
        else:
            rgb_small, prompt_small, guide_small, reliability_small = rgb_feature, prompt, guide, reliability

        batch, _, small_h, small_w = rgb_small.shape
        tokens = small_h * small_w
        x = rgb_small + prompt_small
        x_tokens = self.norm(x.flatten(2).transpose(1, 2).float()).to(dtype=x.dtype)
        qkv = self.qkv(x_tokens).reshape(batch, tokens, 3, self.num_heads, self.head_dim)
        qkv = qkv.permute(2, 0, 3, 1, 4)
        query, key, value = qkv.unbind(dim=0)
        attention = torch.matmul(query.float(), key.float().transpose(-2, -1)) * (self.head_dim ** -0.5)

        geometry = self.geometry_projection(torch.cat([prompt_small, guide_small, reliability_small], dim=1))
        geometry = geometry.flatten(2).unsqueeze(-1)
        geometry = geometry.expand(-1, -1, -1, tokens)
        topology_distance = (geometry - geometry.transpose(-2, -1)).abs()
        topology_distance = topology_distance / (topology_distance.detach().flatten(2).amax(dim=2).view(batch, self.num_heads, 1, 1).clamp_min(1e-6))
        pair_reliability = torch.sqrt(
            reliability_small.flatten(2).unsqueeze(-1) * reliability_small.flatten(2).unsqueeze(-2) + 1e-6
        )
        strength = F.softplus(self.topology_temperature.float()).clamp(0.0, 2.0)
        attention = attention - strength * topology_distance.float() * pair_reliability.float()
        attention = torch.softmax(attention, dim=-1).to(dtype=value.dtype)
        semantic = torch.matmul(attention, value).transpose(1, 2).reshape(batch, tokens, self.channels)
        semantic = self.proj(semantic).transpose(1, 2).reshape(batch, self.channels, small_h, small_w)
        semantic = semantic + self.local_position(
            value.transpose(1, 2).reshape(batch, self.channels, small_h, small_w)
        )
        if semantic.shape[2:] != (height, width):
            semantic = F.interpolate(semantic, size=(height, width), mode="bilinear", align_corners=False)
        residual_reliability = (
            rgb_feature.new_ones(rgb_feature.shape[0], 1, *rgb_feature.shape[2:])
            if self.external_reliability_gate
            else reliability
        )
        return rgb_feature + self.semantic_scale * residual_reliability * self.layer_scale * semantic


class FactorizedPromptRouter(nn.Module):
    """Route layout and boundary prompts before one reliability-gated injection."""

    def __init__(self, channels, stage_index, num_stages, inner_fusion, encoding):
        super().__init__()
        self.inner_fusion = inner_fusion
        self.encoding = str(encoding)
        self.final_only_reliability = self.encoding in {
            "factorized_final",
            "factorized_final_soft",
        }
        if self.final_only_reliability and hasattr(self.inner_fusion, "external_reliability_gate"):
            self.inner_fusion.external_reliability_gate = True
        progress = float(stage_index) / float(max(num_stages - 1, 1))
        layout_prior = 0.25 + 0.50 * progress
        if self.encoding == "factorized_swapped":
            layout_prior = 1.0 - layout_prior
        if self.encoding in {"unified", "independent"}:
            layout_prior = 0.50
        prior = torch.tensor([layout_prior, 1.0 - layout_prior]).clamp_min(1e-4)
        self.register_buffer("route_prior_logits", prior.log().view(1, 2, 1, 1))

        self.route_predictor = None
        if self.encoding in {
            "factorized_routed",
            "factorized_final",
            "factorized_final_soft",
            "factorized_channel_routed",
        }:
            hidden = max(channels // 16, 16)
            self.route_predictor = nn.Sequential(
                ConvGNAct(5, hidden, kernel_size=3, activation=nn.GELU, num_groups=4),
                nn.Conv2d(hidden, 2, kernel_size=1),
            )
            nn.init.zeros_(self.route_predictor[-1].weight)
            nn.init.zeros_(self.route_predictor[-1].bias)
        self.last_route_weights = None

    @staticmethod
    def _agreement(rgb_feature, prompt):
        rgb_unit = F.normalize(rgb_feature.float(), dim=1, eps=1e-6)
        prompt_unit = F.normalize(prompt.float(), dim=1, eps=1e-6)
        return (rgb_unit * prompt_unit).sum(dim=1, keepdim=True).to(rgb_feature.dtype)

    def forward(self, rgb_feature, prompt, guide, reliability=None):
        if not isinstance(prompt, (tuple, list)) or len(prompt) != 2:
            raise ValueError("FactorizedPromptRouter expects (layout_prompt, boundary_prompt)")
        layout_prompt = _match_feature_map(prompt[0], rgb_feature)
        boundary_prompt = _match_feature_map(prompt[1], rgb_feature)
        guide_map = _single_channel_map(guide, rgb_feature, default=0.0)

        if reliability is None:
            reliability = rgb_feature.new_ones(
                rgb_feature.shape[0], 2, *rgb_feature.shape[2:]
            )
        reliability = _match_feature_map(reliability, rgb_feature).clamp(0.0, 1.0)
        if reliability.shape[1] == 1:
            reliability = reliability.expand(-1, 2, -1, -1)
        elif reliability.shape[1] != 2:
            raise ValueError(f"Factorized reliability must have 1 or 2 channels, got {reliability.shape[1]}")
        routing_reliability = (
            torch.ones_like(reliability) if self.final_only_reliability else reliability
        )

        layout_agreement = self._agreement(rgb_feature, layout_prompt)
        boundary_agreement = self._agreement(rgb_feature, boundary_prompt)
        route_logits = self.route_prior_logits.to(dtype=rgb_feature.dtype).expand(
            rgb_feature.shape[0], -1, rgb_feature.shape[2], rgb_feature.shape[3]
        )
        if self.route_predictor is not None:
            route_input = torch.cat(
                [
                    layout_agreement,
                    boundary_agreement,
                    routing_reliability[:, 0:1],
                    routing_reliability[:, 1:2],
                    guide_map,
                ],
                dim=1,
            )
            route_logits = route_logits + self.route_predictor(route_input)
        route_weights = torch.softmax(route_logits.float(), dim=1).to(rgb_feature.dtype)
        self.last_route_weights = route_weights.detach()

        routed_prompt = (
            route_weights[:, 0:1] * layout_prompt
            + route_weights[:, 1:2] * boundary_prompt
        )
        routed_reliability = (
            route_weights[:, 0:1] * reliability[:, 0:1]
            + route_weights[:, 1:2] * reliability[:, 1:2]
        )
        if self.final_only_reliability:
            # The inner block still receives reliability for topology-aware
            # conditioning, but owns no final reliability multiplication.
            inner = self.inner_fusion(rgb_feature, routed_prompt, guide, routed_reliability)
            final_reliability = _single_channel_map(routed_reliability, rgb_feature)
            return rgb_feature + final_reliability * (inner - rgb_feature)
        return self.inner_fusion(rgb_feature, routed_prompt, guide, routed_reliability)


STAGE_FUSION_MODES = {"rgb", "prompt", "shallow", "deep", "stagewise", "topology"}
ARCHITECTURE_VARIANTS = {"legacy", "refined", "factorized"}


def build_stage_fusion(
    mode,
    channels,
    stage_index,
    num_stages,
    max_attention_tokens,
    local_scale_init,
    semantic_scale_init,
    architecture_variant="legacy",
    geometry_encoding="factorized_routed",
):
    if architecture_variant not in ARCHITECTURE_VARIANTS:
        raise ValueError(f"Unknown architecture variant: {architecture_variant!r}")
    uses_linear_fusion = architecture_variant in {"refined", "factorized"}
    shallow_fusion = (
        TopologyAwareLocalFusion
        if mode == "topology"
        else ReliabilityConditionedPixelFusion
        if uses_linear_fusion
        else ShallowGeometryFusion
    )
    deep_fusion = (
        TopologyAwareSemanticAttention
        if mode == "topology"
        else ReliabilityConditionedLinearAttention
        if uses_linear_fusion
        else GeometryGuidedSemanticAttention
    )
    if mode == "rgb":
        return RGBIdentityFusion()
    if mode == "prompt":
        fusion = GeometryPromptAdapter(channels, scale_init=local_scale_init)
    elif mode == "shallow":
        if stage_index < min(2, num_stages):
            fusion = shallow_fusion(channels, scale_init=local_scale_init)
        else:
            fusion = GeometryPromptAdapter(channels, scale_init=local_scale_init)
    elif mode == "deep":
        if stage_index == num_stages - 1:
            fusion = deep_fusion(
                channels, max_attention_tokens=max_attention_tokens, scale_init=semantic_scale_init
            )
        else:
            fusion = GeometryPromptAdapter(channels, scale_init=local_scale_init)
    elif mode in {"stagewise", "topology"}:
        if stage_index < min(2, num_stages):
            fusion = shallow_fusion(channels, scale_init=local_scale_init)
        elif stage_index == num_stages - 1:
            fusion = deep_fusion(
                channels, max_attention_tokens=max_attention_tokens, scale_init=semantic_scale_init
            )
        else:
            fusion = GeometryPromptAdapter(channels, scale_init=local_scale_init)
    else:
        raise ValueError(f"Unknown stage fusion mode: {mode!r}")

    if architecture_variant == "factorized":
        return FactorizedPromptRouter(
            channels,
            stage_index,
            num_stages,
            fusion,
            geometry_encoding,
        )
    return fusion


class CrossScaleFeatureAligner(nn.Module):
    def __init__(self, shallow_channels, deep_channels):
        super().__init__()
        hidden = max(shallow_channels // 2, 1)
        self.alignment = nn.Sequential(
            nn.Conv2d(deep_channels, shallow_channels, kernel_size=1, bias=False),
            nn.BatchNorm2d(shallow_channels),
            nn.ReLU(inplace=True),
        )
        self.channel_gate = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(shallow_channels, hidden, kernel_size=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(hidden, shallow_channels, kernel_size=1),
            nn.Sigmoid(),
        )
        self.spatial_gate = nn.Sequential(
            nn.Conv2d(shallow_channels * 2, hidden, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(hidden, 1, kernel_size=1),
            nn.Sigmoid(),
        )

    def forward(self, shallow_feature, deep_feature):
        deep_feature = F.interpolate(
            deep_feature, size=shallow_feature.shape[2:], mode="bilinear", align_corners=False
        )
        deep_feature = self.alignment(deep_feature)
        channel_gate = self.channel_gate(deep_feature)
        spatial_gate = self.spatial_gate(torch.cat([shallow_feature, deep_feature], dim=1))
        return shallow_feature * spatial_gate * channel_gate + deep_feature


class ResidualCrossScaleAligner(CrossScaleFeatureAligner):
    """Identity-preserving top-down alignment for complementary stage features."""

    def __init__(self, shallow_channels, deep_channels, scale_init=0.10):
        super().__init__(shallow_channels, deep_channels)
        self.residual_scale = nn.Parameter(
            torch.full((1, shallow_channels, 1, 1), float(scale_init))
        )

    def forward(self, shallow_feature, deep_feature):
        deep_feature = F.interpolate(
            deep_feature, size=shallow_feature.shape[2:], mode="bilinear", align_corners=False
        )
        deep_feature = self.alignment(deep_feature)
        channel_gate = self.channel_gate(deep_feature)
        spatial_gate = self.spatial_gate(torch.cat([shallow_feature, deep_feature], dim=1))
        return shallow_feature + self.residual_scale * spatial_gate * channel_gate * deep_feature


class LightweightPyramidContext(nn.Module):
    def __init__(self, channels, scale_init=0.10):
        super().__init__()
        hidden = max(channels // 4, 32)
        self.local_context = nn.Sequential(
            nn.Conv2d(
                channels,
                channels,
                kernel_size=3,
                padding=2,
                dilation=2,
                groups=channels,
                bias=False,
            ),
            nn.GroupNorm(make_group_count(channels, max_groups=16), channels),
            nn.GELU(),
        )
        self.context_gate = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(channels, hidden, kernel_size=1, bias=False),
            nn.GELU(),
            nn.Conv2d(hidden, channels, kernel_size=1),
            nn.Sigmoid(),
        )
        self.context_scale = nn.Parameter(torch.full((1, channels, 1, 1), float(scale_init)))

    def forward(self, feature):
        return feature + self.context_scale * self.context_gate(feature) * self.local_context(feature)
