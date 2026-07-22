import os

import segmentation_models_pytorch as smp
import torch
import torch.nn as nn

from .fusion import (
    ARCHITECTURE_VARIANTS,
    STAGE_FUSION_MODES,
    CrossScaleFeatureAligner,
    LightweightPyramidContext,
    ResidualCrossScaleAligner,
    build_stage_fusion,
)
from .geometry import (
    GEOMETRY_ENCODINGS,
    FactorizedGeometryReliabilityHead,
    GeometryGuidePyramid,
    GeometryPromptEncoder,
    GeometryReliabilityHead,
    GeometryStructureEncoder,
    HHAEdgeDescriptor,
    build_factorized_hha_encoder,
    normalized_hha_invalid_hint,
)
from .layers import LearnableGate, is_main_process, set_encoder_drop_path


class StageAwareGeometryEncoder(nn.Module):
    """RGB encoder with stage-specific, reliability-aware HHA guidance."""

    def __init__(
        self,
        pretrained_path=None,
        encoder_name="mit_b2",
        drop_path_rate=0.0,
        prompt_channels=32,
        max_attention_tokens=512,
        local_scale_init=0.05,
        semantic_scale_init=0.05,
        fusion_mode="stagewise",
        use_reliability=True,
        architecture_variant="legacy",
        geometry_encoding="factorized_routed",
        geometry_channels="dha",
    ):
        super().__init__()
        if fusion_mode not in STAGE_FUSION_MODES:
            raise ValueError(f"Unknown fusion mode: {fusion_mode!r}")
        self.fusion_mode = fusion_mode
        if architecture_variant not in ARCHITECTURE_VARIANTS:
            raise ValueError(f"Unknown architecture variant: {architecture_variant!r}")
        self.architecture_variant = architecture_variant
        if architecture_variant == "factorized" and geometry_encoding not in GEOMETRY_ENCODINGS:
            raise ValueError(f"Unknown geometry encoding: {geometry_encoding!r}")
        self.geometry_encoding = str(geometry_encoding)
        geometry_channels = str(geometry_channels).strip().lower()
        if not geometry_channels or any(name not in "dha" for name in geometry_channels):
            raise ValueError(f"geometry_channels must be a non-empty subset of 'dha', got {geometry_channels!r}")
        self.geometry_channels = "".join(name for name in "dha" if name in geometry_channels)
        self.register_buffer(
            "geometry_channel_mask",
            torch.tensor(
                [1.0 if name in self.geometry_channels else 0.0 for name in "dha"]
            ).view(1, 3, 1, 1),
            persistent=False,
        )
        self.use_reliability = bool(use_reliability and fusion_mode != "rgb")

        encoder_weights = "imagenet" if pretrained_path == "imagenet" else None
        self.rgb_backbone = smp.encoders.get_encoder(
            encoder_name, in_channels=3, weights=encoder_weights
        )
        if drop_path_rate > 0:
            num_layers = set_encoder_drop_path(self.rgb_backbone, drop_path_rate)
            if is_main_process():
                print(
                    f"RGB backbone stochastic depth: rate={drop_path_rate:g}, "
                    f"layers={num_layers}"
                )

        encoder_channels = self.rgb_backbone.out_channels
        self.valid_indices = [
            index
            for index, channels in enumerate(encoder_channels)
            if index > 0 and channels > 0
        ]
        self.out_channels = [encoder_channels[index] for index in self.valid_indices]
        if not self.out_channels:
            raise ValueError(f"Encoder {encoder_name!r} did not expose usable feature stages")

        if pretrained_path and pretrained_path != "imagenet":
            if not os.path.exists(pretrained_path):
                raise FileNotFoundError(
                    f"Pretrained encoder checkpoint not found: {pretrained_path}"
                )
            state_dict = torch.load(pretrained_path, map_location="cpu")
            state_dict = state_dict.get("state_dict", state_dict)
            torch.nn.Module.load_state_dict(self.rgb_backbone, state_dict, strict=False)
            if is_main_process():
                print(f"Loaded RGB backbone weights from {pretrained_path}")

        uses_geometry = fusion_mode != "rgb"
        self.reliability_head = (
            FactorizedGeometryReliabilityHead()
            if self.use_reliability and architecture_variant == "factorized"
            else GeometryReliabilityHead()
            if self.use_reliability
            else None
        )
        self.structure_encoder = (
            GeometryStructureEncoder()
            if uses_geometry and architecture_variant != "factorized"
            else None
        )
        self.edge_descriptor = (
            HHAEdgeDescriptor()
            if uses_geometry and architecture_variant != "factorized"
            else None
        )
        self.guide_pyramid = GeometryGuidePyramid(self.out_channels) if uses_geometry else None
        if uses_geometry and architecture_variant == "factorized":
            self.prompt_encoder = build_factorized_hha_encoder(
                geometry_encoding,
                self.out_channels,
                prompt_channels=prompt_channels,
            )
        elif uses_geometry:
            self.prompt_encoder = GeometryPromptEncoder(
                self.out_channels,
                prompt_channels=prompt_channels,
                use_output_scales=architecture_variant == "legacy",
            )
        else:
            self.prompt_encoder = None
        self.reliability_gate = (
            LearnableGate(init_lo=0.25, init_span=0.75)
            if uses_geometry and architecture_variant == "legacy"
            else None
        )

        num_stages = len(self.out_channels)
        self.fusion_stages = nn.ModuleList(
            [
                build_stage_fusion(
                    fusion_mode,
                    channels,
                    stage_index,
                    num_stages,
                    max_attention_tokens,
                    local_scale_init,
                    semantic_scale_init,
                    architecture_variant=architecture_variant,
                    geometry_encoding=geometry_encoding,
                )
                for stage_index, channels in enumerate(self.out_channels)
            ]
        )
        self.cross_scale_aligners = nn.ModuleList(
            [
                (
                    ResidualCrossScaleAligner(shallow_channels, deep_channels)
                    if architecture_variant in {"refined", "factorized"}
                    else CrossScaleFeatureAligner(shallow_channels, deep_channels)
                )
                for shallow_channels, deep_channels in zip(
                    self.out_channels[:-1], self.out_channels[1:]
                )
            ]
        )
        self.context_head = LightweightPyramidContext(self.out_channels[-1])

    def forward(self, rgb, hha, return_context=False, hha_invalid_hint=None):
        backbone_features = self.rgb_backbone(rgb)
        rgb_features = [backbone_features[index] for index in self.valid_indices]
        routed_hha = hha * self.geometry_channel_mask.to(dtype=hha.dtype)
        factorized_prompts = None

        if self.fusion_mode == "rgb":
            reliability = hha.new_ones(hha.shape[0], 1, hha.shape[2], hha.shape[3])
            edge_map = reliability.new_zeros(reliability.shape)
            guides = [feature.new_zeros(feature.shape) for feature in rgb_features]
            fused_features = list(rgb_features)
        elif self.architecture_variant == "legacy":
            if self.reliability_head is None:
                reliability = hha.new_ones(hha.shape[0], 1, hha.shape[2], hha.shape[3])
                routed_reliability = None
            else:
                reliability = self.reliability_head(routed_hha, invalid_hint=hha_invalid_hint)
                reliability = reliability.clamp(0.0, 1.0)
                routed_reliability = reliability

            prompt_reliability = (
                reliability if self.use_reliability else torch.ones_like(reliability)
            )
            hha_guided = routed_hha * self.reliability_gate(prompt_reliability)
            geometry = self.structure_encoder(routed_hha[:, 0:2] * prompt_reliability)
            edge_map = self.edge_descriptor(hha_guided) * prompt_reliability
            target_sizes = [feature.shape[2:] for feature in rgb_features]
            guides = self.guide_pyramid(edge_map, target_sizes)
            prompts = self.prompt_encoder(
                geometry, edge_map, prompt_reliability, target_sizes
            )
            fused_features = [
                fusion_stage(rgb_feature, prompt, guide, routed_reliability)
                for fusion_stage, rgb_feature, prompt, guide in zip(
                    self.fusion_stages, rgb_features, prompts, guides
                )
            ]
        elif self.architecture_variant == "factorized":
            if self.reliability_head is None:
                reliability = hha.new_ones(hha.shape[0], 2, hha.shape[2], hha.shape[3])
            else:
                reliability = self.reliability_head(routed_hha, invalid_hint=hha_invalid_hint)
                reliability = reliability.clamp(0.0, 1.0)

            invalid = normalized_hha_invalid_hint(hha, hha_invalid_hint)
            final_only_reliability = self.geometry_encoding in {
                "factorized_final",
                "factorized_final_soft",
            }
            if self.geometry_encoding == "factorized_final_soft":
                # Preserve hard invalid masking while avoiding aggressive attenuation
                # of valid geometry whose learned reliability is only moderately low.
                routed_reliability = (0.75 + 0.25 * reliability) * (1.0 - invalid)
            else:
                routed_reliability = reliability
            valid_hha = routed_hha * (1.0 - invalid)
            prompt_reliability = (
                torch.ones_like(reliability) if final_only_reliability else reliability
            )
            target_sizes = [feature.shape[2:] for feature in rgb_features]
            layout_prompts, boundary_prompts, edge_map = self.prompt_encoder(
                valid_hha,
                prompt_reliability,
                target_sizes,
            )
            guides = self.guide_pyramid(edge_map, target_sizes)
            prompts = list(zip(layout_prompts, boundary_prompts))
            factorized_prompts = {
                "layout": layout_prompts,
                "boundary": boundary_prompts,
            }
            fused_features = [
                fusion_stage(rgb_feature, prompt, guide, routed_reliability)
                for fusion_stage, rgb_feature, prompt, guide in zip(
                    self.fusion_stages, rgb_features, prompts, guides
                )
            ]
        else:
            if self.reliability_head is None:
                reliability = hha.new_ones(hha.shape[0], 1, hha.shape[2], hha.shape[3])
                routed_reliability = None
            else:
                reliability = self.reliability_head(routed_hha, invalid_hint=hha_invalid_hint)
                reliability = reliability.clamp(0.0, 1.0)
                routed_reliability = reliability

            prompt_reliability = (
                reliability if self.use_reliability else torch.ones_like(reliability)
            )
            invalid = normalized_hha_invalid_hint(hha, hha_invalid_hint)
            valid_hha = routed_hha * (1.0 - invalid)
            geometry = self.structure_encoder(valid_hha[:, 0:2])
            edge_map = self.edge_descriptor(valid_hha)
            target_sizes = [feature.shape[2:] for feature in rgb_features]
            guides = self.guide_pyramid(edge_map, target_sizes)
            prompts = self.prompt_encoder(
                geometry, edge_map, prompt_reliability, target_sizes
            )
            fused_features = [
                fusion_stage(rgb_feature, prompt, guide, routed_reliability)
                for fusion_stage, rgb_feature, prompt, guide in zip(
                    self.fusion_stages, rgb_features, prompts, guides
                )
            ]

        fused_features[-1] = self.context_head(fused_features[-1])
        for index in range(len(fused_features) - 2, -1, -1):
            fused_features[index] = self.cross_scale_aligners[index](
                fused_features[index], fused_features[index + 1]
            )

        if return_context:
            return {
                "features": fused_features,
                "rgb_features": rgb_features,
                "geometry_guides": guides,
                "geometry_reliability": reliability,
                "geometry_edges": edge_map,
                "architecture_variant": self.architecture_variant,
                "geometry_encoding": self.geometry_encoding,
                "geometry_channels": self.geometry_channels,
                "geometry_routes": [
                    getattr(stage, "last_route_weights", None)
                    for stage in self.fusion_stages
                ],
                "geometry_channel_weights": getattr(
                    getattr(self.prompt_encoder, "channel_relation", None),
                    "last_channel_weights",
                    None,
                ),
                "geometry_boundary_weights": getattr(
                    self.prompt_encoder,
                    "last_boundary_weights",
                    None,
                ),
                "factorized_prompts": factorized_prompts,
            }
        return fused_features
