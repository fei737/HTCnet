import os

import segmentation_models_pytorch as smp
import torch
import torch.nn as nn

from .common import is_main_process, set_encoder_drop_path
from .fusion import CrossScaleFeatureAligner, LocalSemanticFusionBlock, StripPyramidASPP
from .geometry import (
    AngularBoundaryDescriptor,
    AngularGuidancePyramid,
    CrossModalReliabilityEstimator,
    FrequencyAwareGeometryPrompt,
    GeometryPromptRecovery,
    GeometryReliabilityEstimator,
    RGBGuidedGeometryRecovery,
    StructuralContrastEnhancer,
    StructuralPromptFilter,
)


class UncertaintyGuidedFusionEncoder(nn.Module):
    """RGB-D encoder with reliability-guided geometry prompts.

    Output:
        ``fused_features`` is a list of multi-scale tensors passed to the
        decoder. For ``mit_b2`` and 480x640 input, the typical shapes are:
        [B, 64, 120, 160], [B, 128, 60, 80], [B, 320, 30, 40],
        [B, 512, 15, 20].
    """

    def __init__(
        self,
        pretrained_path=None,
        encoder_name="mit_b2",
        safe_mode=False,
        layout_mode="ssm",
        drop_path_rate=0.0,
        use_prompt_recovery=True,
        prompt_recovery_mode="rgpr",
        use_confidence_routing=True,
        consistency_routing_mode="gacr",
        prompt_channels=32,
        layout_state_dim=16,
        geometry_routing_mode="rcfr",
        reliability_strength=0.30,
        attention_tokens=1024,
        local_scale_init=1e-4,
        semantic_scale_init=1e-4,
        fusion_branch_mode="both",
        use_prompt_autocorr=True,
        use_directional_edge_refine=True,
        use_grad_checkpoint=False,
    ):
        super().__init__()
        self.safe_mode = safe_mode
        self.use_grad_checkpoint = use_grad_checkpoint
        self.layout_mode = layout_mode
        self.use_prompt_recovery = use_prompt_recovery
        self.prompt_recovery_mode = prompt_recovery_mode
        self.use_confidence_routing = use_confidence_routing
        self.consistency_routing_mode = consistency_routing_mode
        self.geometry_routing_mode = geometry_routing_mode
        self.reliability_strength = float(reliability_strength)
        self.fusion_branch_mode = fusion_branch_mode
        self.use_prompt_autocorr = bool(use_prompt_autocorr)
        self.use_directional_edge_refine = bool(use_directional_edge_refine)
        if prompt_recovery_mode not in {"legacy", "rgpr"}:
            raise ValueError(f"Unknown prompt_recovery_mode: {prompt_recovery_mode!r}")
        if consistency_routing_mode not in {"none", "gacr"}:
            raise ValueError(f"Unknown consistency_routing_mode: {consistency_routing_mode!r}")

        encoder_weights = "imagenet" if pretrained_path == "imagenet" else None
        self.rgb_encoder = smp.encoders.get_encoder(encoder_name, in_channels=3, weights=encoder_weights)
        if drop_path_rate > 0:
            n_dp = set_encoder_drop_path(self.rgb_encoder, drop_path_rate)
            if is_main_process():
                if n_dp > 0:
                    print(f"Stochastic depth enabled on encoder: drop_path_rate={drop_path_rate}, layers={n_dp}")
                else:
                    print(f"drop_path_rate={drop_path_rate} requested but encoder '{encoder_name}' has no DropPath layers; skipped.")

        self.mueb = GeometryReliabilityEstimator(in_channels=3)
        if prompt_recovery_mode == "rgpr":
            self.hha_recovery = RGBGuidedGeometryRecovery(in_channels=3)
        else:
            self.hha_recovery = GeometryPromptRecovery(in_channels=3)
        self.gacr = CrossModalReliabilityEstimator() if consistency_routing_mode == "gacr" else None
        self.bsde = StructuralContrastEnhancer(in_channels=2)

        self.encoder_channels = self.rgb_encoder.out_channels
        self.valid_indices = [i for i, ch in enumerate(self.encoder_channels) if ch > 0 and i > 0]
        self.out_channels = [self.encoder_channels[i] for i in self.valid_indices]

        self.agfd = AngularBoundaryDescriptor()
        self.magp = AngularGuidancePyramid(self.out_channels)
        self.fdgpf = FrequencyAwareGeometryPrompt(
            self.out_channels,
            layout_mode=layout_mode,
            layout_state_dim=layout_state_dim,
            prompt_channels=prompt_channels,
            routing_mode=geometry_routing_mode,
            reliability_strength=reliability_strength,
            use_prompt_autocorr=use_prompt_autocorr,
            use_directional_edge_refine=use_directional_edge_refine,
        )
        self.gstf_layers = nn.ModuleDict({
            str(i): StructuralPromptFilter(self.encoder_channels[i])
            for i in self.valid_indices[:2]
        })

        if pretrained_path and pretrained_path != "imagenet":
            if not os.path.exists(pretrained_path):
                raise FileNotFoundError(f"Pretrained encoder checkpoint not found: {pretrained_path}")
            state_dict = torch.load(pretrained_path, map_location="cpu")
            state_dict = state_dict.get("state_dict", state_dict)
            torch.nn.Module.load_state_dict(self.rgb_encoder, state_dict, strict=False)
            if is_main_process():
                print(f"Loaded pretrained encoder weights from {pretrained_path}")

        self.dglsr_layers = nn.ModuleList()
        self.cs_dgam_layers = nn.ModuleList()

        num_scales = len(self.valid_indices)
        for order, i in enumerate(self.valid_indices):
            ch = self.encoder_channels[i]
            self.dglsr_layers.append(LocalSemanticFusionBlock(
                rgb_channels=ch,
                hd_channels=ch,
                aux_channels=ch,
                out_channels=ch,
                scale_index=order,
                num_scales=num_scales,
                safe_mode=safe_mode,
                max_attention_tokens=attention_tokens,
                local_scale_init=local_scale_init,
                semantic_scale_init=semantic_scale_init,
                routing_mode=geometry_routing_mode,
                reliability_strength=reliability_strength,
                fusion_branch_mode=fusion_branch_mode,
                use_grad_checkpoint=use_grad_checkpoint,
            ))

        for shallow_idx, deep_idx in zip(self.valid_indices[:-1], self.valid_indices[1:]):
            self.cs_dgam_layers.append(
                CrossScaleFeatureAligner(self.encoder_channels[shallow_idx], self.encoder_channels[deep_idx])
            )

        deepest_ch = self.encoder_channels[-1]
        self.aspp = StripPyramidASPP(deepest_ch, deepest_ch)

    def forward(self, rgb, hha, return_context=False):
        hha_confidence = 0.5 + 0.5 * self.mueb(hha)
        if self.use_prompt_recovery:
            if self.prompt_recovery_mode == "rgpr":
                hha = self.hha_recovery(hha, hha_confidence, rgb)
            else:
                hha = self.hha_recovery(hha, hha_confidence)
            hha_confidence = 0.5 + 0.5 * self.mueb(hha)
        hd = hha[:, 0:2, :, :]
        hha_guided = hha * (0.25 + 0.75 * hha_confidence)
        hd = self.bsde(hd * hha_confidence)
        angle_grad = self.agfd(hha_guided) * hha_confidence
        geometry_reliability = hha_confidence
        if self.gacr is not None:
            geometry_reliability = self.gacr(rgb, hha, angle_grad, hha_confidence)

        feats_rgb = self.rgb_encoder(rgb)
        target_sizes = [feats_rgb[i].shape[2:] for i in self.valid_indices]
        side_guides = self.magp(angle_grad, target_sizes)
        prompt_feats = self.fdgpf(hd, angle_grad, hha_confidence, target_sizes)

        feats_hd = [None for _ in feats_rgb]
        for prompt_idx, feat_idx in enumerate(self.valid_indices):
            feats_hd[feat_idx] = prompt_feats[prompt_idx]
        for guide_idx, feat_idx in enumerate(self.valid_indices[:2]):
            feats_hd[feat_idx] = self.gstf_layers[str(feat_idx)](feats_hd[feat_idx], side_guides[guide_idx])

        fused_features = []
        fusion_ptr = 0
        for i in range(len(feats_rgb)):
            if i in self.valid_indices:
                reliability = geometry_reliability if self.use_confidence_routing else None
                fused_features.append(self.dglsr_layers[fusion_ptr](
                    feats_rgb[i],
                    feats_hd[i],
                    side_guides[fusion_ptr],
                    geometry_reliability=reliability,
                ))
                fusion_ptr += 1

        fused_features[-1] = self.aspp(fused_features[-1])
        for idx in range(len(fused_features) - 2, -1, -1):
            fused_features[idx] = self.cs_dgam_layers[idx](fused_features[idx], fused_features[idx + 1])

        if return_context:
            return {
                "features": fused_features,
                "side_guides": side_guides,
                "rgb_features": feats_rgb,
                "hha_confidence": hha_confidence,
                "geometry_reliability": geometry_reliability,
                "angle_grad": angle_grad,
            }
        return fused_features


# Backward-compatible alias for existing imports and checkpoints.
DGRQEncoder = UncertaintyGuidedFusionEncoder
