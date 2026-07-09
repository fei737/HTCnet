import torch
import torch.nn as nn
import torch.nn.functional as F

from decoder import (
    BoundaryDetailHead,
    EfficientUpsamplingHead,
    LightweightMLPDecoder,
    QueryGuidedFusionDecoder,
    SemanticBoundaryRefinement,
)
from encoder import (
    FixedBoundaryExtractor,
    FrequencyAwareGeometryPrompt,
    FrequencyDisentangledGeometricPromptField,
    UncertaintyGuidedFusionEncoder,
)


class UGFLiteNet(nn.Module):
    """UGF-Lite network with reliability-guided encoder fusion and selectable decoder."""

    def __init__(self, n_classes=14, pretrained_path=None, return_aux=False, encoder_name="mit_b2", safe_mode=False,
                 layout_mode="ssm", drop_path_rate=0.0, use_prompt_recovery=True, prompt_recovery_mode="rgpr",
                 use_confidence_routing=True, consistency_routing_mode="gacr",
                 prompt_channels=32, layout_state_dim=16, geometry_routing_mode="rcfr",
                 reliability_strength=0.30, decoder_channels=256, attention_tokens=1024,
                 local_scale_init=1e-4, semantic_scale_init=1e-4, query_points=4,
                 query_scale_init=0.05, geometry_scale_init=0.1,
                 detail_logit_scale=0.3, rgb_boundary_scale=0.1, use_grad_checkpoint=False,
                 decoder_mode="simple_mlp", fusion_branch_mode="both", use_prompt_autocorr=True,
                 use_directional_edge_refine=True):
        super().__init__()
        self.return_aux = return_aux
        self.decoder_mode = decoder_mode
        self.detail_logit_scale = float(detail_logit_scale)
        self.rgb_boundary_scale = float(rgb_boundary_scale)
        if decoder_mode not in {"qdhs", "simple_mlp"}:
            raise ValueError(f"Unsupported decoder_mode: {decoder_mode}")

        self.encoder = UncertaintyGuidedFusionEncoder(
            pretrained_path=pretrained_path,
            encoder_name=encoder_name,
            safe_mode=safe_mode,
            layout_mode=layout_mode,
            drop_path_rate=drop_path_rate,
            use_prompt_recovery=use_prompt_recovery,
            prompt_recovery_mode=prompt_recovery_mode,
            use_confidence_routing=use_confidence_routing,
            consistency_routing_mode=consistency_routing_mode,
            prompt_channels=prompt_channels,
            layout_state_dim=layout_state_dim,
            geometry_routing_mode=geometry_routing_mode,
            reliability_strength=reliability_strength,
            attention_tokens=attention_tokens,
            local_scale_init=local_scale_init,
            semantic_scale_init=semantic_scale_init,
            fusion_branch_mode=fusion_branch_mode,
            use_prompt_autocorr=use_prompt_autocorr,
            use_directional_edge_refine=use_directional_edge_refine,
            use_grad_checkpoint=use_grad_checkpoint,
        )

        decoder_in_channels = self.encoder.out_channels
        if decoder_mode == "simple_mlp":
            self.simple_decoder = LightweightMLPDecoder(
                in_channels_list=decoder_in_channels,
                n_classes=n_classes,
                decoder_channels=decoder_channels,
            )
        else:
            self.qdhs_decoder = QueryGuidedFusionDecoder(
                in_channels_list=decoder_in_channels,
                n_classes=n_classes,
                decoder_channels=decoder_channels,
                query_points=query_points,
                query_scale_init=query_scale_init,
                geometry_scale_init=geometry_scale_init,
            )
            self.msde_bdn = BoundaryDetailHead(
                detail_channels=decoder_channels,
                shallow_channels=decoder_in_channels[0],
                guide_channels=decoder_in_channels[0],
                n_classes=n_classes,
            )
            self.rgb_high_freq_guide = FixedBoundaryExtractor(decoder_channels)
            self.se_cbrl = SemanticBoundaryRefinement(n_classes, boundary_channels=decoder_channels)
            self.cerrh_seg = EfficientUpsamplingHead(
                in_channels=n_classes,
                out_channels=n_classes,
                scale_factor=4,
            )
            self.cerrh_edge = EfficientUpsamplingHead(
                in_channels=1,
                out_channels=1,
                scale_factor=4,
            )

    # Compatibility properties for existing hooks, diagnostics, and utilities.
    @property
    def rgb_encoder(self):
        return self.encoder.rgb_encoder

    @property
    def mueb(self):
        return self.encoder.mueb

    @property
    def bsde(self):
        return self.encoder.bsde

    @property
    def agfd(self):
        return self.encoder.agfd

    @property
    def magp(self):
        return self.encoder.magp

    @property
    def fdgpf(self):
        return self.encoder.fdgpf

    @property
    def dglsr_layers(self):
        return self.encoder.dglsr_layers

    @property
    def aspp(self):
        return self.encoder.aspp

    def encode(self, rgb, hha, return_context=False):
        """Return decoder-ready multi-scale fused features.

        For ``mit_b2`` at 480x640, ``features`` typically has shapes:
        [B, 64, 120, 160], [B, 128, 60, 80], [B, 320, 30, 40],
        [B, 512, 15, 20].
        """
        return self.encoder(rgb, hha, return_context=return_context)

    def forward(self, rgb, hha):
        encoded = self.encode(rgb, hha, return_context=self.decoder_mode == "qdhs")
        if self.decoder_mode == "qdhs":
            fused_list = encoded["features"]
            side_guides = encoded["side_guides"]
            feats_rgb = encoded["rgb_features"]
        else:
            fused_list = encoded

        if self.decoder_mode == "simple_mlp":
            seg_logits, edge_logits, aux_seg_logits = self.simple_decoder(
                [f.contiguous() for f in fused_list],
                output_size=rgb.shape[2:],
            )
            if self.return_aux:
                return seg_logits, edge_logits, aux_seg_logits
            return seg_logits

        seg_base, detail_feat, aux_seg_base = self.qdhs_decoder([f.contiguous() for f in fused_list])
        detail_feat, edge_feat, detail_logits, boundary_feat = self.msde_bdn(
            detail_feat,
            fused_list[0],
            side_guides[0],
            seg_base,
        )
        detail_logits = F.interpolate(detail_logits, size=seg_base.shape[2:], mode="bilinear", align_corners=False)
        edge_feat_120 = F.interpolate(edge_feat, size=seg_base.shape[2:], mode="bilinear", align_corners=False)
        edge_feat_120 = torch.nan_to_num(edge_feat_120, nan=0.0, posinf=30.0, neginf=-30.0).clamp(-30.0, 30.0)
        edge_prob_120 = torch.sigmoid(edge_feat_120)
        seg_base = seg_base + self.detail_logit_scale * detail_logits * (1.0 + edge_prob_120)

        if boundary_feat.shape[2:] != seg_base.shape[2:]:
            boundary_feat = F.interpolate(boundary_feat, size=seg_base.shape[2:], mode="bilinear", align_corners=False)
        rgb_high_freq = self.rgb_high_freq_guide(feats_rgb[self.encoder.valid_indices[0]])
        if rgb_high_freq.shape[2:] != boundary_feat.shape[2:]:
            rgb_high_freq = F.interpolate(rgb_high_freq, size=boundary_feat.shape[2:], mode="bilinear", align_corners=False)
        boundary_feat = boundary_feat + self.rgb_boundary_scale * rgb_high_freq

        seg_refined = self.se_cbrl(seg_base, edge_prob_120, boundary_feat).contiguous()
        seg_logits = self.cerrh_seg(seg_refined)
        edge_logits = self.cerrh_edge(edge_feat_120.contiguous())

        if self.return_aux:
            aux_seg_logits = F.interpolate(aux_seg_base, size=rgb.shape[2:], mode="bilinear", align_corners=False)
            return seg_logits, edge_logits, aux_seg_logits
        return seg_logits


DGRQNet = UGFLiteNet


__all__ = [
    "UGFLiteNet",
    "DGRQNet",
    "FrequencyAwareGeometryPrompt",
    "FrequencyDisentangledGeometricPromptField",
    "remap_dgrq_state_dict",
]


STATE_DICT_KEY_RENAMES = (
    ("hha_confidence.", "encoder.mueb."),
    ("hd_enhancer.", "encoder.bsde."),
    ("angle_grad.", "encoder.agfd."),
    ("side_guide.", "encoder.magp."),
    ("hha_prompt_encoder.", "encoder.fdgpf."),
    ("gspg.", "encoder.fdgpf."),
    ("hd_structure_filters.", "encoder.gstf_layers."),
    ("fusion_layers.", "encoder.dglsr_layers."),
    ("inter_layer_refiners.", "encoder.cs_dgam_layers."),
    ("rgb_encoder.", "encoder.rgb_encoder."),
    ("mueb.", "encoder.mueb."),
    ("hha_recovery.", "encoder.hha_recovery."),
    ("bsde.", "encoder.bsde."),
    ("agfd.", "encoder.agfd."),
    ("magp.", "encoder.magp."),
    ("fdgpf.", "encoder.fdgpf."),
    ("gstf_layers.", "encoder.gstf_layers."),
    ("dglsr_layers.", "encoder.dglsr_layers."),
    ("cs_dgam_layers.", "encoder.cs_dgam_layers."),
    ("aspp.", "encoder.aspp."),
    ("decoder.", "qdhs_decoder."),
    ("detail_edge_head.", "msde_bdn."),
    ("gated_edge.", "se_cbrl."),
    ("carafe_seg.", "cerrh_seg."),
    ("carafe_edge.", "cerrh_edge."),
)


def remap_dgrq_state_dict(state_dict):
    remapped = {}
    for key, value in state_dict.items():
        new_key = key
        for old_prefix, new_prefix in STATE_DICT_KEY_RENAMES:
            if key.startswith(old_prefix):
                new_key = new_prefix + key[len(old_prefix):]
                break
        remapped[new_key] = value
    return remapped
