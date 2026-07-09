import torch
import torch.nn as nn
import torch.nn.functional as F

from encoder.common import ConvBNAct


class LightweightMLPDecoder(nn.Module):
    """Lightweight SegFormer-style decoder for decoder ablation."""

    def __init__(self, in_channels_list, n_classes, decoder_channels=256):
        super().__init__()
        self.lateral_convs = nn.ModuleList([
            ConvBNAct(in_channels, decoder_channels, kernel_size=1)
            for in_channels in in_channels_list
        ])
        self.fusion = nn.Sequential(
            ConvBNAct(decoder_channels * len(in_channels_list), decoder_channels, kernel_size=1),
            ConvBNAct(decoder_channels, decoder_channels, kernel_size=3, activation=nn.GELU),
        )
        self.classifier = nn.Conv2d(decoder_channels, n_classes, kernel_size=1)
        self.aux_classifier = nn.Conv2d(decoder_channels, n_classes, kernel_size=1)
        self.edge_head = nn.Sequential(
            ConvBNAct(decoder_channels, max(decoder_channels // 2, 32), kernel_size=3, activation=nn.GELU),
            nn.Conv2d(max(decoder_channels // 2, 32), 1, kernel_size=1),
        )
        for head in (self.classifier, self.aux_classifier):
            nn.init.normal_(head.weight, mean=0.0, std=0.01)
            if head.bias is not None:
                nn.init.zeros_(head.bias)

    def forward(self, features, output_size):
        laterals = [proj(feat) for proj, feat in zip(self.lateral_convs, features)]
        target_size = laterals[0].shape[2:]
        fused = torch.cat([
            feat if feat.shape[2:] == target_size
            else F.interpolate(feat, size=target_size, mode="bilinear", align_corners=False)
            for feat in laterals
        ], dim=1)
        detail_feat = self.fusion(fused)
        seg_logits = self.classifier(detail_feat)
        edge_logits = self.edge_head(detail_feat)
        aux_source = laterals[1] if len(laterals) > 1 else detail_feat
        aux_logits = self.aux_classifier(aux_source)

        seg_logits = F.interpolate(seg_logits, size=output_size, mode="bilinear", align_corners=False)
        edge_logits = F.interpolate(edge_logits, size=output_size, mode="bilinear", align_corners=False)
        aux_logits = F.interpolate(aux_logits, size=output_size, mode="bilinear", align_corners=False)
        return seg_logits, edge_logits, aux_logits


# Backward-compatible alias for existing configs and scripts.
SimpleMLPFusionDecoder = LightweightMLPDecoder
