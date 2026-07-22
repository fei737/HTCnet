import torch
import torch.nn as nn
import torch.nn.functional as F

from .encoder.layers import ConvBNAct


class LightweightSegmentationDecoder(nn.Module):
    """SegFormer-style decoder with auxiliary segmentation and edge heads."""

    def __init__(self, in_channels, n_classes, decoder_channels=256):
        super().__init__()
        self.lateral_projections = nn.ModuleList(
            [
                ConvBNAct(channels, decoder_channels, kernel_size=1)
                for channels in in_channels
            ]
        )
        self.fusion = nn.Sequential(
            ConvBNAct(decoder_channels * len(in_channels), decoder_channels, kernel_size=1),
            ConvBNAct(
                decoder_channels,
                decoder_channels,
                kernel_size=3,
                activation=nn.GELU,
            ),
        )
        self.segmentation_head = nn.Conv2d(decoder_channels, n_classes, kernel_size=1)
        self.auxiliary_head = nn.Conv2d(decoder_channels, n_classes, kernel_size=1)
        edge_channels = max(decoder_channels // 2, 32)
        self.edge_head = nn.Sequential(
            ConvBNAct(
                decoder_channels,
                edge_channels,
                kernel_size=3,
                activation=nn.GELU,
            ),
            nn.Conv2d(edge_channels, 1, kernel_size=1),
        )
        for head in (self.segmentation_head, self.auxiliary_head):
            nn.init.normal_(head.weight, mean=0.0, std=0.01)
            if head.bias is not None:
                nn.init.zeros_(head.bias)

    def forward(self, features, output_size):
        lateral_features = [
            projection(feature)
            for projection, feature in zip(self.lateral_projections, features)
        ]
        target_size = lateral_features[0].shape[2:]
        fused = torch.cat(
            [
                feature
                if feature.shape[2:] == target_size
                else F.interpolate(
                    feature,
                    size=target_size,
                    mode="bilinear",
                    align_corners=False,
                )
                for feature in lateral_features
            ],
            dim=1,
        )
        decoded = self.fusion(fused)
        logits = (
            self.segmentation_head(decoded),
            self.edge_head(decoded),
            self.auxiliary_head(
                lateral_features[1] if len(lateral_features) > 1 else decoded
            ),
        )
        return tuple(
            F.interpolate(item, size=output_size, mode="bilinear", align_corners=False)
            for item in logits
        )
