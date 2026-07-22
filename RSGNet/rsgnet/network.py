import torch.nn as nn

from .decoder import LightweightSegmentationDecoder
from .encoder import StageAwareGeometryEncoder


class RSGNet(nn.Module):
    """Reliability-aware Stage-wise Geometry-guided Network."""

    def __init__(
        self,
        n_classes=37,
        pretrained_path=None,
        return_aux=False,
        encoder_name="mit_b2",
        drop_path_rate=0.0,
        prompt_channels=32,
        decoder_channels=256,
        attention_tokens=512,
        local_scale_init=0.05,
        semantic_scale_init=0.05,
        fusion_mode="stagewise",
        use_reliability=True,
        architecture_variant="legacy",
        geometry_encoding="factorized_routed",
        geometry_channels="dha",
    ):
        super().__init__()
        self.return_aux = bool(return_aux)
        self.encoder = StageAwareGeometryEncoder(
            pretrained_path=pretrained_path,
            encoder_name=encoder_name,
            drop_path_rate=drop_path_rate,
            prompt_channels=prompt_channels,
            max_attention_tokens=attention_tokens,
            local_scale_init=local_scale_init,
            semantic_scale_init=semantic_scale_init,
            fusion_mode=fusion_mode,
            use_reliability=use_reliability,
            architecture_variant=architecture_variant,
            geometry_encoding=geometry_encoding,
            geometry_channels=geometry_channels,
        )
        self.decoder = LightweightSegmentationDecoder(
            in_channels=self.encoder.out_channels,
            n_classes=n_classes,
            decoder_channels=decoder_channels,
        )

    def encode(self, rgb, hha, return_context=False, hha_invalid_hint=None):
        return self.encoder(
            rgb,
            hha,
            return_context=return_context,
            hha_invalid_hint=hha_invalid_hint,
        )

    def forward(self, rgb, hha, return_context=False, hha_invalid_hint=None):
        encoded = self.encode(
            rgb,
            hha,
            return_context=return_context,
            hha_invalid_hint=hha_invalid_hint,
        )
        features = encoded["features"] if return_context else encoded
        segmentation, edges, auxiliary = self.decoder(
            features, output_size=rgb.shape[2:]
        )
        if self.return_aux:
            if return_context:
                return segmentation, edges, auxiliary, encoded
            return segmentation, edges, auxiliary
        if return_context:
            return segmentation, encoded
        return segmentation
