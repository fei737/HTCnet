import torch
import torch.nn as nn
import torch.nn.functional as F

from encoder.common import ConvBNAct, ConvGNAct


class MultiScaleSparseDeformableObjectQueryAttention(nn.Module):
    def __init__(self, dim, num_levels, num_points=4):
        super().__init__()
        self.dim = dim
        self.num_levels = num_levels
        self.num_points = num_points
        self.level_embed = nn.Parameter(torch.zeros(num_levels, dim))
        self.reference_mlp = nn.Sequential(
            nn.Linear(dim, dim),
            nn.GELU(),
            nn.Linear(dim, num_levels * 2),
        )
        self.offset_mlp = nn.Sequential(
            nn.Linear(dim, dim),
            nn.GELU(),
            nn.Linear(dim, num_levels * num_points * 2),
        )
        self.weight_mlp = nn.Linear(dim, num_levels * num_points)
        self.out_proj = nn.Linear(dim, dim)
        nn.init.normal_(self.level_embed, std=0.02)
        nn.init.zeros_(self.offset_mlp[-1].weight)
        nn.init.zeros_(self.offset_mlp[-1].bias)

    @staticmethod
    def _sine_position(dim, h, w, device, dtype):
        quarter_dim = max(dim // 4, 1)
        y = torch.linspace(0.0, 1.0, h, device=device, dtype=dtype)
        x = torch.linspace(0.0, 1.0, w, device=device, dtype=dtype)
        yy, xx = torch.meshgrid(y, x, indexing="ij")
        omega = torch.arange(quarter_dim, device=device, dtype=dtype)
        omega = 1.0 / (10000 ** (omega / max(quarter_dim - 1, 1)))
        pos = torch.cat([
            torch.sin(xx[..., None] * omega),
            torch.cos(xx[..., None] * omega),
            torch.sin(yy[..., None] * omega),
            torch.cos(yy[..., None] * omega),
        ], dim=-1)
        if pos.shape[-1] < dim:
            pos = F.pad(pos, (0, dim - pos.shape[-1]))
        elif pos.shape[-1] > dim:
            pos = pos[..., :dim]
        return pos.permute(2, 0, 1).unsqueeze(0)

    def forward(self, queries, features):
        b, q, c = queries.shape
        references = self.reference_mlp(queries).view(b, q, self.num_levels, 1, 2).sigmoid()
        offsets = self.offset_mlp(queries).view(b, q, self.num_levels, self.num_points, 2)
        offsets = 0.5 * torch.tanh(offsets)
        weights = self.weight_mlp(queries).view(b, q, self.num_levels, self.num_points)
        weights = torch.softmax(weights.flatten(2), dim=-1).view(b, q, self.num_levels, self.num_points)

        sampled_per_level = []
        for level, feat in enumerate(features):
            _, _, h, w = feat.shape
            level_bias = self.level_embed[level].view(1, c, 1, 1).to(dtype=feat.dtype, device=feat.device)
            pos = self._sine_position(c, h, w, feat.device, feat.dtype)
            feat = feat + level_bias + pos
            coords = (references[:, :, level] + offsets[:, :, level]).clamp(0.0, 1.0)
            grid = coords.mul(2.0).sub(1.0).view(b, q * self.num_points, 1, 2)
            sampled = F.grid_sample(feat, grid, mode="bilinear", padding_mode="border", align_corners=False)
            sampled = sampled.squeeze(-1).transpose(1, 2).view(b, q, self.num_points, c)
            sampled_per_level.append(sampled)

        sampled = torch.stack(sampled_per_level, dim=2)
        context = (sampled * weights.unsqueeze(-1)).sum(dim=(2, 3))
        return self.out_proj(context)


class QueryGuidedFusionDecoder(nn.Module):
    def __init__(
        self,
        in_channels_list,
        n_classes,
        decoder_channels=256,
        query_heads=8,
        ppm_scales=(1, 2, 3, 6),
        query_points=4,
        query_scale_init=0.05,
        geometry_scale_init=0.1,
    ):
        super().__init__()
        self.n_classes = n_classes
        self.decoder_channels = decoder_channels
        self.lateral_convs = nn.ModuleList([
            ConvBNAct(in_channels, decoder_channels, kernel_size=1)
            for in_channels in in_channels_list
        ])
        self.ppm_stages = nn.ModuleList([
            nn.Sequential(
                nn.AdaptiveAvgPool2d(scale),
                ConvGNAct(decoder_channels, decoder_channels, kernel_size=1),
            )
            for scale in ppm_scales
        ])
        self.ppm_bottleneck = ConvBNAct(
            decoder_channels * (len(ppm_scales) + 1),
            decoder_channels,
            kernel_size=3,
        )
        self.fpn_blocks = nn.ModuleList([
            ConvBNAct(decoder_channels, decoder_channels, kernel_size=3)
            for _ in range(max(len(in_channels_list) - 1, 0))
        ])
        self.fusion = ConvBNAct(decoder_channels * len(in_channels_list), decoder_channels, kernel_size=1)
        self.detail_refine = nn.Sequential(
            ConvBNAct(decoder_channels, decoder_channels, kernel_size=3, groups=decoder_channels, activation=nn.GELU),
            ConvBNAct(decoder_channels, decoder_channels, kernel_size=1, activation=nn.GELU),
        )
        self.memory_projs = nn.ModuleList([
            ConvBNAct(decoder_channels, decoder_channels, kernel_size=1)
            for _ in in_channels_list
        ])
        self.query_embed = nn.Embedding(n_classes, decoder_channels)
        self.dynamic_query_mlp = nn.Sequential(
            nn.Linear(decoder_channels, decoder_channels * 2),
            nn.GELU(),
            nn.Linear(decoder_channels * 2, n_classes * decoder_channels),
        )
        self.query_norm = nn.LayerNorm(decoder_channels)
        self.multiscale_query_attn = MultiScaleSparseDeformableObjectQueryAttention(
            decoder_channels,
            num_levels=len(in_channels_list),
            num_points=query_points,
        )
        self.query_ffn = nn.Sequential(
            nn.Linear(decoder_channels, decoder_channels * 4),
            nn.GELU(),
            nn.Linear(decoder_channels * 4, decoder_channels),
        )
        self.mask_proj = nn.Conv2d(decoder_channels, decoder_channels, kernel_size=1, bias=False)
        self.semantic_stream = nn.Sequential(
            ConvBNAct(decoder_channels, decoder_channels, kernel_size=3, activation=nn.GELU),
            ConvBNAct(decoder_channels, decoder_channels, kernel_size=1, activation=nn.GELU),
        )
        self.geometry_stream = nn.Sequential(
            ConvBNAct(decoder_channels, decoder_channels, kernel_size=3, groups=decoder_channels, activation=nn.GELU),
            ConvBNAct(decoder_channels, decoder_channels, kernel_size=1, activation=nn.GELU),
        )
        self.classifier = nn.Conv2d(decoder_channels, n_classes, kernel_size=1)
        self.geometry_classifier = nn.Conv2d(decoder_channels, n_classes, kernel_size=1)
        self.reconstruction_gate = nn.Sequential(
            nn.Conv2d(decoder_channels * 2, decoder_channels, kernel_size=1, bias=False),
            nn.BatchNorm2d(decoder_channels),
            nn.GELU(),
            nn.Conv2d(decoder_channels, n_classes, kernel_size=1),
            nn.Sigmoid(),
        )
        self.aux_classifier = nn.Conv2d(decoder_channels, n_classes, kernel_size=1)
        self.query_scale = nn.Parameter(torch.tensor(float(query_scale_init)))
        self.geometry_scale = nn.Parameter(torch.tensor(float(geometry_scale_init)))
        for head in (self.classifier, self.geometry_classifier, self.aux_classifier):
            nn.init.normal_(head.weight, mean=0.0, std=0.01)
            if head.bias is not None:
                nn.init.zeros_(head.bias)

    def _ppm_forward(self, x):
        ppm_outs = [x]
        for stage in self.ppm_stages:
            pooled = stage(x)
            pooled = F.interpolate(pooled, size=x.shape[2:], mode="bilinear", align_corners=False)
            ppm_outs.append(pooled)
        return self.ppm_bottleneck(torch.cat(ppm_outs, dim=1))

    def forward(self, features):
        laterals = [proj(feat) for proj, feat in zip(self.lateral_convs, features)]
        laterals[-1] = self._ppm_forward(laterals[-1])

        for idx in range(len(laterals) - 2, -1, -1):
            laterals[idx] = laterals[idx] + F.interpolate(
                laterals[idx + 1],
                size=laterals[idx].shape[2:],
                mode="bilinear",
                align_corners=False,
            )
            laterals[idx] = self.fpn_blocks[idx](laterals[idx])

        target_size = laterals[0].shape[2:]
        pyramid = torch.cat([
            feat if feat.shape[2:] == target_size
            else F.interpolate(feat, size=target_size, mode="bilinear", align_corners=False)
            for feat in laterals
        ], dim=1)
        detail_feat = self.detail_refine(self.fusion(pyramid))
        semantic_feat = self.semantic_stream(detail_feat)
        geometry_feat = self.geometry_stream(detail_feat)

        scene_vector = F.adaptive_avg_pool2d(semantic_feat + geometry_feat, 1).flatten(1)
        global_token = scene_vector.unsqueeze(1)
        dynamic_queries = self.dynamic_query_mlp(scene_vector).view(
            detail_feat.size(0),
            self.n_classes,
            self.decoder_channels,
        )
        static_queries = self.query_embed.weight.unsqueeze(0).expand(detail_feat.size(0), -1, -1)
        queries = static_queries + dynamic_queries + global_token

        memory_features = [proj(feat) for proj, feat in zip(self.memory_projs, laterals)]
        attn_out = self.multiscale_query_attn(self.query_norm(queries), memory_features)
        queries = queries + attn_out
        queries = queries + self.query_ffn(self.query_norm(queries))

        pixel_feat = F.normalize(self.mask_proj(semantic_feat), dim=1)
        query_feat = F.normalize(queries, dim=-1)
        mask_logits = torch.einsum("bqc,bchw->bqhw", query_feat, pixel_feat)

        semantic_logits = self.classifier(semantic_feat) + self.query_scale * mask_logits
        geometry_logits = self.geometry_classifier(geometry_feat)
        geometry_gate = self.reconstruction_gate(torch.cat([semantic_feat, geometry_feat], dim=1))
        seg_logits = semantic_logits * (1.0 + self.geometry_scale * geometry_gate) + self.geometry_scale * geometry_logits
        aux_source = laterals[1] if len(laterals) > 1 else laterals[0]
        aux_logits = self.aux_classifier(aux_source)
        return seg_logits, detail_feat, aux_logits


# Backward-compatible alias for existing configs and scripts.
QueryDrivenHierarchicalSpatialFusionDecoder = QueryGuidedFusionDecoder
