import torch
import torch.nn as nn
import torch.nn.functional as F

from encoder.common import ConvBNAct, FixedBoundaryExtractor, local_contrast


class SemanticBoundaryRefinement(nn.Module):
    def __init__(self, n_classes, boundary_channels):
        super().__init__()
        self.semantic_proj = ConvBNAct(n_classes, n_classes, kernel_size=3, activation=nn.GELU)
        self.edge_proj = nn.Sequential(
            ConvBNAct(2, max(n_classes, 16), kernel_size=3, activation=nn.GELU),
            nn.Conv2d(max(n_classes, 16), n_classes, kernel_size=1),
            nn.Sigmoid(),
        )
        self.boundary_proj = nn.Sequential(
            ConvBNAct(boundary_channels, n_classes, kernel_size=1, activation=nn.GELU),
            nn.Conv2d(n_classes, n_classes, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(n_classes),
        )
        self.fusion = nn.Sequential(
            ConvBNAct(n_classes * 3, n_classes, kernel_size=1, activation=nn.GELU),
            nn.Dropout2d(0.1),
            nn.Conv2d(n_classes, n_classes, kernel_size=1),
        )

    @staticmethod
    def _seg_boundary_map(seg_logits):
        probs = F.softmax(seg_logits, dim=1)
        boundary_map = local_contrast(probs).mean(dim=1, keepdim=True)
        normalizer = boundary_map.flatten(1).amax(dim=1).view(-1, 1, 1, 1).clamp_min(1e-6)
        return boundary_map / normalizer

    def forward(self, seg_base, edge_prob, boundary_feat):
        seg_boundary = self._seg_boundary_map(seg_base)
        edge_gate = self.edge_proj(torch.cat([edge_prob, seg_boundary], dim=1))
        seg_feat = self.semantic_proj(seg_base)
        boundary_bias = self.boundary_proj(boundary_feat)
        edge_activated_seg = (seg_feat + boundary_bias) * (1.0 + edge_gate)
        out = self.fusion(torch.cat([seg_base, edge_activated_seg, boundary_bias], dim=1))
        return out + seg_base


class EfficientUpsamplingHead(nn.Module):
    def __init__(self, in_channels, out_channels, scale_factor=4):
        super().__init__()
        self.scale_factor = scale_factor
        self.proj = nn.Conv2d(in_channels, out_channels, kernel_size=1) if in_channels != out_channels else nn.Identity()
        self.refine = nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1, groups=out_channels)

    def forward(self, x):
        x = self.proj(x)
        x = x + self.refine(x)
        return F.interpolate(x, scale_factor=self.scale_factor, mode="bilinear", align_corners=False)


class BoundaryDetailHead(nn.Module):
    def __init__(self, detail_channels, shallow_channels, guide_channels, n_classes):
        super().__init__()
        mid_channels = max(detail_channels // 4, 32)
        self.n_classes = n_classes
        self.shallow_proj = nn.Sequential(
            ConvBNAct(shallow_channels, mid_channels, kernel_size=1, activation=nn.GELU),
            ConvBNAct(mid_channels, mid_channels, kernel_size=3, activation=nn.GELU),
        )
        self.guide_proj = nn.Sequential(
            ConvBNAct(guide_channels, mid_channels, kernel_size=1, activation=nn.GELU),
            ConvBNAct(mid_channels, mid_channels, kernel_size=3, activation=nn.GELU),
        )
        self.detail_proj = ConvBNAct(detail_channels, mid_channels, kernel_size=1, activation=nn.GELU)
        self.semantic_proj = nn.Sequential(
            ConvBNAct(1, mid_channels, kernel_size=3, activation=nn.GELU),
            ConvBNAct(mid_channels, mid_channels, kernel_size=3, activation=nn.GELU),
        )
        self.shallow_boundary = FixedBoundaryExtractor(mid_channels)
        self.detail_boundary = FixedBoundaryExtractor(mid_channels)
        self.boundary_fuse = nn.Sequential(
            ConvBNAct(mid_channels * 5, detail_channels, kernel_size=3, activation=nn.GELU),
            ConvBNAct(detail_channels, detail_channels, kernel_size=3, groups=detail_channels, activation=nn.GELU),
            nn.Conv2d(detail_channels, detail_channels, kernel_size=1, bias=False),
            nn.BatchNorm2d(detail_channels),
        )
        self.boundary_gate = nn.Sequential(
            nn.Conv2d(detail_channels, mid_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(mid_channels),
            nn.GELU(),
            nn.Conv2d(mid_channels, 1, kernel_size=1),
            nn.Sigmoid(),
        )
        self.detail_logit_proj = nn.Conv2d(detail_channels, n_classes, kernel_size=1, bias=False)
        self.edge_head = nn.Sequential(
            ConvBNAct(detail_channels * 2 + 1, 128, kernel_size=3, activation=nn.GELU),
            ConvBNAct(128, 64, kernel_size=3, activation=nn.GELU),
            nn.Conv2d(64, 1, kernel_size=1),
        )

    @staticmethod
    def _seg_boundary_map(seg_logits):
        probs = F.softmax(seg_logits, dim=1)
        boundary_map = local_contrast(probs).mean(dim=1, keepdim=True)
        normalizer = boundary_map.flatten(1).amax(dim=1).view(-1, 1, 1, 1).clamp_min(1e-6)
        return boundary_map / normalizer

    def forward(self, detail_feat, shallow_feat, guide_feat, seg_logits):
        shallow_feat = self.shallow_proj(shallow_feat)
        guide_feat = self.guide_proj(guide_feat)
        if shallow_feat.shape[2:] != detail_feat.shape[2:]:
            shallow_feat = F.interpolate(shallow_feat, size=detail_feat.shape[2:], mode="bilinear", align_corners=False)
        if guide_feat.shape[2:] != detail_feat.shape[2:]:
            guide_feat = F.interpolate(guide_feat, size=detail_feat.shape[2:], mode="bilinear", align_corners=False)

        detail_context = self.detail_proj(detail_feat)
        semantic_boundary = self._seg_boundary_map(seg_logits.detach())
        if semantic_boundary.shape[2:] != detail_feat.shape[2:]:
            semantic_boundary = F.interpolate(semantic_boundary, size=detail_feat.shape[2:], mode="bilinear", align_corners=False)
        semantic_feat = self.semantic_proj(semantic_boundary)
        shallow_boundary = self.shallow_boundary(shallow_feat) * (0.2 + semantic_boundary)
        detail_boundary = self.detail_boundary(detail_context) * (0.4 + semantic_boundary)
        guide_feat = guide_feat * (0.3 + semantic_boundary)
        boundary_feat = self.boundary_fuse(torch.cat([
            detail_context,
            shallow_boundary,
            detail_boundary,
            guide_feat,
            semantic_feat,
        ], dim=1))
        boundary_gate = self.boundary_gate(boundary_feat)
        detail_feat = detail_feat + boundary_feat * (0.15 + 0.85 * boundary_gate) * (0.25 + semantic_boundary)
        detail_logits = self.detail_logit_proj(detail_feat)
        semantic_logit = torch.logit(semantic_boundary.clamp(1e-4, 1.0 - 1e-4))
        edge_residual = self.edge_head(torch.cat([detail_feat, boundary_feat, semantic_boundary], dim=1))
        edge_logits = semantic_logit + 0.35 * edge_residual
        return detail_feat, edge_logits, detail_logits, boundary_feat


# Backward-compatible aliases for existing configs and scripts.
SemanticEdgeCoGatedBoundaryRefinementLayer = SemanticBoundaryRefinement
ComputationalEfficientResolutionRefinementHead = EfficientUpsamplingHead
MultiSourceDetailEnhancedBoundaryDiscriminationNetwork = BoundaryDetailHead
