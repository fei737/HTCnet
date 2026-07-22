import math

import segmentation_models_pytorch as smp
import torch
import torch.nn as nn
import torch.nn.functional as F


def sanitize_logits(logits, clamp_value=30.0):
    if logits is None:
        return None
    return torch.nan_to_num(logits.float(), nan=0.0, posinf=clamp_value, neginf=-clamp_value).clamp(
        -clamp_value,
        clamp_value,
    )


def reliability_calibration_loss(prediction, target, weight=None):
    """Weighted soft-label BCE for dense geometry reliability calibration.

    ``prediction`` is already a probability map (the estimator ends in a
    sigmoid), so it cannot be passed through ``binary_cross_entropy_with_logits``
    without changing the objective.  PyTorch disallows probability BCE under
    autocast; perform the small calibration calculation explicitly in FP32.
    """
    device_type = prediction.device.type
    with torch.autocast(device_type=device_type, enabled=False):
        if prediction.shape[2:] != target.shape[2:]:
            prediction = F.interpolate(
                prediction.float(), size=target.shape[2:], mode="bilinear", align_corners=False
            )
        prediction = prediction.float().clamp(1e-5, 1.0 - 1e-5)
        target = target.float().clamp(0.0, 1.0)
        if target.shape[1] == 1 and prediction.shape[1] > 1:
            target = target.expand(-1, prediction.shape[1], -1, -1)
        if target.shape[1] != prediction.shape[1]:
            raise ValueError(
                "Reliability target/prediction channel mismatch: "
                f"target={target.shape[1]}, prediction={prediction.shape[1]}"
            )
        loss = F.binary_cross_entropy(prediction, target, reduction="none")
        if weight is None:
            return loss.mean()
        weight = weight.float().clamp_min(0.0)
        if weight.shape[2:] != target.shape[2:]:
            weight = F.interpolate(weight, size=target.shape[2:], mode="nearest")
        if weight.shape[1] == 1 and loss.shape[1] > 1:
            weight = weight.expand(-1, loss.shape[1], -1, -1)
        return (loss * weight).sum() / weight.sum().clamp_min(1.0)


class OHEMCrossEntropyLoss(nn.Module):
    def __init__(self, thresh=0.7, ignore_index=255, min_kept=30000, n_classes=41, class_weights=None):
        super().__init__()
        self.thresh = -math.log(thresh)
        self.ignore_index = ignore_index
        self.min_kept = min_kept
        self.n_classes = n_classes
        # When fixed (dataset-level) class weights are supplied they are folded
        # directly into CrossEntropyLoss; the batch-adaptive reweighting below is
        # then skipped so rare classes are not up-weighted twice.
        self.use_external_weights = class_weights is not None
        if self.use_external_weights:
            self.register_buffer("class_weights", torch.as_tensor(class_weights, dtype=torch.float32))
        else:
            self.class_weights = None
        self.criterion = nn.CrossEntropyLoss(
            weight=self.class_weights,
            ignore_index=self.ignore_index,
            reduction="none",
        )
        self.mining_criterion = nn.CrossEntropyLoss(
            weight=None,
            ignore_index=self.ignore_index,
            reduction="none",
        )

    def forward(self, logits, labels, use_ohem=True):
        logits = sanitize_logits(logits)
        if labels.dtype != torch.long:
            labels = labels.long()

        valid_mask = (
            (labels != self.ignore_index)
            & (labels >= 0)
            & (labels < self.n_classes)
        )
        labels = labels.clone()
        labels[~valid_mask] = self.ignore_index

        loss = self.criterion(logits, labels)
        mining_loss = self.mining_criterion(logits, labels)
        if valid_mask.sum() == 0:
            return loss.sum() * 0.0

        if not use_ohem:
            return (loss * valid_mask.float()).sum() / valid_mask.sum().clamp_min(1).float()

        labels_valid = labels[valid_mask]
        if self.use_external_weights:
            # Class balancing is already baked into ``loss`` via CrossEntropyLoss
            # weights; mine hard pixels using unweighted CE so rare-class weights
            # do not consume the OHEM budget by themselves.
            weighted_loss = loss * valid_mask.float()
        else:
            class_counts = torch.bincount(labels_valid, minlength=self.n_classes).float()
            class_counts[class_counts == 0] = 1.0
            class_weights = (valid_mask.sum() / (class_counts * self.n_classes)).clamp(1.0, 5.0)

            with torch.no_grad():
                weight_map = torch.zeros_like(loss)
                weight_map[valid_mask] = class_weights[labels_valid]

            weighted_loss = loss * weight_map * valid_mask.float()
        weighted_loss_flat = weighted_loss.reshape(-1)
        mining_loss_flat = (mining_loss * valid_mask.float()).reshape(-1)
        valid_mask_flat = valid_mask.reshape(-1)

        num_valid = int(valid_mask_flat.sum().item())
        kept = min(self.min_kept, num_valid)
        if kept > 0:
            with torch.no_grad():
                sorted_loss, _ = torch.sort(mining_loss_flat.detach(), descending=True)
                threshold_value = sorted_loss[kept - 1].item()
                actual_thresh = min(self.thresh, threshold_value)

            keep_mask = (mining_loss_flat >= actual_thresh).float() * valid_mask_flat.float()
            kept_loss_sum = (weighted_loss_flat * keep_mask).sum()
            kept_loss_count = keep_mask.sum().clamp_min(1.0)
            return kept_loss_sum / kept_loss_count

        return loss.sum() * 0.0


def balanced_bce_with_logits(logits, targets, max_pos_weight=10.0):
    targets = targets.float()
    logits = torch.nan_to_num(logits.float(), nan=0.0, posinf=30.0, neginf=-30.0).clamp(-30.0, 30.0)
    targets = torch.nan_to_num(targets, nan=0.0, posinf=1.0, neginf=0.0).clamp(0.0, 1.0)
    pos = targets.sum()
    neg = targets.numel() - pos
    pos_weight = (neg / (pos + 1e-6)).clamp(min=1.0, max=max_pos_weight).detach()
    return F.binary_cross_entropy_with_logits(logits, targets, pos_weight=pos_weight)


def boundary_weighted_cross_entropy(logits, labels, edge_labels, ignore_index=255, edge_weight=2.0, class_weights=None):
    logits = sanitize_logits(logits)
    labels = labels.long()
    invalid_mask = (labels < 0) | (labels >= logits.shape[1])
    if invalid_mask.any():
        labels = labels.clone()
        labels[invalid_mask] = ignore_index
    ce = F.cross_entropy(logits, labels, weight=class_weights, ignore_index=ignore_index, reduction="none")
    valid_mask = labels != ignore_index
    weights = 1.0 + edge_weight * edge_labels.squeeze(1).float()
    weighted_loss = ce * weights * valid_mask.float()
    valid_count = valid_mask.sum().clamp_min(1).float()
    return weighted_loss.sum() / valid_count


def _resize_for_feature_precision(tensor, size, mode):
    if tensor is None or tensor.shape[2:] == size:
        return tensor
    if mode in ("nearest", "area"):
        return F.interpolate(tensor.float(), size=size, mode=mode)
    return F.interpolate(tensor.float(), size=size, mode=mode, align_corners=False)


def _feature_precision_size(height, width, max_size):
    if max_size is None or max_size <= 0:
        return height, width
    long_side = max(height, width)
    if long_side <= max_size:
        return height, width
    scale = float(max_size) / float(long_side)
    return max(1, int(round(height * scale))), max(1, int(round(width * scale)))


def probability_edge_map(logits):
    logits = torch.nan_to_num(logits.float(), nan=0.0, posinf=30.0, neginf=-30.0).clamp(-30.0, 30.0)
    probs = F.softmax(logits, dim=1)
    batch, channels, _, _ = probs.shape
    local_mean = F.avg_pool2d(probs, kernel_size=3, stride=1, padding=1)
    local_max = F.max_pool2d(probs, kernel_size=3, stride=1, padding=1)
    local_min = -F.max_pool2d(-probs, kernel_size=3, stride=1, padding=1)
    edge_prob = (0.5 * ((probs - local_mean).abs() + (local_max - local_min))).mean(dim=1, keepdim=True)
    normalizer = edge_prob.flatten(1).amax(dim=1).view(batch, 1, 1, 1).clamp_min(1e-6)
    return (edge_prob / normalizer).clamp(0.0, 1.0)


def feature_precision_loss(seg_logits, edge_labels, hha_grad_target=None, valid_mask=None, max_size=160):
    target_size = _feature_precision_size(seg_logits.shape[2], seg_logits.shape[3], max_size)
    if target_size != tuple(seg_logits.shape[2:]):
        seg_logits = F.interpolate(seg_logits, size=target_size, mode="bilinear", align_corners=False)
        edge_labels = _resize_for_feature_precision(edge_labels, target_size, mode="area")
        hha_grad_target = _resize_for_feature_precision(hha_grad_target, target_size, mode="area")
        valid_mask = _resize_for_feature_precision(valid_mask, target_size, mode="nearest")

    target = edge_labels.float()
    if hha_grad_target is not None:
        target = torch.maximum(target, hha_grad_target.float())
    if valid_mask is not None:
        target = target * valid_mask.float()

    pred_edges = probability_edge_map(seg_logits)
    if valid_mask is not None:
        pred_edges = pred_edges * valid_mask.float()

    with torch.amp.autocast(device_type=seg_logits.device.type, enabled=False):
        pred_edges = torch.nan_to_num(pred_edges.float(), nan=0.0, posinf=1.0, neginf=0.0).clamp(1e-6, 1.0 - 1e-6)
        target = torch.nan_to_num(target.float(), nan=0.0, posinf=1.0, neginf=0.0).clamp(0.0, 1.0)
        return F.binary_cross_entropy(pred_edges, target)


class RSGNetLoss(nn.Module):
    def __init__(
        self,
        lambda_lovasz=0.5,
        lambda_dice=0.4,
        lambda_edge=0.1,
        lambda_boundary=0.05,
        lambda_feature_precision=0.05,
        hha_edge_weight=0.0,
        max_edge_pos_weight=10.0,
        ohem_min_kept=30000,
        boundary_ce_weight=1.0,
        aux_weight=0.1,
        feature_precision_max_size=160,
        n_classes=41,
        ignore_index=255,
        class_weights=None,
    ):
        super().__init__()
        self.ignore_index = ignore_index
        if class_weights is not None:
            self.register_buffer("class_weights", torch.as_tensor(class_weights, dtype=torch.float32))
        else:
            self.class_weights = None
        self.ohem_ce = OHEMCrossEntropyLoss(
            thresh=0.7,
            ignore_index=ignore_index,
            min_kept=ohem_min_kept,
            n_classes=n_classes,
            class_weights=class_weights,
        )
        self.lovasz = smp.losses.LovaszLoss(mode="multiclass", ignore_index=ignore_index)
        self.dice = smp.losses.DiceLoss(mode="multiclass", ignore_index=ignore_index)
        self.lambda_lovasz = lambda_lovasz
        self.lambda_dice = lambda_dice
        self.lambda_edge = lambda_edge
        self.lambda_boundary = lambda_boundary
        self.lambda_feature_precision = lambda_feature_precision
        self.hha_edge_weight = hha_edge_weight
        self.max_edge_pos_weight = max_edge_pos_weight
        self.boundary_ce_weight = boundary_ce_weight
        self.aux_weight = aux_weight
        self.feature_precision_max_size = feature_precision_max_size

    def forward(
        self,
        seg_logits,
        seg_labels,
        edge_logits,
        edge_labels,
        aux_logits=None,
        hha_grad_target=None,
        valid_edge_mask=None,
        edge_loss_scale=1.0,
        boundary_loss_scale=1.0,
        feature_loss_scale=1.0,
        ce_only=False,
        use_ohem=True,
    ):
        seg_logits = sanitize_logits(seg_logits)
        edge_logits = sanitize_logits(edge_logits)
        aux_logits = sanitize_logits(aux_logits)

        loss_ce = self.ohem_ce(seg_logits, seg_labels, use_ohem=use_ohem)
        zero = loss_ce.detach() * 0.0
        loss_aux = 0.0
        if aux_logits is not None:
            loss_aux = self.ohem_ce(aux_logits, seg_labels, use_ohem=use_ohem) * self.aux_weight
        if ce_only:
            edge_anchor = zero
            if edge_logits is not None:
                edge_anchor = edge_logits.mean() * 0.0
            return loss_ce + loss_aux + edge_anchor, loss_ce, zero, zero, zero, zero

        loss_lov = self.lovasz(seg_logits, seg_labels) if self.lambda_lovasz > 0 else zero
        loss_dice = self.dice(seg_logits, seg_labels) if self.lambda_dice > 0 else zero
        if self.lambda_boundary > 0 and boundary_loss_scale > 0:
            loss_boundary = boundary_weighted_cross_entropy(
                seg_logits,
                seg_labels,
                edge_labels,
                ignore_index=self.ignore_index,
                edge_weight=self.boundary_ce_weight,
                class_weights=self.class_weights,
            )
        else:
            loss_boundary = zero

        if self.lambda_edge > 0 and edge_loss_scale > 0:
            edge_logits = edge_logits.squeeze(1)
            loss_edge = balanced_bce_with_logits(
                edge_logits,
                edge_labels.squeeze(1),
                max_pos_weight=self.max_edge_pos_weight,
            )
        else:
            loss_edge = zero

        if hha_grad_target is not None and self.hha_edge_weight > 0 and self.lambda_edge > 0:
            loss_edge += self.hha_edge_weight * balanced_bce_with_logits(
                edge_logits,
                hha_grad_target.squeeze(1),
                max_pos_weight=self.max_edge_pos_weight,
            )

        if self.lambda_feature_precision > 0 and feature_loss_scale > 0:
            loss_feature_precision = feature_precision_loss(
                seg_logits,
                edge_labels,
                hha_grad_target,
                valid_mask=valid_edge_mask,
                max_size=self.feature_precision_max_size,
            )
        else:
            loss_feature_precision = zero

        total_loss = (
            loss_ce
            + self.lambda_lovasz * loss_lov
            + self.lambda_dice * loss_dice
            + self.lambda_edge * edge_loss_scale * loss_edge
            + self.lambda_boundary * boundary_loss_scale * loss_boundary
            + self.lambda_feature_precision * feature_loss_scale * loss_feature_precision
            + loss_aux
        )
        return total_loss, loss_ce, loss_lov, loss_dice, loss_edge, loss_feature_precision


def edge_target_from_mask(masks, ignore_index=255):
    masks = masks.long().clone()
    masks[masks < 0] = ignore_index
    valid = masks != ignore_index

    def _shift_with_ignore(x, dy, dx):
        shifted = torch.full_like(x, ignore_index)
        h_slice_src = slice(max(-dy, 0), x.shape[1] - max(dy, 0))
        w_slice_src = slice(max(-dx, 0), x.shape[2] - max(dx, 0))
        h_slice_dst = slice(max(dy, 0), x.shape[1] - max(-dy, 0))
        w_slice_dst = slice(max(dx, 0), x.shape[2] - max(-dx, 0))
        shifted[:, h_slice_dst, w_slice_dst] = x[:, h_slice_src, w_slice_src]
        return shifted

    boundary = torch.zeros_like(masks, dtype=torch.bool)
    neighbor_offsets = [(-1, 0), (1, 0), (0, -1), (0, 1), (-1, -1), (-1, 1), (1, -1), (1, 1)]
    for dy, dx in neighbor_offsets:
        neighbor = _shift_with_ignore(masks, dy, dx)
        neighbor_valid = neighbor != ignore_index
        boundary |= valid & neighbor_valid & (neighbor != masks)

    thin_edge = boundary.float().unsqueeze(1)
    edge_halo = F.max_pool2d(thin_edge, kernel_size=3, stride=1, padding=1) - thin_edge
    edge_target = (thin_edge + 0.25 * edge_halo).clamp(0.0, 1.0)
    return edge_target * valid.unsqueeze(1).float()
