import logging
import hashlib
import os
import random
import sys

import cv2
import json
import numpy as np
import torch
import torch.distributed as dist
import torch.nn as nn
import torch.nn.functional as F
from torch.optim.swa_utils import AveragedModel
from tqdm import tqdm

from data import apply_label_map
from rsgnet.losses import sanitize_logits


def set_random_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def is_dist_initialized():
    return dist.is_available() and dist.is_initialized()


def is_main_process():
    return (not is_dist_initialized()) or dist.get_rank() == 0


def cleanup_distributed():
    if is_dist_initialized():
        dist.destroy_process_group()


def setup_logger(save_dir, enabled=True):
    logger = logging.getLogger("TrainLogger")
    logger.setLevel(logging.INFO)
    logger.propagate = False
    for handler in list(logger.handlers):
        logger.removeHandler(handler)
        handler.close()

    if enabled:
        formatter = logging.Formatter("%(asctime)s - %(message)s", datefmt="%Y-%m-%d %H:%M:%S")
        file_handler = logging.FileHandler(os.path.join(save_dir, "train.log"), mode="a")
        file_handler.setFormatter(formatter)

        class TqdmLoggingHandler(logging.StreamHandler):
            def emit(self, record):
                msg = self.format(record)
                tqdm.write(msg, file=self.stream)
                self.flush()

        console_handler = TqdmLoggingHandler(stream=sys.stdout)
        console_handler.setFormatter(formatter)
        logger.addHandler(file_handler)
        logger.addHandler(console_handler)
    else:
        logger.addHandler(logging.NullHandler())
    return logger


def build_ema_model(model, decay):
    raw_model = model.module if hasattr(model, "module") else model

    def ema_avg_fn(averaged_param, current_param, num_averaged):
        effective_decay = min(decay, (1.0 + num_averaged) / (10.0 + num_averaged))
        return averaged_param * effective_decay + current_param * (1.0 - effective_decay)

    try:
        ema_model = AveragedModel(raw_model, avg_fn=ema_avg_fn, use_buffers=True)
    except TypeError:
        ema_model = AveragedModel(raw_model, avg_fn=ema_avg_fn)
    ema_model.eval()
    for param in ema_model.parameters():
        param.requires_grad_(False)
    return ema_model


def forward_segmentation(model, rgb, hha):
    out = model(rgb, hha)
    return out[0] if isinstance(out, tuple) else out


def get_scaled_size(height, width, scale, divisor=32, size_mode="legacy"):
    if size_mode == "legacy":
        scaled_h = max(divisor, int(round(height * scale / divisor)) * divisor)
        scaled_w = max(divisor, int(round(width * scale / divisor)) * divisor)
    elif size_mode == "dformerv2":
        # DFormerV2 first truncates the scaled side, then rounds it upward to
        # the next multiple of 32 (utils/val_mm.py:evaluate_msf).
        raw_h = int(height * scale)
        raw_w = int(width * scale)
        scaled_h = max(divisor, ((raw_h + divisor - 1) // divisor) * divisor)
        scaled_w = max(divisor, ((raw_w + divisor - 1) // divisor) * divisor)
    else:
        raise ValueError(f"Unknown TTA size mode: {size_mode!r}")
    return scaled_h, scaled_w


def sliding_window_inference(
    model,
    rgb,
    hha,
    crop_size,
    stride_rate=2.0 / 3.0,
    align_corners=False,
    small_image_mode="pad",
):
    if rgb.shape[0] != 1:
        raise ValueError("Sliding-window evaluation requires batch_size=1")
    crop_h, crop_w = (int(crop_size[0]), int(crop_size[1]))
    image_h, image_w = rgb.shape[2:]
    resized_small_image = False
    if crop_h > image_h or crop_w > image_w:
        if small_image_mode == "resize":
            # Match DFormerV2 slide_inference: if either side is short, resize
            # both modalities to the square crop before forwarding.
            rgb = F.interpolate(
                rgb,
                size=(crop_h, crop_w),
                mode="bilinear",
                align_corners=align_corners,
            )
            hha = F.interpolate(
                hha,
                size=(crop_h, crop_w),
                mode="bilinear",
                align_corners=align_corners,
            )
            resized_small_image = True
        elif small_image_mode == "pad":
            pad_h = max(crop_h - image_h, 0)
            pad_w = max(crop_w - image_w, 0)
            rgb = F.pad(rgb, (0, pad_w, 0, pad_h), value=0.0)
            hha = F.pad(hha, (0, pad_w, 0, pad_h), value=0.0)
        else:
            raise ValueError(f"Unknown small-image sliding mode: {small_image_mode!r}")

    padded_h, padded_w = rgb.shape[2:]
    stride_h = max(1, int(round(crop_h * float(stride_rate))))
    stride_w = max(1, int(round(crop_w * float(stride_rate))))
    y_starts = list(range(0, max(padded_h - crop_h, 0) + 1, stride_h))
    x_starts = list(range(0, max(padded_w - crop_w, 0) + 1, stride_w))
    if not y_starts or y_starts[-1] != padded_h - crop_h:
        y_starts.append(padded_h - crop_h)
    if not x_starts or x_starts[-1] != padded_w - crop_w:
        x_starts.append(padded_w - crop_w)

    logits_sum = None
    count = rgb.new_zeros((1, 1, padded_h, padded_w))
    for y in y_starts:
        for x in x_starts:
            rgb_crop = rgb[:, :, y:y + crop_h, x:x + crop_w]
            hha_crop = hha[:, :, y:y + crop_h, x:x + crop_w]
            crop_logits = forward_segmentation(model, rgb_crop, hha_crop)
            crop_logits = F.interpolate(
                crop_logits,
                size=(crop_h, crop_w),
                mode="bilinear",
                align_corners=align_corners,
            )
            if logits_sum is None:
                logits_sum = crop_logits.new_zeros((1, crop_logits.shape[1], padded_h, padded_w))
            logits_sum[:, :, y:y + crop_h, x:x + crop_w] += crop_logits
            count[:, :, y:y + crop_h, x:x + crop_w] += 1.0

    logits = logits_sum / count.clamp_min(1.0)
    if resized_small_image:
        return logits
    return logits[:, :, :image_h, :image_w]


def inference_with_tta(
    model,
    rgb,
    hha,
    scales=(1.0,),
    use_flip=False,
    crop_size=None,
    stride_rate=2.0 / 3.0,
    fusion="logits",
    align_corners=False,
    size_mode="legacy",
    small_image_mode="pad",
):
    if fusion not in {"logits", "probabilities"}:
        raise ValueError(f"Unknown TTA fusion: {fusion!r}")
    base_h, base_w = rgb.shape[2:]
    prediction_sum = None
    num_predictions = 0

    for scale in scales:
        if size_mode == "legacy" and abs(scale - 1.0) < 1e-6:
            scaled_rgb, scaled_hha = rgb, hha
        else:
            scaled_size = get_scaled_size(base_h, base_w, scale, size_mode=size_mode)
            scaled_rgb = F.interpolate(
                rgb,
                size=scaled_size,
                mode="bilinear",
                align_corners=align_corners,
            )
            scaled_hha = F.interpolate(
                hha,
                size=scaled_size,
                mode="bilinear",
                align_corners=align_corners,
            )

        if crop_size is None:
            logits = forward_segmentation(model, scaled_rgb, scaled_hha)
        else:
            logits = sliding_window_inference(
                model,
                scaled_rgb,
                scaled_hha,
                crop_size=crop_size,
                stride_rate=stride_rate,
                align_corners=align_corners,
                small_image_mode=small_image_mode,
            )
        logits = F.interpolate(
            logits,
            size=(base_h, base_w),
            mode="bilinear",
            align_corners=align_corners,
        )
        prediction = logits.softmax(dim=1) if fusion == "probabilities" else logits
        prediction_sum = prediction if prediction_sum is None else prediction_sum + prediction
        num_predictions += 1

        if use_flip:
            flipped_rgb = torch.flip(scaled_rgb, dims=[3])
            flipped_hha = torch.flip(scaled_hha, dims=[3])
            if crop_size is None:
                flip_logits = forward_segmentation(model, flipped_rgb, flipped_hha)
            else:
                flip_logits = sliding_window_inference(
                    model,
                    flipped_rgb,
                    flipped_hha,
                    crop_size=crop_size,
                    stride_rate=stride_rate,
                    align_corners=align_corners,
                    small_image_mode=small_image_mode,
                )
            flip_logits = torch.flip(flip_logits, dims=[3])
            flip_logits = F.interpolate(
                flip_logits,
                size=(base_h, base_w),
                mode="bilinear",
                align_corners=align_corners,
            )
            flip_prediction = (
                flip_logits.softmax(dim=1)
                if fusion == "probabilities"
                else flip_logits
            )
            prediction_sum = prediction_sum + flip_prediction
            num_predictions += 1

    return prediction_sum / max(num_predictions, 1)


def semantic_boundary(labels, valid_mask):
    """Return a two-sided semantic boundary map without crossing ignored pixels."""
    boundary = torch.zeros_like(valid_mask, dtype=torch.bool)
    horizontal = (
        (labels[:, :, 1:] != labels[:, :, :-1])
        & valid_mask[:, :, 1:]
        & valid_mask[:, :, :-1]
    )
    vertical = (
        (labels[:, 1:, :] != labels[:, :-1, :])
        & valid_mask[:, 1:, :]
        & valid_mask[:, :-1, :]
    )
    boundary[:, :, 1:] |= horizontal
    boundary[:, :, :-1] |= horizontal
    boundary[:, 1:, :] |= vertical
    boundary[:, :-1, :] |= vertical
    return boundary


def dilate_binary_mask(mask, radius=2):
    kernel_size = 2 * int(radius) + 1
    return F.max_pool2d(
        mask.float().unsqueeze(1),
        kernel_size=kernel_size,
        stride=1,
        padding=int(radius),
    ).squeeze(1) > 0


def validate(
    model,
    loader,
    device,
    n_classes,
    scales=(1.0,),
    use_flip=False,
    ignore_index=255,
    miou_start_class=1,
    amp_enabled=False,
    amp_dtype=torch.float16,
    crop_size=None,
    stride_rate=2.0 / 3.0,
    tta_fusion="logits",
    tta_align_corners=False,
    tta_size_mode="legacy",
    tta_small_image_mode="pad",
    return_details=False,
):
    model.eval()
    inter = torch.zeros(n_classes, device=device)
    union = torch.zeros(n_classes, device=device)
    interior_inter = torch.zeros(n_classes, device=device)
    interior_union = torch.zeros(n_classes, device=device)
    boundary_stats = torch.zeros(4, device=device)

    with torch.no_grad():
        for rgb, hha, masks in loader:
            rgb = rgb.to(device, non_blocking=True)
            hha = hha.to(device, non_blocking=True)
            masks = masks.to(device, non_blocking=True)
            invalid_mask = (masks != ignore_index) & ((masks < 0) | (masks >= n_classes))
            masks[invalid_mask] = ignore_index

            with torch.autocast(device_type=device.type, dtype=amp_dtype, enabled=amp_enabled):
                raw_out = inference_with_tta(
                    model,
                    rgb,
                    hha,
                    scales=scales,
                    use_flip=use_flip,
                    crop_size=crop_size,
                    stride_rate=stride_rate,
                    fusion=tta_fusion,
                    align_corners=tta_align_corners,
                    size_mode=tta_size_mode,
                    small_image_mode=tta_small_image_mode,
                )
            out = sanitize_logits(raw_out)
            preds = torch.argmax(out, dim=1)
            valid_mask = masks != ignore_index
            gt_boundary = semantic_boundary(masks, valid_mask)
            pred_boundary = semantic_boundary(preds, valid_mask)
            gt_boundary_band = dilate_binary_mask(gt_boundary, radius=2)
            pred_boundary_band = dilate_binary_mask(pred_boundary, radius=2)
            interior_valid = valid_mask & ~gt_boundary_band

            boundary_stats[0] += (pred_boundary & gt_boundary_band).sum()
            boundary_stats[1] += pred_boundary.sum()
            boundary_stats[2] += (gt_boundary & pred_boundary_band).sum()
            boundary_stats[3] += gt_boundary.sum()

            for cls in range(miou_start_class, n_classes):
                pred_mask = (preds == cls) & valid_mask
                gt_mask = (masks == cls) & valid_mask
                inter[cls] += (pred_mask & gt_mask).sum()
                union[cls] += (pred_mask | gt_mask).sum()
                interior_pred = (preds == cls) & interior_valid
                interior_gt = (masks == cls) & interior_valid
                interior_inter[cls] += (interior_pred & interior_gt).sum()
                interior_union[cls] += (interior_pred | interior_gt).sum()

    # During DDP validation each rank evaluates a disjoint validation shard.
    # Aggregate the sufficient statistics before computing per-class IoU so
    # the result is identical to single-process evaluation.
    if is_dist_initialized():
        dist.all_reduce(inter, op=dist.ReduceOp.SUM)
        dist.all_reduce(union, op=dist.ReduceOp.SUM)
        dist.all_reduce(interior_inter, op=dist.ReduceOp.SUM)
        dist.all_reduce(interior_union, op=dist.ReduceOp.SUM)
        dist.all_reduce(boundary_stats, op=dist.ReduceOp.SUM)

    class_iou = torch.full((n_classes,), float("nan"), device=device)
    valid_classes = union > 0
    class_iou[valid_classes] = inter[valid_classes] / (union[valid_classes] + 1e-6)
    evaluated = valid_classes.clone()
    evaluated[:miou_start_class] = False
    if evaluated.sum() == 0:
        miou = 0.0
    else:
        miou = class_iou[evaluated].mean().item()
    interior_class_iou = torch.full((n_classes,), float("nan"), device=device)
    interior_classes = interior_union > 0
    interior_class_iou[interior_classes] = (
        interior_inter[interior_classes] / (interior_union[interior_classes] + 1e-6)
    )
    interior_evaluated = interior_classes.clone()
    interior_evaluated[:miou_start_class] = False
    interior_miou = (
        interior_class_iou[interior_evaluated].mean().item()
        if interior_evaluated.any()
        else 0.0
    )
    boundary_precision = (boundary_stats[0] / boundary_stats[1].clamp_min(1.0)).item()
    boundary_recall = (boundary_stats[2] / boundary_stats[3].clamp_min(1.0)).item()
    boundary_f1 = (
        2.0 * boundary_precision * boundary_recall
        / max(boundary_precision + boundary_recall, 1e-8)
    )
    if not return_details:
        return miou
    return {
        "miou": miou,
        "class_iou": [None if not torch.isfinite(value) else value.item() for value in class_iou.cpu()],
        "intersection": inter.cpu().tolist(),
        "union": union.cpu().tolist(),
        "interior_miou": interior_miou,
        "interior_class_iou": [
            None if not torch.isfinite(value) else value.item()
            for value in interior_class_iou.cpu()
        ],
        "boundary_f1": boundary_f1,
        "boundary_precision": boundary_precision,
        "boundary_recall": boundary_recall,
    }


def freeze_encoder_bn(model):
    raw_model = model.module if hasattr(model, "module") else model
    backbone = getattr(getattr(raw_model, "encoder", None), "rgb_backbone", None)
    if backbone is None:
        return
    for module in backbone.modules():
        if isinstance(module, nn.modules.batchnorm._BatchNorm):
            module.eval()


def rampup_factor(epoch, warmup_epochs, start_factor=0.15):
    if warmup_epochs <= 1:
        return 1.0
    progress = min(max(epoch, 0) / float(warmup_epochs - 1), 1.0)
    return start_factor + (1.0 - start_factor) * progress


def compute_class_weights(
    dataset,
    n_classes,
    ignore_index,
    miou_start_class=1,
    mode="inverse_log",
    smoothing=1.02,
    clamp_max=8.0,
    cache_path=None,
):
    """Compute fixed dataset-level class weights from training-label pixel counts.

    The frequency is gathered from the raw label files (with the dataset's label
    map applied) so it is deterministic and independent of training-time random
    crops/flips. Result is cached to ``cache_path`` keyed on the file list so the
    full scan only runs once.

    mode:
      - ``inverse_log``  : w_c = 1 / log(smoothing + p_c)   (gentle, recommended)
      - ``inverse_freq`` : w_c = median(freq) / freq_c       (aggressive)
    Weights are normalized to mean 1.0 over the evaluated classes and then the
    relative spread is clamped to ``[1/clamp_max, clamp_max]`` around that mean
    (and renormalized) so the balancing effect is predictable regardless of mode.
    Classes below ``miou_start_class`` and ``ignore_index`` get weight 0.
    """
    image_ids = list(dataset.image_ids)
    split_digest = hashlib.sha1("\n".join(image_ids).encode("utf-8")).hexdigest()[:16]
    cache_key = f"{mode}|{smoothing}|{clamp_max}|{n_classes}|{len(image_ids)}|{split_digest}"
    if cache_path and os.path.exists(cache_path):
        try:
            with open(cache_path, "r", encoding="utf-8") as f:
                cached = json.load(f)
            if cached.get("key") == cache_key:
                return torch.tensor(cached["weights"], dtype=torch.float32)
        except (OSError, ValueError, KeyError):
            pass

    counts = np.zeros(n_classes, dtype=np.float64)
    label_dir = os.path.join(dataset.root_dir, dataset.label_dir_name)
    for img_name in tqdm(image_ids, desc="Scanning class frequencies", disable=not is_main_process()):
        base_name = os.path.splitext(img_name)[0]
        label_path = os.path.join(label_dir, f"{base_name}.png")
        label_raw = cv2.imread(label_path, cv2.IMREAD_UNCHANGED)
        if label_raw is None:
            continue
        label = apply_label_map(label_raw, dataset.label_lut, n_classes, label_path, ignore_index=ignore_index)
        valid = (label != ignore_index) & (label < n_classes)
        binc = np.bincount(label[valid].astype(np.int64).ravel(), minlength=n_classes)
        counts += binc[:n_classes]

    counts = np.maximum(counts, 1.0)
    freq = counts / counts.sum()
    if mode == "inverse_freq":
        weights = np.median(freq) / freq
    else:  # inverse_log
        weights = 1.0 / np.log(smoothing + freq)

    # Zero out void / non-evaluated classes so they neither dominate nor distort
    # the normalization of the evaluated foreground classes.
    weights[:miou_start_class] = 0.0
    if 0 <= ignore_index < n_classes:
        weights[ignore_index] = 0.0

    eval_mask = weights > 0
    if eval_mask.any():
        # Normalize to mean 1.0, THEN clamp the relative spread around the mean.
        # Clamping before normalization is wrong for inverse_log on SUN-RGBD: the
        # raw values all exceed clamp_max, collapse to a constant, and balancing
        # silently vanishes. Clamping the normalized ratio keeps a predictable
        # [1/clamp_max, clamp_max] dynamic range in every mode.
        weights[eval_mask] /= weights[eval_mask].mean()
        lo = 1.0 / float(clamp_max)
        weights[eval_mask] = np.clip(weights[eval_mask], lo, float(clamp_max))
        weights[eval_mask] /= weights[eval_mask].mean()

    weights_t = torch.tensor(weights, dtype=torch.float32)
    if cache_path and is_main_process():
        try:
            with open(cache_path, "w", encoding="utf-8") as f:
                json.dump({"key": cache_key, "weights": weights.tolist()}, f)
        except OSError:
            pass
    return weights_t
