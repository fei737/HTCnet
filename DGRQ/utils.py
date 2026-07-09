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

from dataset import apply_label_map
from losses import sanitize_logits


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


def get_scaled_size(height, width, scale, divisor=32):
    scaled_h = max(divisor, int(round(height * scale / divisor)) * divisor)
    scaled_w = max(divisor, int(round(width * scale / divisor)) * divisor)
    return scaled_h, scaled_w


def inference_with_tta(model, rgb, hha, scales=(1.0,), use_flip=False):
    base_h, base_w = rgb.shape[2:]
    logits_sum = None
    num_predictions = 0

    for scale in scales:
        if abs(scale - 1.0) < 1e-6:
            scaled_rgb, scaled_hha = rgb, hha
        else:
            scaled_size = get_scaled_size(base_h, base_w, scale)
            scaled_rgb = F.interpolate(rgb, size=scaled_size, mode="bilinear", align_corners=False)
            scaled_hha = F.interpolate(hha, size=scaled_size, mode="bilinear", align_corners=False)

        logits = forward_segmentation(model, scaled_rgb, scaled_hha)
        logits = F.interpolate(logits, size=(base_h, base_w), mode="bilinear", align_corners=False)
        logits_sum = logits if logits_sum is None else logits_sum + logits
        num_predictions += 1

        if use_flip:
            flip_logits = forward_segmentation(
                model,
                torch.flip(scaled_rgb, dims=[3]),
                torch.flip(scaled_hha, dims=[3]),
            )
            flip_logits = torch.flip(flip_logits, dims=[3])
            flip_logits = F.interpolate(flip_logits, size=(base_h, base_w), mode="bilinear", align_corners=False)
            logits_sum = logits_sum + flip_logits
            num_predictions += 1

    return logits_sum / max(num_predictions, 1)


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
):
    model.eval()
    inter = torch.zeros(n_classes, device=device)
    union = torch.zeros(n_classes, device=device)

    with torch.no_grad():
        for rgb, hha, masks in loader:
            rgb = rgb.to(device, non_blocking=True)
            hha = hha.to(device, non_blocking=True)
            masks = masks.to(device, non_blocking=True)
            invalid_mask = (masks != ignore_index) & ((masks < 0) | (masks >= n_classes))
            masks[invalid_mask] = ignore_index

            with torch.autocast(device_type=device.type, dtype=amp_dtype, enabled=amp_enabled):
                raw_out = inference_with_tta(model, rgb, hha, scales=scales, use_flip=use_flip)
            out = sanitize_logits(raw_out)
            preds = torch.argmax(out, dim=1)
            valid_mask = masks != ignore_index

            for cls in range(miou_start_class, n_classes):
                pred_mask = (preds == cls) & valid_mask
                gt_mask = (masks == cls) & valid_mask
                inter[cls] += (pred_mask & gt_mask).sum()
                union[cls] += (pred_mask | gt_mask).sum()

    valid_classes = union[miou_start_class:] > 0
    if valid_classes.sum() == 0:
        return 0.0
    return torch.mean(inter[miou_start_class:][valid_classes] / (union[miou_start_class:][valid_classes] + 1e-6)).item()


def freeze_encoder_bn(model):
    raw_model = model.module if hasattr(model, "module") else model
    for encoder_name in ("rgb_encoder", "hd_encoder"):
        encoder = getattr(raw_model, encoder_name, None)
        if encoder is None:
            continue
        for module in encoder.modules():
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
