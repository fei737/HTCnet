import os
import random
import cv2
import math
import json
import logging
import argparse
import sys

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.distributed as dist
import segmentation_models_pytorch as smp
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.optim.swa_utils import AveragedModel
from torch.utils.data import Dataset, DataLoader
from torch.utils.data.distributed import DistributedSampler
from tqdm import tqdm

from models import PFNet


INPUT_HEIGHT = 480
INPUT_WIDTH = 640
RGB_MEAN = (0.485, 0.456, 0.406)
RGB_STD = (0.229, 0.224, 0.225)
HHA_MEAN = (0.5, 0.5, 0.5)
HHA_STD = (0.5, 0.5, 0.5)

_SOBEL_X = torch.tensor([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]], dtype=torch.float32).view(1, 1, 3, 3)
_SOBEL_Y = torch.tensor([[-1, -2, -1], [0, 0, 0], [1, 2, 1]], dtype=torch.float32).view(1, 1, 3, 3)


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


def _to_hw(size):
    if isinstance(size, (tuple, list)):
        return int(size[0]), int(size[1])
    return int(size), int(size)


def resize_with_pad(image, target_size, interpolation, pad_value):
    target_h, target_w = _to_hw(target_size)
    h, w = image.shape[:2]
    scale = min(target_h / max(h, 1), target_w / max(w, 1))
    new_h = max(int(round(h * scale)), 1)
    new_w = max(int(round(w * scale)), 1)
    resized = cv2.resize(image, (new_w, new_h), interpolation=interpolation)

    pad_h = target_h - new_h
    pad_w = target_w - new_w
    top = pad_h // 2
    bottom = pad_h - top
    left = pad_w // 2
    right = pad_w - left

    if image.ndim == 3 and np.isscalar(pad_value):
        pad_value = (pad_value,) * image.shape[2]

    return cv2.copyMakeBorder(
        resized,
        top,
        bottom,
        left,
        right,
        borderType=cv2.BORDER_CONSTANT,
        value=pad_value,
    )


def _parse_label_map_line(line):
    line = line.split("#", 1)[0].strip()
    if not line:
        return None
    parts = line.replace(",", " ").split()
    if len(parts) < 2:
        raise ValueError(f"Invalid label-map line: {line}")
    return int(parts[0]), int(parts[1])


def load_label_map(label_map_path, n_classes, ignore_index=255):
    if not label_map_path:
        return None
    if not os.path.exists(label_map_path):
        raise FileNotFoundError(f"Label map not found: {label_map_path}")

    mapping = {}
    if label_map_path.lower().endswith(".json"):
        with open(label_map_path, "r", encoding="utf-8") as f:
            raw_mapping = json.load(f)
        if isinstance(raw_mapping, dict):
            mapping = {int(k): int(v) for k, v in raw_mapping.items()}
        elif isinstance(raw_mapping, list):
            mapping = {int(item["raw"]): int(item["train"]) for item in raw_mapping}
        else:
            raise ValueError("JSON label map must be a dict or a list of {'raw', 'train'} items.")
    else:
        with open(label_map_path, "r", encoding="utf-8") as f:
            for line in f:
                parsed = _parse_label_map_line(line)
                if parsed is not None:
                    raw_id, train_id = parsed
                    mapping[raw_id] = train_id

    if not mapping:
        raise ValueError(f"Label map is empty: {label_map_path}")
    bad_targets = sorted({v for v in mapping.values() if v != ignore_index and (v < 0 or v >= n_classes)})
    if bad_targets:
        raise ValueError(f"Label map has target ids outside [0, {n_classes - 1}]: {bad_targets[:10]}")

    lut = np.full(max(mapping.keys()) + 1, fill_value=ignore_index, dtype=np.uint8)
    for raw_id, train_id in mapping.items():
        if raw_id < 0:
            raise ValueError(f"Label map has negative raw id: {raw_id}")
        lut[raw_id] = train_id
    return lut


def apply_label_map(label, label_lut, n_classes, label_path, ignore_index=255):
    if label.ndim == 3:
        raise ValueError(
            f"Label image is RGB/color-coded, but this loader expects class-index masks: {label_path}. "
            "Please export indexed SUN40 masks or provide a preprocessing script that maps colors to class ids."
        )

    label = label.astype(np.int64, copy=False)
    if label_lut is not None:
        remapped = np.full_like(label, fill_value=ignore_index, dtype=np.uint8)
        valid = label < len(label_lut)
        remapped[valid] = label_lut[label[valid]]
        return remapped

    unique = np.unique(label)
    invalid_values = unique[(unique != ignore_index) & ((unique < 0) | (unique >= n_classes))]
    if invalid_values.size > 0:
        max_label = int(label.max()) if label.size else 0
        preview = unique[:30].tolist()
        raise ValueError(
            f"Label ids exceed n_classes={n_classes} in {label_path}. "
            f"max={max_label}, sample_unique={preview}. "
            f"Use --label-map to convert raw ids to train ids 0..n_classes-1 or ignore_index={ignore_index} before training."
        )
    return label.astype(np.uint8, copy=False)


def setup_logger(save_dir, enabled=True):
    logger = logging.getLogger("TrainLogger")
    logger.setLevel(logging.INFO)
    logger.propagate = False
    for handler in list(logger.handlers):
        logger.removeHandler(handler)
        handler.close()

    if enabled:
        formatter = logging.Formatter("%(asctime)s - %(message)s", datefmt="%Y-%m-%d %H:%M:%S")
        fh = logging.FileHandler(os.path.join(save_dir, "train.log"), mode="a")
        fh.setLevel(logging.INFO)
        fh.setFormatter(formatter)
        class TqdmLoggingHandler(logging.StreamHandler):
            def emit(self, record):
                try:
                    msg = self.format(record)
                    tqdm.write(msg, file=self.stream)
                    self.flush()
                except Exception:
                    self.handleError(record)

        ch = TqdmLoggingHandler(stream=sys.stdout)
        ch.setLevel(logging.INFO)
        ch.setFormatter(formatter)
        logger.addHandler(fh)
        logger.addHandler(ch)
    else:
        logger.addHandler(logging.NullHandler())
    return logger


def get_angle_grad_target(hha):
    x = hha[:, 2:3, :, :]
    kernel_x = _SOBEL_X.to(x.device).type_as(x)
    kernel_y = _SOBEL_Y.to(x.device).type_as(x)
    gx = F.conv2d(x, kernel_x, padding=1)
    gy = F.conv2d(x, kernel_y, padding=1)
    g = torch.abs(gx) + torch.abs(gy)
    return (g > 0.1).float()


def sanitize_mask_values(mask, n_classes, ignore_index=255):
    invalid_mask = (mask != ignore_index) & ((mask < 0) | (mask >= n_classes))
    invalid_count = int(invalid_mask.sum()) if isinstance(invalid_mask, np.ndarray) else int(invalid_mask.sum().item())
    if invalid_count > 0:
        mask = mask.copy() if isinstance(mask, np.ndarray) else mask.clone()
        mask[invalid_mask] = ignore_index
    return mask, invalid_count


class SUNDataset(Dataset):
    def __init__(
        self,
        root_dir,
        mode="train",
        crop_size=(INPUT_HEIGHT, INPUT_WIDTH),
        rgb_mean=RGB_MEAN,
        rgb_std=RGB_STD,
        hha_mean=HHA_MEAN,
        hha_std=HHA_STD,
        n_classes=41,
        label_map_path="",
        label_dir_name="Labels",
        ignore_index=255,
    ):
        self.root_dir = os.path.join(root_dir, "SUNRGBD")
        self.mode = mode
        self.crop_h, self.crop_w = _to_hw(crop_size)
        self.label_dir_name = label_dir_name
        self.n_classes = n_classes
        self.ignore_index = ignore_index
        self.label_lut = load_label_map(label_map_path, n_classes, ignore_index=ignore_index)
        self.rgb_mean = torch.tensor(rgb_mean).view(3, 1, 1).float()
        self.rgb_std = torch.tensor(rgb_std).view(3, 1, 1).float()
        self.hha_mean = torch.tensor(hha_mean).view(3, 1, 1).float()
        self.hha_std = torch.tensor(hha_std).view(3, 1, 1).float()
        self.rgb_pad_value = tuple(int(round(v * 255.0)) for v in rgb_mean)
        self.hha_pad_value = tuple(int(round(v * 255.0)) for v in hha_mean)
        list_path = os.path.join(self.root_dir, f"{mode}.txt")
        required_dirs = ["RGB", "HHA", self.label_dir_name]
        missing_dirs = [name for name in required_dirs if not os.path.isdir(os.path.join(self.root_dir, name))]
        if missing_dirs:
            raise FileNotFoundError(
                f"SUNRGBD dataset is incomplete under {self.root_dir}. Missing folders: {missing_dirs}. "
                "Expected structure: SUNRGBD/RGB, SUNRGBD/HHA, SUNRGBD/Labels, SUNRGBD/train.txt, SUNRGBD/test.txt."
            )
        if not os.path.isfile(list_path):
            raise FileNotFoundError(
                f"Official split file not found: {list_path}. "
                "For academic reproducibility, prepare SUNRGBD/train.txt and SUNRGBD/test.txt instead of random splitting."
            )
        with open(list_path, "r", encoding="utf-8") as f:
            self.image_ids = [self._normalize_image_id(line.strip()) for line in f.readlines() if line.strip()]
        if not self.image_ids:
            raise ValueError(f"SUNRGBD split file is empty: {list_path}")

    @staticmethod
    def _normalize_image_id(image_id):
        image_id = image_id.replace("\\", "/").split("/")[-1]
        return image_id if os.path.splitext(image_id)[1] else f"{image_id}.png"

    def __len__(self):
        return len(self.image_ids)

    def __getitem__(self, index):
        img_name = self.image_ids[index]
        base_name = os.path.splitext(img_name)[0]

        rgb_path = os.path.join(self.root_dir, "RGB", img_name)
        hha_path = os.path.join(self.root_dir, "HHA", f"{base_name}.png")
        label_path = os.path.join(self.root_dir, self.label_dir_name, f"{base_name}.png")

        rgb_raw = cv2.imread(rgb_path)
        hha_raw = cv2.imread(hha_path)
        label_raw = cv2.imread(label_path, cv2.IMREAD_UNCHANGED)

        if rgb_raw is None:
            raise FileNotFoundError(f"RGB not found: {rgb_path}")
        if hha_raw is None:
            raise FileNotFoundError(f"HHA not found: {hha_path}")
        if label_raw is None:
            raise FileNotFoundError(f"Label not found: {label_path}")

        rgb = cv2.cvtColor(rgb_raw, cv2.COLOR_BGR2RGB)
        hha = cv2.cvtColor(hha_raw, cv2.COLOR_BGR2RGB)
        label = apply_label_map(label_raw, self.label_lut, self.n_classes, label_path, ignore_index=self.ignore_index)
        label, invalid_count = sanitize_mask_values(label, self.n_classes, ignore_index=self.ignore_index)
        if invalid_count > 0:
            raise ValueError(
                f"Invalid label ids found after preprocessing in {label_path}: "
                f"{invalid_count} pixels are outside [0, {self.n_classes - 1}]"
            )

        if self.mode == "train":
            h, w = rgb.shape[:2]
            rgb = rgb[5:h - 5, 5:w - 5]
            hha = hha[5:h - 5, 5:w - 5]
            label = label[5:h - 5, 5:w - 5]

            if random.random() > 0.5:
                rgb = cv2.flip(rgb, 1)
                hha = cv2.flip(hha, 1)
                label = cv2.flip(label, 1)

            if random.random() > 0.5:
                hsv = cv2.cvtColor(rgb, cv2.COLOR_RGB2HSV)
                hsv = np.array(hsv, dtype=np.float64)
                hsv[:, :, 1] *= random.uniform(0.7, 1.3)
                hsv[:, :, 2] *= random.uniform(0.7, 1.3)
                hsv[:, :, 1][hsv[:, :, 1] > 255] = 255
                hsv[:, :, 2][hsv[:, :, 2] > 255] = 255
                hsv = np.array(hsv, dtype=np.uint8)
                rgb = cv2.cvtColor(hsv, cv2.COLOR_HSV2RGB)

            scale = random.uniform(0.75, 1.75)
            h, w = rgb.shape[:2]
            new_h = max(int(h * scale), self.crop_h)
            new_w = max(int(w * scale), self.crop_w)

            rgb = cv2.resize(rgb, (new_w, new_h), interpolation=cv2.INTER_LINEAR)
            hha = cv2.resize(hha, (new_w, new_h), interpolation=cv2.INTER_LINEAR)
            label = cv2.resize(label, (new_w, new_h), interpolation=cv2.INTER_NEAREST)

            y1 = random.randint(0, new_h - self.crop_h)
            x1 = random.randint(0, new_w - self.crop_w)
            rgb = rgb[y1:y1 + self.crop_h, x1:x1 + self.crop_w]
            hha = hha[y1:y1 + self.crop_h, x1:x1 + self.crop_w]
            label = label[y1:y1 + self.crop_h, x1:x1 + self.crop_w]

            if random.random() > 0.5:
                cut_h = random.randint(60, 120)
                cut_w = random.randint(60, 120)
                cut_y = random.randint(0, self.crop_h - cut_h)
                cut_x = random.randint(0, self.crop_w - cut_w)
                rgb[cut_y:cut_y + cut_h, cut_x:cut_x + cut_w, :] = 0
        else:
            target_size = (self.crop_h, self.crop_w)
            rgb = resize_with_pad(rgb, target_size, interpolation=cv2.INTER_LINEAR, pad_value=self.rgb_pad_value)
            hha = resize_with_pad(hha, target_size, interpolation=cv2.INTER_LINEAR, pad_value=self.hha_pad_value)
            label = resize_with_pad(label, target_size, interpolation=cv2.INTER_NEAREST, pad_value=self.ignore_index)

        rgb = np.ascontiguousarray(rgb)
        hha = np.ascontiguousarray(hha)
        label = np.ascontiguousarray(label)

        rgb_t = torch.from_numpy(rgb.transpose(2, 0, 1)).float() / 255.0
        hha_t = torch.from_numpy(hha.transpose(2, 0, 1)).float() / 255.0
        rgb_t = (rgb_t - self.rgb_mean) / self.rgb_std
        hha_t = (hha_t - self.hha_mean) / self.hha_std
        mask_t = torch.from_numpy(label).long()
        return rgb_t, hha_t, mask_t


def build_ema_model(model, decay):
    raw_model = model.module if hasattr(model, "module") else model

    def ema_avg_fn(averaged_param, current_param, num_averaged):
        # Follow raw weights quickly at the beginning, then approach the target decay.
        d = min(decay, (1.0 + num_averaged) / (10.0 + num_averaged))
        return averaged_param * d + current_param * (1.0 - d)

    try:
        ema_model = AveragedModel(raw_model, avg_fn=ema_avg_fn, use_buffers=True)
    except TypeError:
        ema_model = AveragedModel(raw_model, avg_fn=ema_avg_fn)
    ema_model.eval()
    for param in ema_model.parameters():
        param.requires_grad_(False)
    return ema_model


class OHEMCrossEntropyLoss(nn.Module):
    def __init__(self, thresh=0.7, ignore_index=255, min_kept=30000, n_classes=41):
        super().__init__()
        self.thresh = -math.log(thresh)
        self.ignore_index = ignore_index
        self.min_kept = min_kept
        self.n_classes = n_classes
        self.criterion = nn.CrossEntropyLoss(ignore_index=ignore_index, reduction="none")

    def forward(self, logits, labels):
        if labels.dtype != torch.long:
            labels = labels.long()
        invalid_mask = (labels < 0) | (labels >= self.n_classes)
        if invalid_mask.any():
            labels = labels.clone()
            labels[invalid_mask] = self.ignore_index

        loss = self.criterion(logits, labels)
        valid_mask = labels != self.ignore_index

        if valid_mask.sum() == 0:
            return loss.sum() * 0.0

        labels_valid = labels[valid_mask]
        class_counts = torch.bincount(labels_valid, minlength=self.n_classes).float()
        class_counts[class_counts == 0] = 1.0
        class_weights = (valid_mask.sum() / (class_counts * self.n_classes)).clamp(1.0, 5.0)

        with torch.no_grad():
            weight_map = torch.zeros_like(loss)
            weight_map[valid_mask] = class_weights[labels_valid]

        weighted_loss = loss * weight_map * valid_mask.float()
        weighted_loss_flat = weighted_loss.reshape(-1)
        valid_mask_flat = valid_mask.reshape(-1)

        num_valid = int(valid_mask_flat.sum().item())
        kept = min(self.min_kept, num_valid)
        if kept > 0:
            with torch.no_grad():
                sorted_loss, _ = torch.sort(weighted_loss_flat.detach(), descending=True)
                threshold_value = sorted_loss[kept - 1].item()
                actual_thresh = min(self.thresh, threshold_value)

            keep_mask = (weighted_loss_flat >= actual_thresh).float() * valid_mask_flat.float()
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


def boundary_weighted_cross_entropy(logits, labels, edge_labels, ignore_index=255, edge_weight=2.0):
    labels = labels.long()
    invalid_mask = (labels < 0) | (labels >= logits.shape[1])
    if invalid_mask.any():
        labels = labels.clone()
        labels[invalid_mask] = ignore_index
    ce = F.cross_entropy(logits, labels, ignore_index=ignore_index, reduction="none")
    valid_mask = labels != ignore_index
    weights = 1.0 + edge_weight * edge_labels.squeeze(1).float()
    weighted_loss = ce * weights * valid_mask.float()
    valid_count = valid_mask.sum().clamp_min(1).float()
    return weighted_loss.sum() / valid_count


def probability_edge_map(logits):
    logits = torch.nan_to_num(logits.float(), nan=0.0, posinf=30.0, neginf=-30.0).clamp(-30.0, 30.0)
    probs = F.softmax(logits, dim=1)
    b, c, _, _ = probs.shape
    kernel_x = torch.tensor([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]], dtype=probs.dtype, device=probs.device).view(1, 1, 3, 3)
    kernel_y = torch.tensor([[-1, -2, -1], [0, 0, 0], [1, 2, 1]], dtype=probs.dtype, device=probs.device).view(1, 1, 3, 3)
    kernel_x = kernel_x.repeat(c, 1, 1, 1)
    kernel_y = kernel_y.repeat(c, 1, 1, 1)
    gx = F.conv2d(probs, kernel_x, padding=1, groups=c)
    gy = F.conv2d(probs, kernel_y, padding=1, groups=c)
    edge_prob = (torch.abs(gx) + torch.abs(gy)).mean(dim=1, keepdim=True)
    normalizer = edge_prob.flatten(1).amax(dim=1).view(b, 1, 1, 1).clamp_min(1e-6)
    return (edge_prob / normalizer).clamp(0.0, 1.0)


def feature_precision_loss(seg_logits, edge_labels, hha_grad_target=None, valid_mask=None):
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


class PFNetCombinedLoss(nn.Module):
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
        n_classes=41,
        ignore_index=255,
    ):
        super().__init__()
        self.ignore_index = ignore_index
        self.ohem_ce = OHEMCrossEntropyLoss(thresh=0.7, ignore_index=ignore_index, min_kept=ohem_min_kept, n_classes=n_classes)
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
    ):
        loss_ce = self.ohem_ce(seg_logits, seg_labels)
        loss_lov = self.lovasz(seg_logits, seg_labels)
        loss_dice = self.dice(seg_logits, seg_labels)
        loss_boundary = boundary_weighted_cross_entropy(
            seg_logits, seg_labels, edge_labels, ignore_index=self.ignore_index, edge_weight=self.boundary_ce_weight
        )
        edge_logits = edge_logits.squeeze(1)
        loss_edge = balanced_bce_with_logits(
            edge_logits, edge_labels.squeeze(1), max_pos_weight=self.max_edge_pos_weight
        )

        if hha_grad_target is not None and self.hha_edge_weight > 0:
            loss_edge += self.hha_edge_weight * balanced_bce_with_logits(
                edge_logits, hha_grad_target.squeeze(1), max_pos_weight=self.max_edge_pos_weight
            )

        loss_feature_precision = feature_precision_loss(
            seg_logits, edge_labels, hha_grad_target, valid_mask=valid_edge_mask
        )

        loss_aux = 0.0
        if aux_logits is not None:
            loss_aux = self.ohem_ce(aux_logits, seg_labels) * self.aux_weight

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


def forward_segmentation(model, rgb, hha):
    out = model(rgb, hha)
    if isinstance(out, tuple):
        out = out[0]
    return out


def get_scaled_size(height, width, scale, divisor=32):
    scaled_h = max(divisor, int(round(height * scale / divisor)) * divisor)
    scaled_w = max(divisor, int(round(width * scale / divisor)) * divisor)
    return scaled_h, scaled_w


def rampup_factor(epoch, warmup_epochs, start_factor=0.15):
    if warmup_epochs <= 1:
        return 1.0
    progress = min(max(epoch, 0) / float(warmup_epochs - 1), 1.0)
    return start_factor + (1.0 - start_factor) * progress


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


def _summarize_class_hist(mask, max_items=12):
    values, counts = torch.unique(mask, return_counts=True)
    pairs = sorted(
        zip(values.detach().cpu().tolist(), counts.detach().cpu().tolist()),
        key=lambda item: item[1],
        reverse=True,
    )
    return ", ".join(f"{int(v)}:{int(c)}" for v, c in pairs[:max_items]) if pairs else "empty"


def validate(model, loader, device, n_classes, scales=(1.0,), use_flip=False, debug_stats=False, ignore_index=255, miou_start_class=1):
    model.eval()
    inter = torch.zeros(n_classes, device=device)
    union = torch.zeros(n_classes, device=device)
    with torch.no_grad():
        for batch_idx, (rgb, hha, masks) in enumerate(loader):
            rgb = rgb.to(device, non_blocking=True)
            hha = hha.to(device, non_blocking=True)
            masks = masks.to(device, non_blocking=True)
            invalid_mask = (masks != ignore_index) & ((masks < 0) | (masks >= n_classes))
            masks[invalid_mask] = ignore_index
            out = inference_with_tta(model, rgb, hha, scales=scales, use_flip=use_flip)
            preds = torch.argmax(out, dim=1)
            valid_mask = masks != ignore_index

            if debug_stats and batch_idx == 0:
                valid_pixels = int(valid_mask.sum().item())
                total_pixels = int(valid_mask.numel())
                gt_valid = masks[valid_mask]
                pred_valid = preds[valid_mask]
                print("[val-debug] valid pixels:", f"{valid_pixels}/{total_pixels} ({valid_pixels / max(total_pixels, 1):.4f})")
                print("[val-debug] gt valid hist:", _summarize_class_hist(gt_valid))
                print("[val-debug] pred valid hist:", _summarize_class_hist(pred_valid))
                print("[val-debug] pred full hist:", _summarize_class_hist(preds))

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


def main(args):
    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    use_ddp = world_size > 1 and not args.no_data_parallel and torch.cuda.is_available()

    try:
        if use_ddp:
            visible_cuda_devices = torch.cuda.device_count()
            if local_rank >= visible_cuda_devices:
                raise RuntimeError(
                    f"LOCAL_RANK={local_rank} but only {visible_cuda_devices} CUDA device(s) are visible. "
                    f"CUDA_VISIBLE_DEVICES={os.environ.get('CUDA_VISIBLE_DEVICES', '<unset>')}. "
                    "Use valid GPU ids or let run.sh launch no more processes than visible devices."
                )
            torch.cuda.set_device(local_rank)
            try:
                dist.init_process_group(
                    backend="nccl",
                    init_method="env://",
                    device_id=torch.device("cuda", local_rank),
                )
            except TypeError:
                dist.init_process_group(backend="nccl", init_method="env://")
            device = torch.device("cuda", local_rank)
        else:
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        os.makedirs(args.save_dir, exist_ok=True)
        logger = setup_logger(args.save_dir, enabled=is_main_process() if use_ddp else True)
        if is_main_process() or not use_ddp:
            logger.info("=" * 50)
            logger.info("启动新的训练进程 (SUN Dataset)")
            logger.info(f"训练配置: {args}")
            logger.info("=" * 50)

        set_random_seed(args.seed + (dist.get_rank() if is_dist_initialized() else 0))
        if device.type == "cuda":
            torch.backends.cudnn.benchmark = True

        model = PFNet(
            n_classes=args.n_classes,
            pretrained_path=args.pretrained_encoder,
            return_aux=True,
            encoder_name=args.encoder_name,
            safe_mode=args.safe_mode,
        )

        checkpoint = None
        start_epoch = 0
        best_miou = 0.0
        if args.resume and os.path.exists(args.resume):
            checkpoint = torch.load(args.resume, map_location="cpu")
            state_dict = checkpoint.get("model_state_dict", checkpoint)
            model.load_state_dict(state_dict, strict=False)
            if is_main_process() or not use_ddp:
                logger.info(f"已加载恢复权重: {args.resume}")

        model.to(device)
        if use_ddp:
            model = DDP(
                model,
                device_ids=[local_rank],
                output_device=local_rank,
                find_unused_parameters=False,
                broadcast_buffers=False,
                gradient_as_bucket_view=True,
            )
            logger.info(f"启用 DistributedDataParallel，多卡数量: {world_size}，当前 rank: {dist.get_rank()}")

        ema_model = None
        if args.ema_decay > 0 and (is_main_process() or not use_ddp):
            ema_model = build_ema_model(model, decay=args.ema_decay)

        loader_kwargs = {
            "num_workers": args.num_workers,
            "pin_memory": device.type == "cuda",
        }
        if args.num_workers > 0:
            loader_kwargs["persistent_workers"] = True

        train_dataset = SUNDataset(
            args.data_root,
            mode="train",
            crop_size=(args.input_height, args.input_width),
            n_classes=args.n_classes,
            label_map_path=args.label_map,
            label_dir_name=args.label_dir_name,
            ignore_index=args.ignore_index,
        )
        val_dataset = SUNDataset(
            args.data_root,
            mode="test",
            crop_size=(args.input_height, args.input_width),
            n_classes=args.n_classes,
            label_map_path=args.label_map,
            label_dir_name=args.label_dir_name,
            ignore_index=args.ignore_index,
        )

        train_sampler = None
        if use_ddp:
            train_sampler = DistributedSampler(
                train_dataset,
                num_replicas=world_size,
                rank=dist.get_rank(),
                shuffle=True,
                drop_last=True,
            )

        train_loader = DataLoader(
            train_dataset,
            batch_size=args.batch_size,
            shuffle=train_sampler is None,
            sampler=train_sampler,
            drop_last=True,
            **loader_kwargs,
        )
        val_loader = DataLoader(
            val_dataset,
            batch_size=args.batch_size,
            shuffle=False,
            **loader_kwargs,
        )

        criterion = PFNetCombinedLoss(
            lambda_lovasz=args.lambda_lovasz,
            lambda_dice=args.lambda_dice,
            lambda_edge=args.lambda_edge,
            lambda_boundary=args.lambda_boundary,
            lambda_feature_precision=args.lambda_feature_precision,
            hha_edge_weight=args.hha_edge_weight,
            max_edge_pos_weight=args.max_edge_pos_weight,
            ohem_min_kept=args.ohem_min_kept,
            boundary_ce_weight=args.boundary_ce_weight,
            n_classes=args.n_classes,
            ignore_index=args.ignore_index,
        ).to(device)

        raw_model = model.module if hasattr(model, "module") else model
        rgb_params = raw_model.rgb_encoder.parameters()
        hd_params = raw_model.hd_encoder.parameters()
        side_params = list(raw_model.angle_grad.parameters()) + list(raw_model.side_guide.parameters())
        head_params = [
            p for n, p in model.named_parameters()
            if "encoder" not in n and "angle_grad" not in n and "side_guide" not in n
        ]

        optimizer = torch.optim.AdamW(
            [
                {"params": rgb_params, "lr": args.lr * 0.1},
                {"params": hd_params, "lr": args.lr * 0.8},
                {"params": side_params, "lr": args.lr * 1.0},
                {"params": head_params, "lr": args.lr},
            ],
            weight_decay=1e-4,
        )

        accumulation_steps = max(1, 16 // args.batch_size)
        actual_steps_per_epoch = len(train_loader) // accumulation_steps + (1 if len(train_loader) % accumulation_steps != 0 else 0)
        total_steps = args.epochs * actual_steps_per_epoch
        warmup_steps = 5 * actual_steps_per_epoch

        def warmup_poly_lambda(step):
            if step < warmup_steps:
                return float(step) / float(max(1, warmup_steps))
            decay_steps = max(1, total_steps - warmup_steps)
            return (1.0 - (step - warmup_steps) / decay_steps) ** 0.9

        scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=warmup_poly_lambda)

        if checkpoint is not None and "model_state_dict" in checkpoint:
            start_epoch = checkpoint.get("epoch", 0)
            best_miou = checkpoint.get("best_miou", 0.0)
            if "optimizer_state_dict" in checkpoint:
                optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
            if "scheduler_state_dict" in checkpoint:
                scheduler.load_state_dict(checkpoint["scheduler_state_dict"])
            if ema_model is not None and checkpoint.get("ema_state_dict") is not None:
                ema_model.load_state_dict(checkpoint["ema_state_dict"], strict=False)
            if is_main_process() or not use_ddp:
                logger.info(f"恢复训练状态: Epoch {start_epoch}, 历史最佳 mIoU={best_miou:.4f}")

        amp_enabled = (not args.no_amp) and device.type == "cuda"
        amp_dtype = torch.bfloat16 if args.amp_dtype == "bf16" else torch.float16
        scaler = torch.amp.GradScaler("cuda", enabled=amp_enabled and amp_dtype == torch.float16)

        for epoch in range(start_epoch, args.epochs):
            if train_sampler is not None:
                train_sampler.set_epoch(epoch)

            edge_loss_scale = rampup_factor(epoch, args.edge_warmup_epochs, args.loss_warmup_start)
            boundary_loss_scale = rampup_factor(epoch, args.boundary_warmup_epochs, args.loss_warmup_start)
            feature_loss_scale = rampup_factor(epoch, args.feature_warmup_epochs, args.loss_warmup_start)

            if is_main_process() or not use_ddp:
                logger.info(
                    f"Loss curriculum | edge={edge_loss_scale:.3f} "
                    f"boundary={boundary_loss_scale:.3f} feature={feature_loss_scale:.3f}"
                )

            model.train()
            freeze_encoder_bn(model)
            progress = tqdm(
                train_loader,
                desc=f"Epoch {epoch + 1}/{args.epochs}",
                disable=use_ddp and not is_main_process(),
                dynamic_ncols=True,
            )
            optimizer.zero_grad(set_to_none=True)
            epoch_loss_sum = 0.0
            epoch_loss_count = 0.0
            skipped_nonfinite = 0

            for step, (rgb, hha, masks) in enumerate(progress):
                rgb = rgb.to(device, non_blocking=True)
                hha = hha.to(device, non_blocking=True)
                masks = masks.to(device, non_blocking=True)
                invalid_mask = (masks != args.ignore_index) & ((masks < 0) | (masks >= args.n_classes))
                invalid_count = int(invalid_mask.sum().item())
                if invalid_count > 0:
                    masks = masks.clone()
                    masks[invalid_mask] = args.ignore_index
                    if is_main_process() and step < 5:
                        logger.warning(
                            f"Epoch {epoch + 1} Step {step + 1}: clamped {invalid_count} invalid label pixels to ignore_index={args.ignore_index}"
                        )

                valid_edge_mask = (masks != args.ignore_index).float().unsqueeze(1)
                edge_tgt = edge_target_from_mask(masks, ignore_index=args.ignore_index).float()
                angle_grad_tgt = get_angle_grad_target(hha).float() * valid_edge_mask

                with torch.autocast(device_type=device.type, dtype=amp_dtype, enabled=amp_enabled):
                    seg_logits, edge_logits, aux_logits = model(rgb, hha)
                    seg_logits = seg_logits.float()
                    edge_logits = edge_logits.float()
                    if aux_logits is not None:
                        aux_logits = aux_logits.float()

                    total_loss, loss_ce, loss_lov, loss_dice, loss_edge, loss_feature_precision = criterion(
                        seg_logits,
                        masks,
                        edge_logits,
                        edge_tgt,
                        aux_logits=aux_logits,
                        hha_grad_target=angle_grad_tgt,
                        valid_edge_mask=valid_edge_mask,
                        edge_loss_scale=edge_loss_scale,
                        boundary_loss_scale=boundary_loss_scale,
                        feature_loss_scale=feature_loss_scale,
                    )
                if not torch.isfinite(total_loss):
                    skipped_nonfinite += 1
                    optimizer.zero_grad(set_to_none=True)
                    if is_main_process() and skipped_nonfinite <= 5:
                        logger.warning(
                            "Skipped non-finite loss at "
                            f"epoch={epoch + 1} step={step + 1}: "
                            f"total={float(total_loss.detach().cpu())} "
                            f"ce={float(loss_ce.detach().cpu())} "
                            f"lov={float(loss_lov.detach().cpu())} "
                            f"dice={float(loss_dice.detach().cpu())} "
                            f"edge={float(loss_edge.detach().cpu())} "
                            f"feature={float(loss_feature_precision.detach().cpu())}"
                        )
                    continue
                loss = total_loss / accumulation_steps
                scaler.scale(loss).backward()

                if (step + 1) % accumulation_steps == 0 or (step + 1) == len(train_loader):
                    scaler.unscale_(optimizer)
                    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

                    scale_before = scaler.get_scale()
                    scaler.step(optimizer)
                    scaler.update()
                    scale_after = scaler.get_scale()

                    if scale_before <= scale_after:
                        scheduler.step()
                        if ema_model is not None:
                            ema_model.update_parameters(raw_model)
                    optimizer.zero_grad(set_to_none=True)

                epoch_loss_sum += total_loss.item()
                epoch_loss_count += 1.0
                if not (use_ddp and not is_main_process()):
                    progress.set_postfix(
                        tot=f"{total_loss.item():.3f}",
                        ce=f"{loss_ce.item():.3f}",
                        lov=f"{loss_lov.item():.3f}",
                        edg=f"{loss_edge.item():.3f}",
                    )

            epoch_stats = torch.tensor([epoch_loss_sum, epoch_loss_count, float(skipped_nonfinite)], device=device)
            if use_ddp:
                dist.all_reduce(epoch_stats, op=dist.ReduceOp.SUM)
            avg_loss = (epoch_stats[0] / epoch_stats[1].clamp_min(1.0)).item()

            if is_main_process() or not use_ddp:
                skip_msg = f" | skipped_nonfinite={int(epoch_stats[2].item())}" if epoch_stats[2].item() > 0 else ""
                logger.info(f"Epoch {epoch + 1}/{args.epochs} | Avg Train Loss: {avg_loss:.4f} | LR: {optimizer.param_groups[-1]['lr']:.6f}{skip_msg}")

            raw_model = model.module if hasattr(model, "module") else model
            should_validate = (epoch + 1) % args.val_interval == 0 or (epoch + 1) == args.epochs
            if should_validate:
                if use_ddp:
                    dist.barrier(device_ids=[local_rank])
                if is_main_process() or not use_ddp:
                    use_ema_eval = ema_model is not None and (epoch + 1) >= args.ema_warmup_epochs
                    eval_model = ema_model if use_ema_eval else raw_model
                    val_scales = (0.75, 1.0, 1.25) if args.use_tta else (1.0,)
                    miou = validate(
                        eval_model,
                        val_loader,
                        device=device,
                        n_classes=args.n_classes,
                        scales=val_scales,
                        use_flip=args.use_tta,
                        debug_stats=args.debug_val_stats,
                        ignore_index=args.ignore_index,
                        miou_start_class=args.miou_start_class,
                    )
                    eval_name = "EMA" if use_ema_eval else "raw"
                    logger.info(f"Epoch {epoch + 1} | Validation mIoU ({eval_name}): {miou:.4f}")

                    if miou > best_miou:
                        best_miou = miou
                        ckpt = os.path.join(args.save_dir, "best_model.pth")
                        torch.save(eval_model.state_dict(), ckpt)
                        logger.info(f"New best checkpoint reached: {best_miou:.4f} (已保存至 {ckpt})")
                if use_ddp:
                    dist.barrier(device_ids=[local_rank])
            elif is_main_process() or not use_ddp:
                logger.info(f"Epoch {epoch + 1} | Validation skipped (runs every {args.val_interval} epochs).")

            if is_main_process() or not use_ddp:
                checkpoint_dict = {
                    "epoch": epoch + 1,
                    "model_state_dict": raw_model.state_dict(),
                    "ema_state_dict": ema_model.state_dict() if ema_model is not None else None,
                    "optimizer_state_dict": optimizer.state_dict(),
                    "scheduler_state_dict": scheduler.state_dict(),
                    "best_miou": best_miou,
                }
                torch.save(checkpoint_dict, os.path.join(args.save_dir, "latest_model.pth"))
    finally:
        cleanup_distributed()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-root", type=str, default="/home/pengfei/HTCnet/DataSets")
    parser.add_argument("--save-dir", type=str, default="../Checkpoint_SUN_V1")
    parser.add_argument("--pretrained-encoder", type=str, default="/home/pengfei/HTCnet/Checkpoint/mit_b2.pth")
    parser.add_argument("--resume", type=str, default="")
    parser.add_argument("--label-map", type=str, default="", help='Optional raw-label to train-label map: txt/csv lines "raw train" or JSON dict.')
    parser.add_argument("--label-dir-name", type=str, default="Labels", help="Label folder under DataSets/SUNRGBD, e.g. Labels or Labels40.")
    parser.add_argument("--n-classes", type=int, default=41)
    parser.add_argument("--ignore-index", type=int, default=255, help="Void label id ignored by losses and metrics. Use 0 only when class 0 is not predicted.")
    parser.add_argument("--miou-start-class", type=int, default=1, help="First class id included in mIoU. Keep 1 to report SUN40 foreground mIoU while still predicting background 0.")
    parser.add_argument("--crop-size", type=int, default=None, help="Deprecated square input size. Prefer --input-height/--input-width.")
    parser.add_argument("--input-height", type=int, default=INPUT_HEIGHT)
    parser.add_argument("--input-width", type=int, default=INPUT_WIDTH)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--num-workers", type=int, default=8)
    parser.add_argument("--epochs", type=int, default=200)
    parser.add_argument("--lr", type=float, default=6e-5)
    parser.add_argument("--seed", type=int, default=3407)
    parser.add_argument("--ema-decay", type=float, default=0.9996)
    parser.add_argument("--ema-warmup-epochs", type=int, default=10, help="Validate raw model before this epoch, then switch to EMA if enabled.")
    parser.add_argument("--no-amp", action="store_true", help="Disable native torch.autocast mixed precision training.")
    parser.add_argument("--amp-dtype", type=str, default="fp16", choices=("fp16", "bf16"))
    parser.add_argument("--no-data-parallel", action="store_true")
    parser.add_argument("--use-tta", action="store_true", help="Use multi-scale + flip augmentation during validation.")
    parser.add_argument("--debug-val-stats", action="store_true", help="Print class histograms for the first validation batch.")
    parser.add_argument("--safe-mode", action="store_true", help="Use convolutional safe-mode instead of angle-guided GSA in the deepest fusion stage.")
    parser.add_argument("--lambda-lovasz", type=float, default=0.5)
    parser.add_argument("--lambda-dice", type=float, default=0.4)
    parser.add_argument("--lambda-edge", type=float, default=0.08)
    parser.add_argument("--lambda-boundary", type=float, default=0.03)
    parser.add_argument("--lambda-feature-precision", type=float, default=0.03)
    parser.add_argument("--hha-edge-weight", type=float, default=0.0)
    parser.add_argument("--max-edge-pos-weight", type=float, default=10.0)
    parser.add_argument("--ohem-min-kept", type=int, default=30000)
    parser.add_argument("--boundary-ce-weight", type=float, default=0.7)
    parser.add_argument("--loss-warmup-start", type=float, default=0.15)
    parser.add_argument("--edge-warmup-epochs", type=int, default=10)
    parser.add_argument("--boundary-warmup-epochs", type=int, default=14)
    parser.add_argument("--feature-warmup-epochs", type=int, default=18)
    parser.add_argument("--val-interval", type=int, default=2)
    parser.add_argument("--encoder-name", type=str, default="mit_b2")
    args = parser.parse_args()
    if args.crop_size is not None:
        args.input_height = args.crop_size
        args.input_width = args.crop_size
    main(args)
