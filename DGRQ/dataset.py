import json
import os
import random

import cv2
import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset


INPUT_HEIGHT = 480
INPUT_WIDTH = 640
RGB_MEAN = (0.485, 0.456, 0.406)
RGB_STD = (0.229, 0.224, 0.225)
HHA_MEAN = (0.5, 0.5, 0.5)
HHA_STD = (0.5, 0.5, 0.5)


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


def sanitize_mask_values(mask, n_classes, ignore_index=255):
    invalid_mask = (mask != ignore_index) & ((mask < 0) | (mask >= n_classes))
    invalid_count = int(invalid_mask.sum()) if isinstance(invalid_mask, np.ndarray) else int(invalid_mask.sum().item())
    if invalid_count > 0:
        mask = mask.copy() if isinstance(mask, np.ndarray) else mask.clone()
        mask[invalid_mask] = ignore_index
    return mask, invalid_count


def get_angle_grad_target(hha):
    x = hha[:, 2:3, :, :]
    local_mean = F.avg_pool2d(x, kernel_size=3, stride=1, padding=1)
    local_var = F.avg_pool2d((x - local_mean).pow(2), kernel_size=3, stride=1, padding=1)
    local_max = F.max_pool2d(x, kernel_size=3, stride=1, padding=1)
    local_min = -F.max_pool2d(-x, kernel_size=3, stride=1, padding=1)
    local_contrast = 0.5 * ((x - local_mean).abs() + (local_max - local_min))
    confidence = torch.exp(-local_var.sqrt()).clamp(0.0, 1.0)
    grad = local_contrast * confidence
    threshold = grad.flatten(1).quantile(0.85, dim=1).view(-1, 1, 1, 1).clamp_min(0.05)
    return (grad > threshold).float()


def normalize_modalities(
    rgb,
    hha,
    device=None,
    rgb_mean=RGB_MEAN,
    rgb_std=RGB_STD,
    hha_mean=HHA_MEAN,
    hha_std=HHA_STD,
):
    rgb_t = torch.from_numpy(rgb).permute(2, 0, 1).float() / 255.0
    hha_t = torch.from_numpy(hha).permute(2, 0, 1).float() / 255.0

    rgb_mean_t = torch.tensor(rgb_mean).view(3, 1, 1).float()
    rgb_std_t = torch.tensor(rgb_std).view(3, 1, 1).float()
    hha_mean_t = torch.tensor(hha_mean).view(3, 1, 1).float()
    hha_std_t = torch.tensor(hha_std).view(3, 1, 1).float()

    rgb_t = (rgb_t - rgb_mean_t) / rgb_std_t
    hha_t = (hha_t - hha_mean_t) / hha_std_t

    rgb_t = rgb_t.unsqueeze(0)
    hha_t = hha_t.unsqueeze(0)
    if device is not None:
        rgb_t = rgb_t.to(device)
        hha_t = hha_t.to(device)
    return rgb_t, hha_t


class SUNDataset(Dataset):
    def __init__(
        self,
        root_dir,
        dataset_name="SUNRGBD",
        mode="train",
        crop_size=(INPUT_HEIGHT, INPUT_WIDTH),
        rgb_mean=RGB_MEAN,
        rgb_std=RGB_STD,
        hha_mean=HHA_MEAN,
        hha_std=HHA_STD,
        n_classes=41,
        label_map_path="",
        label_dir_name="Labels",
        hha_name_prefix="",
        label_name_prefix="",
        ignore_index=255,
        train_cutout_prob=0.5,
        aug_level="base",
    ):
        self.dataset_name = dataset_name
        self.root_dir = os.path.join(root_dir, dataset_name)
        self.mode = mode
        self.aug_level = aug_level
        self.crop_h, self.crop_w = _to_hw(crop_size)
        self.label_dir_name = label_dir_name
        self.hha_name_prefix = hha_name_prefix
        self.label_name_prefix = label_name_prefix
        self.n_classes = n_classes
        self.ignore_index = ignore_index
        self.train_cutout_prob = float(max(0.0, min(1.0, train_cutout_prob)))
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
                f"{self.dataset_name} dataset is incomplete under {self.root_dir}. Missing folders: {missing_dirs}. "
                f"Expected structure: {self.dataset_name}/RGB, {self.dataset_name}/HHA, "
                f"{self.dataset_name}/{self.label_dir_name}, {self.dataset_name}/train.txt, "
                f"{self.dataset_name}/test.txt."
            )
        if not os.path.isfile(list_path):
            raise FileNotFoundError(
                f"Official split file not found: {list_path}. "
                f"For academic reproducibility, prepare {self.dataset_name}/train.txt and "
                f"{self.dataset_name}/test.txt instead of random splitting."
            )

        with open(list_path, "r", encoding="utf-8") as f:
            self.image_ids = [self._normalize_image_id(line.strip()) for line in f.readlines() if line.strip()]
        if not self.image_ids:
            raise ValueError(f"{self.dataset_name} split file is empty: {list_path}")

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
        hha_path = os.path.join(self.root_dir, "HHA", f"{self.hha_name_prefix}{base_name}.png")
        label_path = os.path.join(self.root_dir, self.label_dir_name, f"{self.label_name_prefix}{base_name}.png")

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

            strong = self.aug_level == "strong"

            if random.random() > 0.5:
                hsv = cv2.cvtColor(rgb, cv2.COLOR_RGB2HSV)
                hsv = np.array(hsv, dtype=np.float64)
                sat_lo, sat_hi = (0.6, 1.4) if strong else (0.7, 1.3)
                val_lo, val_hi = (0.6, 1.4) if strong else (0.7, 1.3)
                hsv[:, :, 1] *= random.uniform(sat_lo, sat_hi)
                hsv[:, :, 2] *= random.uniform(val_lo, val_hi)
                if strong:
                    # Hue jitter (OpenCV hue range is [0, 180)).
                    hsv[:, :, 0] = (hsv[:, :, 0] + random.uniform(-10.0, 10.0)) % 180.0
                hsv[:, :, 1][hsv[:, :, 1] > 255] = 255
                hsv[:, :, 2][hsv[:, :, 2] > 255] = 255
                hsv = np.array(hsv, dtype=np.uint8)
                rgb = cv2.cvtColor(hsv, cv2.COLOR_HSV2RGB)

            if strong and random.random() > 0.5:
                # Linear brightness/contrast jitter on RGB only (depth/HHA left intact).
                alpha = random.uniform(0.8, 1.2)  # contrast
                beta = random.uniform(-15.0, 15.0)  # brightness
                rgb = np.clip(rgb.astype(np.float32) * alpha + beta, 0, 255).astype(np.uint8)

            scale_lo, scale_hi = (0.5, 2.0) if strong else (0.75, 1.75)
            scale = random.uniform(scale_lo, scale_hi)
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

            if random.random() < self.train_cutout_prob:
                num_cuts = random.randint(1, 3) if strong else 1
                for _ in range(num_cuts):
                    cut_h = random.randint(40, 120)
                    cut_w = random.randint(40, 120)
                    cut_y = random.randint(0, self.crop_h - cut_h)
                    cut_x = random.randint(0, self.crop_w - cut_w)
                    # Erase RGB and HHA together, and void the label region so the
                    # occluded pixels are ignored by the loss rather than learned as
                    # whatever class happened to be underneath.
                    rgb[cut_y:cut_y + cut_h, cut_x:cut_x + cut_w, :] = 0
                    hha[cut_y:cut_y + cut_h, cut_x:cut_x + cut_w, :] = 0
                    label[cut_y:cut_y + cut_h, cut_x:cut_x + cut_w] = self.ignore_index
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
