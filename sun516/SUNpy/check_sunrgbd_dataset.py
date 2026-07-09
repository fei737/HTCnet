import argparse
import os
from collections import Counter

import cv2
import numpy as np


def normalize_id(sample_id):
    sample_id = sample_id.replace("\\", "/").split("/")[-1].strip()
    return sample_id if os.path.splitext(sample_id)[1] else f"{sample_id}.png"


def read_split(split_path):
    if not os.path.isfile(split_path):
        raise FileNotFoundError(f"Split file not found: {split_path}")
    with open(split_path, "r", encoding="utf-8") as f:
        ids = [normalize_id(line) for line in f if line.strip()]
    if not ids:
        raise ValueError(f"Split file is empty: {split_path}")
    return ids


def summarize_counter(counter, max_items=20):
    items = sorted(counter.items(), key=lambda item: item[1], reverse=True)
    return ", ".join(f"{int(k)}:{int(v)}" for k, v in items[:max_items])


def summarize_shapes(shape_counter, max_items=5):
    parts = []
    for (rgb_hw, hha_hw, label_hw), count in shape_counter.most_common(max_items):
        parts.append(f"RGB={rgb_hw}, HHA={hha_hw}, Label={label_hw}: {count}")
    return " | ".join(parts)


def inspect_split(root, split, label_dir_name, n_classes, max_samples):
    split_path = os.path.join(root, f"{split}.txt")
    sample_ids = read_split(split_path)
    if max_samples > 0:
        sample_ids = sample_ids[:max_samples]

    stats = {
        "samples": 0,
        "missing_rgb": 0,
        "missing_hha": 0,
        "missing_label": 0,
        "bad_read": 0,
        "shape_mismatch": 0,
        "color_label": 0,
        "out_of_range": 0,
        "empty_valid": 0,
    }
    class_hist = Counter()
    image_shapes = Counter()
    examples = []

    for image_name in sample_ids:
        stats["samples"] += 1
        base = os.path.splitext(image_name)[0]
        rgb_path = os.path.join(root, "RGB", image_name)
        hha_path = os.path.join(root, "HHA", f"{base}.png")
        label_path = os.path.join(root, label_dir_name, f"{base}.png")

        if not os.path.isfile(rgb_path):
            stats["missing_rgb"] += 1
            examples.append(f"missing RGB: {rgb_path}")
            continue
        if not os.path.isfile(hha_path):
            stats["missing_hha"] += 1
            examples.append(f"missing HHA: {hha_path}")
            continue
        if not os.path.isfile(label_path):
            stats["missing_label"] += 1
            examples.append(f"missing label: {label_path}")
            continue

        rgb = cv2.imread(rgb_path, cv2.IMREAD_COLOR)
        hha = cv2.imread(hha_path, cv2.IMREAD_COLOR)
        label = cv2.imread(label_path, cv2.IMREAD_UNCHANGED)
        if rgb is None or hha is None or label is None:
            stats["bad_read"] += 1
            examples.append(f"failed read: {base}")
            continue

        if label.ndim == 3:
            stats["color_label"] += 1
            examples.append(f"label appears color-coded: {label_path}")
            continue

        rgb_hw = rgb.shape[:2]
        hha_hw = hha.shape[:2]
        label_hw = label.shape[:2]
        image_shapes[(rgb_hw, hha_hw, label_hw)] += 1
        if rgb_hw != hha_hw or rgb_hw != label_hw:
            stats["shape_mismatch"] += 1
            examples.append(f"shape mismatch {base}: RGB={rgb_hw}, HHA={hha_hw}, Label={label_hw}")

        label = label.astype(np.int64)
        bad = (label < 0) | (label >= n_classes)
        if np.any(bad):
            stats["out_of_range"] += 1
            bad_values = np.unique(label[bad])[:20].tolist()
            examples.append(f"label out of range {base}: {bad_values}")

        valid = label != 0
        if not np.any(valid):
            stats["empty_valid"] += 1
            examples.append(f"no valid foreground pixels: {label_path}")

        values, counts = np.unique(label, return_counts=True)
        for value, count in zip(values.tolist(), counts.tolist()):
            class_hist[int(value)] += int(count)

    return stats, class_hist, image_shapes, examples[:30]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-root", type=str, default="/home/pengfei/HTCnet/DataSets")
    parser.add_argument("--label-dir-name", type=str, default="Labels")
    parser.add_argument("--n-classes", type=int, default=41)
    parser.add_argument("--max-samples", type=int, default=0, help="0 means check all samples.")
    args = parser.parse_args()

    root = os.path.join(args.data_root, "SUNRGBD")
    required_dirs = ["RGB", "HHA", args.label_dir_name]
    print(f"[check] SUNRGBD root: {root}")
    for name in required_dirs:
        path = os.path.join(root, name)
        print(f"[check] {name}: {'OK' if os.path.isdir(path) else 'MISSING'} -> {path}")
    for split in ("train", "test"):
        print("\n" + "=" * 70)
        print(f"[check] Split: {split}")
        stats, class_hist, image_shapes, examples = inspect_split(
            root,
            split,
            args.label_dir_name,
            args.n_classes,
            args.max_samples,
        )
        for key, value in stats.items():
            print(f"[check] {key}: {value}")
        print(f"[check] common shapes: {summarize_shapes(image_shapes)}")
        print(f"[check] label hist top20: {summarize_counter(class_hist)}")
        if examples:
            print("[check] examples:")
            for item in examples:
                print(f"  - {item}")

    print("\n[check] Done.")


if __name__ == "__main__":
    main()
