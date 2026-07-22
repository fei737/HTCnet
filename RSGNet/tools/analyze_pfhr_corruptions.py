import argparse
import json
import os

import cv2
import numpy as np


CHANNEL_NAMES = ("disparity", "height", "angle")


def _resolve_directory(data_root, directory):
    return directory if os.path.isabs(directory) else os.path.join(data_root, directory)


def _parse_degraded(value):
    if "=" not in value:
        raise argparse.ArgumentTypeError("--degraded must use tag=directory syntax")
    tag, directory = value.split("=", 1)
    tag = tag.strip()
    directory = directory.strip()
    if not tag or not directory:
        raise argparse.ArgumentTypeError("--degraded requires non-empty tag and directory")
    return tag, directory


def _png_names(directory):
    if not os.path.isdir(directory):
        raise FileNotFoundError(f"HHA directory not found: {directory}")
    return sorted(name for name in os.listdir(directory) if name.lower().endswith(".png"))


def _read_hha(path):
    image = cv2.imread(path, cv2.IMREAD_COLOR)
    if image is None:
        raise ValueError(f"Failed to decode HHA image: {path}")
    return cv2.cvtColor(image, cv2.COLOR_BGR2RGB)


def _select_names(names, max_samples):
    if max_samples <= 0 or max_samples >= len(names):
        return names
    indices = np.linspace(0, len(names) - 1, max_samples, dtype=np.int64)
    return [names[int(index)] for index in np.unique(indices)]


def analyze(data_root, clean_dir, degraded_specs, max_samples):
    clean_path = _resolve_directory(data_root, clean_dir)
    degraded_paths = [
        (tag, _resolve_directory(data_root, directory), directory)
        for tag, directory in degraded_specs
    ]

    common_names = set(_png_names(clean_path))
    for _, directory_path, _ in degraded_paths:
        common_names.intersection_update(_png_names(directory_path))
    names = _select_names(sorted(common_names), max_samples)
    if not names:
        raise ValueError("No common PNG files were found across the requested HHA directories")

    clean_images = {
        name: _read_hha(os.path.join(clean_path, name))
        for name in names
    }
    results = {}
    for tag, directory_path, declared_directory in degraded_paths:
        absolute_delta = np.zeros(3, dtype=np.float64)
        channel_changed = np.zeros(3, dtype=np.float64)
        any_changed = 0
        clean_invalid = 0
        degraded_invalid = 0
        pixel_count = 0

        for name in names:
            clean = clean_images[name]
            degraded = _read_hha(os.path.join(directory_path, name))
            if degraded.shape != clean.shape:
                raise ValueError(
                    f"Shape mismatch for {name}: clean={clean.shape}, degraded={degraded.shape}"
                )
            delta = np.abs(degraded.astype(np.int16) - clean.astype(np.int16))
            absolute_delta += delta.sum(axis=(0, 1), dtype=np.float64)
            channel_changed += (delta > 0).sum(axis=(0, 1), dtype=np.float64)
            any_changed += int(np.any(delta > 0, axis=2).sum())
            clean_invalid += int((clean.max(axis=2) <= 1).sum())
            degraded_invalid += int((degraded.max(axis=2) <= 1).sum())
            pixel_count += int(clean.shape[0] * clean.shape[1])

        results[tag] = {
            "directory": declared_directory,
            "sample_count": len(names),
            "mean_absolute_code_delta": {
                channel: float(value)
                for channel, value in zip(CHANNEL_NAMES, absolute_delta / pixel_count)
            },
            "changed_pixel_percent_by_channel": {
                channel: float(value)
                for channel, value in zip(CHANNEL_NAMES, 100.0 * channel_changed / pixel_count)
            },
            "any_channel_changed_pixel_percent": 100.0 * any_changed / pixel_count,
            "invalid_pixel_delta_percentage_points": (
                100.0 * (degraded_invalid - clean_invalid) / pixel_count
            ),
        }

    return {
        "data_root": os.path.abspath(data_root),
        "clean_directory": clean_dir,
        "common_file_count": len(common_names),
        "analyzed_file_count": len(names),
        "first_file": names[0],
        "last_file": names[-1],
        "corruptions": results,
    }


def main():
    parser = argparse.ArgumentParser(
        description="Summarize channel-wise effects of raw-geometry PFHR corruptions."
    )
    parser.add_argument(
        "--data-root",
        default="/home/pengfei/HTCnet/DataSets/SUNRGBD",
        help="SUNRGBD directory containing clean and degraded HHA folders.",
    )
    parser.add_argument("--clean-dir", default="HHA_PFHR")
    parser.add_argument(
        "--degraded",
        action="append",
        type=_parse_degraded,
        required=True,
        help="Repeat as tag=directory for each physical corruption.",
    )
    parser.add_argument(
        "--max-samples",
        type=int,
        default=101,
        help="Uniformly sample this many common files; 0 analyzes all files.",
    )
    parser.add_argument("--output", default="")
    args = parser.parse_args()

    summary = analyze(
        args.data_root,
        args.clean_dir,
        args.degraded,
        max(0, args.max_samples),
    )
    rendered = json.dumps(summary, indent=2)
    print(rendered)
    if args.output:
        output_path = os.path.abspath(args.output)
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        with open(output_path, "w", encoding="utf-8") as file:
            file.write(rendered + "\n")


if __name__ == "__main__":
    main()
