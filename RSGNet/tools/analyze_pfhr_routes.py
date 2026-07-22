"""Quantify PFHR route behavior on a deterministic evaluation subset."""

import argparse
import json
import os

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Subset

from data import SUNDataset
from evaluate import load_checkpoint_state
from rsgnet import RSGNet
from utils import dilate_binary_mask, semantic_boundary


def _high_frequency_ratio(feature):
    low = F.avg_pool2d(feature, kernel_size=3, stride=1, padding=1)
    return (feature - low).abs().mean() / feature.abs().mean().clamp_min(1e-6)


def _build_model(config, checkpoint_path, device):
    model = RSGNet(
        n_classes=int(config.get("n_classes", 37)),
        pretrained_path=None,
        return_aux=False,
        encoder_name=config.get("encoder_name", "mit_b2"),
        prompt_channels=int(config.get("prompt_channels", 32)),
        decoder_channels=int(config.get("decoder_channels", 256)),
        attention_tokens=int(config.get("attention_tokens", 512)),
        local_scale_init=float(config.get("local_scale_init", 0.10)),
        semantic_scale_init=float(config.get("semantic_scale_init", 0.10)),
        fusion_mode=config.get("fusion_mode", "stagewise"),
        use_reliability=not bool(config.get("disable_reliability", False)),
        architecture_variant=config.get("architecture_variant", "factorized"),
        geometry_encoding=config.get("geometry_encoding", "factorized_routed"),
        geometry_channels=config.get("geometry_channels", "dha"),
    )
    state = load_checkpoint_state(
        checkpoint_path,
        model.state_dict(),
        architecture_variant=config.get("architecture_variant", "factorized"),
        geometry_encoding=config.get("geometry_encoding", "factorized_routed"),
        geometry_source=config.get("geometry_source", "hha"),
        geometry_channels=config.get("geometry_channels", "dha"),
    )
    model.load_state_dict(state, strict=False)
    return model.to(device).eval()


def _build_dataset(config):
    return SUNDataset(
        config.get("data_root", "/home/pengfei/HTCnet/DataSets"),
        dataset_name=config.get("dataset_name", "SUNRGBD"),
        mode="test",
        crop_size=(
            int(config.get("input_height", 480)),
            int(config.get("input_width", 480)),
        ),
        n_classes=int(config.get("n_classes", 37)),
        label_map_path=config.get("label_map", ""),
        label_dir_name=config.get("label_dir_name", "Labels"),
        hha_dir_name=config.get("hha_dir_name", "HHA"),
        hha_name_prefix=config.get("hha_name_prefix", ""),
        hha_channel_order=config.get("hha_channel_order", "dha"),
        geometry_source=config.get("geometry_source", "hha"),
        depth_dir_name=config.get("depth_dir_name", "Depth"),
        depth_name_prefix=config.get("depth_name_prefix", ""),
        depth_scale=float(config.get("depth_scale", 1000.0)),
        max_depth=float(config.get("max_depth", 10.0)),
        label_name_prefix=config.get("label_name_prefix", ""),
        ignore_index=int(config.get("ignore_index", 255)),
        split_policy=config.get("split_policy", "sunrgbd_official"),
        eval_native_size=False,
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--ckpt", required=True)
    parser.add_argument("--max-samples", type=int, default=500)
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--output", default="")
    args = parser.parse_args()

    checkpoint = torch.load(args.ckpt, map_location="cpu")
    config = checkpoint.get("config") if isinstance(checkpoint, dict) else None
    if not isinstance(config, dict):
        raise ValueError("PFHR route analysis requires a checkpoint with saved training config")
    if config.get("architecture_variant") != "factorized":
        raise ValueError("PFHR route analysis only supports factorized C-line checkpoints")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = _build_model(config, args.ckpt, device)
    dataset = _build_dataset(config)
    sample_count = min(max(1, args.max_samples), len(dataset))
    loader = DataLoader(
        Subset(dataset, range(sample_count)),
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=device.type == "cuda",
    )

    stage_stats = None
    reliability_sum = torch.zeros(2, device=device)
    reliability_count = 0
    channel_weight_sum = torch.zeros(3, device=device)
    channel_boundary_sum = torch.zeros(3, device=device)
    channel_interior_sum = torch.zeros(3, device=device)
    boundary_component_sum = torch.zeros(3, device=device)
    channel_weight_count = 0
    channel_boundary_count = 0
    channel_interior_count = 0
    boundary_component_count = 0
    with torch.no_grad():
        for rgb, hha, masks in loader:
            rgb = rgb.to(device, non_blocking=True)
            hha = hha.to(device, non_blocking=True)
            masks = masks.to(device, non_blocking=True)
            _, context = model(rgb, hha, return_context=True)
            routes = context["geometry_routes"]
            prompts = context["factorized_prompts"]
            if routes is None or prompts is None:
                raise RuntimeError("Checkpoint did not produce factorized routes and prompts")
            if stage_stats is None:
                stage_stats = [
                    {
                        "boundary_weight_on_boundary": 0.0,
                        "boundary_pixels": 0.0,
                        "boundary_weight_interior": 0.0,
                        "interior_pixels": 0.0,
                        "layout_weight": 0.0,
                        "route_pixels": 0.0,
                        "layout_hf_ratio": 0.0,
                        "boundary_hf_ratio": 0.0,
                        "batches": 0,
                    }
                    for _ in routes
                ]

            valid_full = masks != int(config.get("ignore_index", 255))
            reliability = context["geometry_reliability"]
            valid_reliability = valid_full.unsqueeze(1).expand_as(reliability)
            reliability_sum += (reliability * valid_reliability).sum(dim=(0, 2, 3))
            reliability_count += int(valid_full.sum().item())

            full_boundary = semantic_boundary(masks, valid_full)
            full_boundary_band = dilate_binary_mask(full_boundary, radius=1) & valid_full
            full_interior = valid_full & ~full_boundary_band
            channel_weights = context.get("geometry_channel_weights")
            if channel_weights is not None:
                if channel_weights.shape[2:] != masks.shape[1:]:
                    channel_weights = F.interpolate(
                        channel_weights,
                        size=masks.shape[1:],
                        mode="bilinear",
                        align_corners=False,
                    )
                channel_weight_sum += (
                    channel_weights * valid_full.unsqueeze(1)
                ).sum(dim=(0, 2, 3))
                channel_boundary_sum += (
                    channel_weights * full_boundary_band.unsqueeze(1)
                ).sum(dim=(0, 2, 3))
                channel_interior_sum += (
                    channel_weights * full_interior.unsqueeze(1)
                ).sum(dim=(0, 2, 3))
                channel_weight_count += int(valid_full.sum().item())
                channel_boundary_count += int(full_boundary_band.sum().item())
                channel_interior_count += int(full_interior.sum().item())

            boundary_weights = context.get("geometry_boundary_weights")
            if boundary_weights is not None:
                if boundary_weights.shape[2:] != masks.shape[1:]:
                    boundary_weights = F.interpolate(
                        boundary_weights,
                        size=masks.shape[1:],
                        mode="bilinear",
                        align_corners=False,
                    )
                boundary_component_sum += (
                    boundary_weights * valid_full.unsqueeze(1)
                ).sum(dim=(0, 2, 3))
                boundary_component_count += int(valid_full.sum().item())

            for index, route in enumerate(routes):
                if route is None:
                    continue
                target_size = route.shape[2:]
                resized_masks = F.interpolate(
                    masks.unsqueeze(1).float(),
                    size=target_size,
                    mode="nearest",
                ).squeeze(1).long()
                valid = resized_masks != int(config.get("ignore_index", 255))
                boundary = semantic_boundary(resized_masks, valid)
                boundary_band = dilate_binary_mask(boundary, radius=1) & valid
                interior = valid & ~boundary_band
                boundary_route = route[:, 1]
                layout_route = route[:, 0]
                stats = stage_stats[index]
                stats["boundary_weight_on_boundary"] += float(
                    (boundary_route * boundary_band).sum()
                )
                stats["boundary_pixels"] += float(boundary_band.sum())
                stats["boundary_weight_interior"] += float(
                    (boundary_route * interior).sum()
                )
                stats["interior_pixels"] += float(interior.sum())
                stats["layout_weight"] += float((layout_route * valid).sum())
                stats["route_pixels"] += float(valid.sum())
                stats["layout_hf_ratio"] += float(
                    _high_frequency_ratio(prompts["layout"][index])
                )
                stats["boundary_hf_ratio"] += float(
                    _high_frequency_ratio(prompts["boundary"][index])
                )
                stats["batches"] += 1

    output_stages = []
    for index, stats in enumerate(stage_stats or []):
        output_stages.append(
            {
                "stage": index + 1,
                "boundary_weight_on_boundary": stats["boundary_weight_on_boundary"]
                / max(stats["boundary_pixels"], 1.0),
                "boundary_weight_interior": stats["boundary_weight_interior"]
                / max(stats["interior_pixels"], 1.0),
                "layout_weight_mean": stats["layout_weight"]
                / max(stats["route_pixels"], 1.0),
                "layout_prompt_hf_ratio": stats["layout_hf_ratio"]
                / max(stats["batches"], 1),
                "boundary_prompt_hf_ratio": stats["boundary_hf_ratio"]
                / max(stats["batches"], 1),
            }
        )
    result = {
        "checkpoint": os.path.abspath(args.ckpt),
        "samples": sample_count,
        "geometry_encoding": config.get("geometry_encoding"),
        "geometry_channels": config.get("geometry_channels", "dha"),
        "mean_layout_reliability": float(reliability_sum[0] / max(reliability_count, 1)),
        "mean_boundary_reliability": float(reliability_sum[1] / max(reliability_count, 1)),
        "hha_channel_weight_mean": {
            name: float(channel_weight_sum[index] / max(channel_weight_count, 1))
            for index, name in enumerate(("disparity", "height", "angle"))
        } if channel_weight_count > 0 else None,
        "hha_channel_weight_on_boundary": {
            name: float(channel_boundary_sum[index] / max(channel_boundary_count, 1))
            for index, name in enumerate(("disparity", "height", "angle"))
        } if channel_boundary_count > 0 else None,
        "hha_channel_weight_interior": {
            name: float(channel_interior_sum[index] / max(channel_interior_count, 1))
            for index, name in enumerate(("disparity", "height", "angle"))
        } if channel_interior_count > 0 else None,
        "boundary_component_weight_mean": {
            name: float(boundary_component_sum[index] / max(boundary_component_count, 1))
            for index, name in enumerate(("disparity", "height", "angle"))
        } if boundary_component_count > 0 else None,
        "stages": output_stages,
    }
    print(json.dumps(result, indent=2))
    if args.output:
        output_path = os.path.abspath(args.output)
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        with open(output_path, "w", encoding="utf-8") as file:
            json.dump(result, file, indent=2)


if __name__ == "__main__":
    main()
