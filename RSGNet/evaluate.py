import argparse
import json
import os

import torch
import torch.distributed as dist
from torch.utils.data import DataLoader, Sampler

from data import (
    INPUT_HEIGHT,
    INPUT_WIDTH,
    SUNDataset,
    apply_protocol,
    parse_scales,
    protocol_class_names,
)
from rsgnet import RSGNet
from utils import cleanup_distributed, is_dist_initialized, is_main_process, validate


class DistributedEvalSampler(Sampler):
    """Shard evaluation data across ranks without padding or duplication."""

    def __init__(self, dataset, num_replicas, rank):
        self.dataset = dataset
        self.num_replicas = int(num_replicas)
        self.rank = int(rank)

    def __iter__(self):
        return iter(range(self.rank, len(self.dataset), self.num_replicas))

    def __len__(self):
        remaining = len(self.dataset) - self.rank
        return max(0, (remaining + self.num_replicas - 1) // self.num_replicas)


def strip_module_prefix(state_dict):
    stripped = {}
    for key, value in state_dict.items():
        if key == "n_averaged":
            continue
        while key.startswith("module."):
            key = key[7:]
        stripped[key] = value
    return stripped


def load_checkpoint_state(
    ckpt_path,
    model_state_dict,
    architecture_variant="legacy",
    geometry_encoding="factorized_routed",
    geometry_source="hha",
    geometry_channels="dha",
):
    checkpoint = torch.load(ckpt_path, map_location="cpu")
    if isinstance(checkpoint, dict):
        checkpoint_config = checkpoint.get("config")
        checkpoint_variant = (
            checkpoint_config.get("architecture_variant", "legacy")
            if isinstance(checkpoint_config, dict)
            else "legacy"
        )
        if checkpoint_variant != architecture_variant:
            raise ValueError(
                "Checkpoint architecture mismatch: "
                f"checkpoint={checkpoint_variant!r}, requested={architecture_variant!r}."
            )
        if architecture_variant == "factorized" and isinstance(checkpoint_config, dict):
            checkpoint_encoding = checkpoint_config.get(
                "geometry_encoding", "factorized_routed"
            )
            if checkpoint_encoding != geometry_encoding:
                raise ValueError(
                    "Checkpoint geometry encoding mismatch: "
                    f"checkpoint={checkpoint_encoding!r}, requested={geometry_encoding!r}."
                )
            checkpoint_source = checkpoint_config.get("geometry_source", "hha")
            if checkpoint_source != geometry_source:
                raise ValueError(
                    "Checkpoint geometry source mismatch: "
                    f"checkpoint={checkpoint_source!r}, requested={geometry_source!r}."
                )
            checkpoint_channels = checkpoint_config.get("geometry_channels", "dha")
            if checkpoint_channels != geometry_channels:
                raise ValueError(
                    "Checkpoint geometry channel mask mismatch: "
                    f"checkpoint={checkpoint_channels!r}, requested={geometry_channels!r}."
                )
        state_dict = checkpoint.get("ema_state_dict") or checkpoint.get("model_state_dict") or checkpoint
    else:
        state_dict = checkpoint
    state_dict = strip_module_prefix(state_dict)

    filtered_state_dict = {}
    for key, value in state_dict.items():
        if key in model_state_dict and model_state_dict[key].shape != value.shape:
            print(
                f"[*] Skip shape-mismatch weight: {key} | "
                f"model={tuple(model_state_dict[key].shape)} ckpt={tuple(value.shape)}"
            )
            continue
        filtered_state_dict[key] = value
    return filtered_state_dict


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-root", type=str, required=True)
    parser.add_argument("--dataset-name", type=str, default="SUNRGBD", help="Dataset folder under --data-root, e.g. SUNRGBD or NYU.")
    parser.add_argument(
        "--protocol",
        type=str,
        default="sunrgbd37",
        choices=("legacy", "sunrgbd37", "sunrgbd37_dformerv2"),
    )
    parser.add_argument("--ckpt", type=str, required=True)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--n-classes", type=int, default=41)
    parser.add_argument("--crop-size", type=int, default=None, help="Deprecated square input size. Prefer --input-height/--input-width.")
    parser.add_argument("--input-height", type=int, default=INPUT_HEIGHT)
    parser.add_argument("--input-width", type=int, default=INPUT_WIDTH)
    parser.add_argument("--allow-custom-input-size", action="store_true")
    parser.add_argument("--label-map", type=str, default="")
    parser.add_argument("--split-policy", type=str, default="file", choices=("file", "sunrgbd_official"))
    parser.add_argument("--label-dir-name", type=str, default="Labels")
    parser.add_argument("--hha-dir-name", type=str, default="HHA")
    parser.add_argument("--hha-name-prefix", type=str, default="", help="Optional filename prefix for HHA images, e.g. 'hha_' for NYU.")
    parser.add_argument(
        "--hha-channel-order",
        type=str,
        default="dha",
        choices=("dha", "ahd"),
    )
    parser.add_argument("--geometry-source", type=str, default="hha", choices=("hha", "depth"))
    parser.add_argument("--depth-dir-name", type=str, default="Depth")
    parser.add_argument("--depth-name-prefix", type=str, default="")
    parser.add_argument("--depth-scale", type=float, default=1000.0)
    parser.add_argument("--max-depth", type=float, default=10.0)
    parser.add_argument("--label-name-prefix", type=str, default="", help="Optional filename prefix for label images.")
    parser.add_argument("--ignore-index", type=int, default=0)
    parser.add_argument("--miou-start-class", type=int, default=1)
    parser.add_argument("--encoder-name", type=str, default="mit_b2")
    parser.add_argument("--use-tta", action="store_true")
    parser.add_argument("--tta-scales", type=str, default="0.75,1.0,1.25")
    parser.add_argument("--eval-native-size", action="store_true")
    parser.add_argument("--sliding-eval", action="store_true")
    parser.add_argument("--eval-crop-height", type=int, default=480)
    parser.add_argument("--eval-crop-width", type=int, default=480)
    parser.add_argument("--eval-stride-rate", type=float, default=2.0 / 3.0)
    parser.add_argument("--eval-pad-height", type=int, default=0)
    parser.add_argument("--eval-pad-width", type=int, default=0)
    parser.add_argument("--tta-fusion", choices=("logits", "probabilities"), default="logits")
    parser.add_argument("--tta-align-corners", action="store_true")
    parser.add_argument("--tta-size-mode", choices=("legacy", "dformerv2"), default="legacy")
    parser.add_argument("--tta-small-image-mode", choices=("pad", "resize"), default="pad")
    parser.add_argument("--eval-hha-degrade-mode", type=str, default="clean",
                        choices=("clean", "dropout", "noise", "shift"))
    parser.add_argument("--eval-hha-degrade-severity", type=float, default=0.0)
    parser.add_argument("--eval-hha-degrade-seed", type=int, default=3407)
    parser.add_argument("--metrics-json", type=str, default="")
    parser.add_argument("--prompt-channels", type=int, default=32)
    parser.add_argument("--decoder-channels", type=int, default=256)
    parser.add_argument("--attention-tokens", type=int, default=1024)
    parser.add_argument("--local-scale-init", type=float, default=0.05)
    parser.add_argument("--semantic-scale-init", type=float, default=0.05)
    parser.add_argument(
        "--fusion-mode",
        type=str,
        default="stagewise",
        choices=("rgb", "prompt", "shallow", "deep", "stagewise", "topology"),
    )
    parser.add_argument(
        "--architecture-variant",
        type=str,
        default="legacy",
        choices=("legacy", "refined", "factorized"),
    )
    parser.add_argument(
        "--geometry-encoding",
        type=str,
        default="factorized_routed",
        choices=(
            "unified",
            "independent",
            "factorized_static",
            "factorized_routed",
            "factorized_swapped",
            "factorized_final",
            "factorized_final_soft",
            "factorized_channel_routed",
        ),
    )
    parser.add_argument(
        "--geometry-channels",
        type=str,
        default="dha",
        choices=("d", "h", "a", "dh", "da", "ha", "dha"),
    )
    parser.add_argument("--disable-reliability", action="store_true")
    args = parser.parse_args()
    if args.crop_size is not None:
        args.input_height = args.crop_size
        args.input_width = args.crop_size
    apply_protocol(args)

    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    use_ddp = world_size > 1
    if use_ddp:
        backend = "nccl" if torch.cuda.is_available() else "gloo"
        if torch.cuda.is_available():
            torch.cuda.set_device(local_rank)
        dist.init_process_group(backend=backend, init_method="env://")

    device = torch.device(f"cuda:{local_rank}" if torch.cuda.is_available() else "cpu")
    if is_main_process():
        print(f"[*] Target device: {device}")
        if use_ddp:
            print(f"[*] Distributed validation enabled: world_size={world_size}")
    amp_enabled = device.type == "cuda"
    amp_dtype = torch.bfloat16 if (device.type == "cuda" and torch.cuda.is_bf16_supported()) else torch.float16

    model = RSGNet(
        n_classes=args.n_classes,
        pretrained_path=None,
        return_aux=False,
        encoder_name=args.encoder_name,
        prompt_channels=args.prompt_channels,
        decoder_channels=args.decoder_channels,
        attention_tokens=args.attention_tokens,
        local_scale_init=args.local_scale_init,
        semantic_scale_init=args.semantic_scale_init,
        fusion_mode=args.fusion_mode,
        use_reliability=not args.disable_reliability,
        architecture_variant=args.architecture_variant,
        geometry_encoding=args.geometry_encoding,
        geometry_channels=args.geometry_channels,
    )
    state_dict = load_checkpoint_state(
        args.ckpt,
        model.state_dict(),
        architecture_variant=args.architecture_variant,
        geometry_encoding=args.geometry_encoding,
        geometry_source=args.geometry_source,
        geometry_channels=args.geometry_channels,
    )
    missing, unexpected = model.load_state_dict(state_dict, strict=False)
    model.to(device).eval()
    if is_main_process():
        print(f"[*] Loaded weights from: {os.path.abspath(args.ckpt)}")
        print(f"[*] Missing keys: {len(missing)} | Unexpected keys: {len(unexpected)}")

    val_dataset = SUNDataset(
        args.data_root,
        dataset_name=args.dataset_name,
        mode="test",
        crop_size=(args.input_height, args.input_width),
        n_classes=args.n_classes,
        label_map_path=args.label_map,
        label_dir_name=args.label_dir_name,
        hha_dir_name=args.hha_dir_name,
        hha_name_prefix=args.hha_name_prefix,
        hha_channel_order=args.hha_channel_order,
        geometry_source=args.geometry_source,
        depth_dir_name=args.depth_dir_name,
        depth_name_prefix=args.depth_name_prefix,
        depth_scale=args.depth_scale,
        max_depth=args.max_depth,
        label_name_prefix=args.label_name_prefix,
        ignore_index=args.ignore_index,
        split_policy=args.split_policy,
        eval_native_size=args.eval_native_size,
        eval_pad_size=(args.eval_pad_height, args.eval_pad_width)
        if args.eval_pad_height > 0 and args.eval_pad_width > 0
        else None,
        eval_hha_degrade_mode=args.eval_hha_degrade_mode,
        eval_hha_degrade_severity=args.eval_hha_degrade_severity,
        eval_hha_degrade_seed=args.eval_hha_degrade_seed,
    )
    val_sampler = None
    if use_ddp:
        val_sampler = DistributedEvalSampler(
            val_dataset,
            num_replicas=world_size,
            rank=dist.get_rank(),
        )
    val_loader = DataLoader(
        val_dataset,
        batch_size=1 if args.eval_native_size else args.batch_size,
        shuffle=False,
        sampler=val_sampler,
        num_workers=args.num_workers,
        pin_memory=device.type == "cuda",
    )

    scales = parse_scales(args.tta_scales) if args.use_tta else (1.0,)
    metrics = validate(
        model,
        val_loader,
        device=device,
        n_classes=args.n_classes,
        scales=scales,
        use_flip=args.use_tta,
        ignore_index=args.ignore_index,
        miou_start_class=args.miou_start_class,
        amp_enabled=amp_enabled,
        amp_dtype=amp_dtype,
        crop_size=(args.eval_crop_height, args.eval_crop_width) if args.sliding_eval else None,
        stride_rate=args.eval_stride_rate,
        tta_fusion=args.tta_fusion,
        tta_align_corners=args.tta_align_corners,
        tta_size_mode=args.tta_size_mode,
        tta_small_image_mode=args.tta_small_image_mode,
        return_details=True,
    )
    miou = metrics["miou"]
    class_names = protocol_class_names(args.protocol, args.n_classes)
    metrics.update({
        "protocol": args.protocol,
        "architecture_variant": args.architecture_variant,
        "geometry_encoding": args.geometry_encoding,
        "hha_channel_order": args.hha_channel_order,
        "geometry_source": args.geometry_source,
        "geometry_channels": args.geometry_channels,
        "fusion_mode": args.fusion_mode,
        "reliability_enabled": not args.disable_reliability,
        "scales": list(scales),
        "flip": bool(args.use_tta),
        "tta_fusion": args.tta_fusion,
        "tta_align_corners": bool(args.tta_align_corners),
        "tta_size_mode": args.tta_size_mode,
        "tta_small_image_mode": args.tta_small_image_mode,
        "eval_pad_size": [args.eval_pad_height, args.eval_pad_width],
        "class_names": list(class_names),
        "hha_degradation": {
            "mode": args.eval_hha_degrade_mode,
            "severity": args.eval_hha_degrade_severity,
            "seed": args.eval_hha_degrade_seed,
        },
    })
    if is_main_process():
        print("\n" + "=" * 50)
        print(f"Final Validation mIoU: {miou:.4f}")
        print(
            f"Interior mIoU: {metrics['interior_miou']:.4f} | "
            f"Boundary F1: {metrics['boundary_f1']:.4f}"
        )
        print(
            "HHA degradation: "
            f"mode={args.eval_hha_degrade_mode} severity={args.eval_hha_degrade_severity:g}"
        )
        for index, (name, iou) in enumerate(zip(class_names, metrics["class_iou"])):
            value = "n/a" if iou is None else f"{iou:.4f}"
            print(f"  {index:02d} {name:<18} {value}")
        print("=" * 50)
        if args.metrics_json:
            metrics_dir = os.path.dirname(os.path.abspath(args.metrics_json))
            os.makedirs(metrics_dir, exist_ok=True)
            with open(args.metrics_json, "w", encoding="utf-8") as file:
                json.dump(metrics, file, indent=2)
            print(f"[*] Metrics JSON: {os.path.abspath(args.metrics_json)}")

    if is_dist_initialized():
        dist.barrier()
    cleanup_distributed()


if __name__ == "__main__":
    main()
