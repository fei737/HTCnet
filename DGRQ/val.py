import argparse
import os

import torch
from torch.utils.data import DataLoader

from dataset import INPUT_HEIGHT, INPUT_WIDTH, SUNDataset
from inference import strip_module_prefix
from models import UGFLiteNet, remap_dgrq_state_dict
from utils import validate


def load_checkpoint_state(ckpt_path, model_state_dict):
    checkpoint = torch.load(ckpt_path, map_location="cpu")
    if isinstance(checkpoint, dict):
        state_dict = checkpoint.get("ema_state_dict") or checkpoint.get("model_state_dict") or checkpoint
    else:
        state_dict = checkpoint
    state_dict = remap_dgrq_state_dict(strip_module_prefix(state_dict))

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
    parser.add_argument("--ckpt", type=str, required=True)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--n-classes", type=int, default=41)
    parser.add_argument("--crop-size", type=int, default=None, help="Deprecated square input size. Prefer --input-height/--input-width.")
    parser.add_argument("--input-height", type=int, default=INPUT_HEIGHT)
    parser.add_argument("--input-width", type=int, default=INPUT_WIDTH)
    parser.add_argument("--label-map", type=str, default="")
    parser.add_argument("--label-dir-name", type=str, default="Labels")
    parser.add_argument("--hha-name-prefix", type=str, default="", help="Optional filename prefix for HHA images, e.g. 'hha_' for NYU.")
    parser.add_argument("--label-name-prefix", type=str, default="", help="Optional filename prefix for label images.")
    parser.add_argument("--ignore-index", type=int, default=0)
    parser.add_argument("--miou-start-class", type=int, default=1)
    parser.add_argument("--encoder-name", type=str, default="mit_b2")
    parser.add_argument("--use-tta", action="store_true")
    parser.add_argument("--safe-mode", action="store_true")
    parser.add_argument("--prompt-channels", type=int, default=32)
    parser.add_argument("--layout-state-dim", type=int, default=16)
    parser.add_argument("--geometry-routing-mode", type=str, default="rcfr", choices=("legacy", "rcfr"))
    parser.add_argument("--reliability-strength", type=float, default=0.30)
    parser.add_argument("--prompt-recovery-mode", type=str, default="rgpr", choices=("legacy", "rgpr"),
                        help="Prompt recovery variant; must match the trained checkpoint.")
    parser.add_argument("--consistency-routing-mode", type=str, default="gacr", choices=("none", "gacr"),
                        help="Consistency routing variant; must match the trained checkpoint.")
    parser.add_argument("--fusion-branch-mode", type=str, default="both", choices=("both", "local", "semantic", "rgb"),
                        help="Fusion branch variant; must match the trained checkpoint.")
    parser.add_argument("--no-prompt-autocorr", action="store_true",
                        help="Disable reliability-aware autocorrelation prompt mixing; must match the trained checkpoint.")
    parser.add_argument("--no-directional-edge-refine", action="store_true",
                        help="Disable directional geometry edge refinement; must match the trained checkpoint.")
    parser.add_argument("--decoder-channels", type=int, default=256)
    parser.add_argument("--decoder-mode", type=str, default="simple_mlp", choices=("qdhs", "simple_mlp"),
                        help="Decoder variant; must match the trained checkpoint.")
    parser.add_argument("--attention-tokens", type=int, default=1024)
    parser.add_argument("--local-scale-init", type=float, default=1e-4)
    parser.add_argument("--semantic-scale-init", type=float, default=1e-4)
    parser.add_argument("--query-points", type=int, default=4)
    parser.add_argument("--query-scale-init", type=float, default=0.05)
    parser.add_argument("--geometry-scale-init", type=float, default=0.1)
    parser.add_argument("--detail-logit-scale", type=float, default=0.3)
    parser.add_argument("--rgb-boundary-scale", type=float, default=0.1)
    parser.add_argument("--no-prompt-recovery", action="store_true",
                        help="Disable bounded HHA prompt recovery; must match the trained checkpoint.")
    parser.add_argument("--no-confidence-routing", action="store_true",
                        help="Disable reliability-gated geometry routing; must match the trained checkpoint.")
    parser.add_argument("--layout-mode", type=str, default="ssm", choices=("ssm", "conv", "avgpool"),
                        help="Geometry prompt layout stream variant; must match the trained checkpoint.")
    args = parser.parse_args()
    if args.crop_size is not None:
        args.input_height = args.crop_size
        args.input_width = args.crop_size

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[*] Target device: {device}")
    amp_enabled = device.type == "cuda"
    amp_dtype = torch.bfloat16 if (device.type == "cuda" and torch.cuda.is_bf16_supported()) else torch.float16

    model = UGFLiteNet(
        n_classes=args.n_classes,
        pretrained_path=None,
        return_aux=False,
        encoder_name=args.encoder_name,
        safe_mode=args.safe_mode,
        layout_mode=args.layout_mode,
        decoder_mode=args.decoder_mode,
        use_prompt_recovery=not args.no_prompt_recovery,
        prompt_recovery_mode=args.prompt_recovery_mode,
        use_confidence_routing=not args.no_confidence_routing,
        consistency_routing_mode=args.consistency_routing_mode,
        prompt_channels=args.prompt_channels,
        layout_state_dim=args.layout_state_dim,
        geometry_routing_mode=args.geometry_routing_mode,
        reliability_strength=args.reliability_strength,
        decoder_channels=args.decoder_channels,
        attention_tokens=args.attention_tokens,
        local_scale_init=args.local_scale_init,
        semantic_scale_init=args.semantic_scale_init,
        fusion_branch_mode=args.fusion_branch_mode,
        use_prompt_autocorr=not args.no_prompt_autocorr,
        use_directional_edge_refine=not args.no_directional_edge_refine,
        query_points=args.query_points,
        query_scale_init=args.query_scale_init,
        geometry_scale_init=args.geometry_scale_init,
        detail_logit_scale=args.detail_logit_scale,
        rgb_boundary_scale=args.rgb_boundary_scale,
    )
    state_dict = load_checkpoint_state(args.ckpt, model.state_dict())
    missing, unexpected = model.load_state_dict(state_dict, strict=False)
    model.to(device).eval()
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
        hha_name_prefix=args.hha_name_prefix,
        label_name_prefix=args.label_name_prefix,
        ignore_index=args.ignore_index,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=device.type == "cuda",
    )

    scales = (0.75, 1.0, 1.25) if args.use_tta else (1.0,)
    miou = validate(
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
    )
    print("\n" + "=" * 50)
    print(f"Final Validation mIoU: {miou:.4f}")
    print("=" * 50)


if __name__ == "__main__":
    main()
