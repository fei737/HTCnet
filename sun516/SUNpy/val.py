import argparse
import os

import torch
from torch.utils.data import DataLoader

from inference import strip_module_prefix
from models import PFNet
from train import SUNDataset, validate


def load_checkpoint_state(ckpt_path, model_state_dict):
    checkpoint = torch.load(ckpt_path, map_location="cpu")
    if isinstance(checkpoint, dict):
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
    parser.add_argument("--ckpt", type=str, required=True)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--n-classes", type=int, default=41)
    parser.add_argument("--crop-size", type=int, default=None, help="Deprecated square input size. Prefer --input-height/--input-width.")
    parser.add_argument("--input-height", type=int, default=480)
    parser.add_argument("--input-width", type=int, default=640)
    parser.add_argument("--label-map", type=str, default="")
    parser.add_argument("--label-dir-name", type=str, default="Labels")
    parser.add_argument("--encoder-name", type=str, default="mit_b2")
    parser.add_argument("--use-tta", action="store_true")
    parser.add_argument("--debug-val-stats", action="store_true")
    parser.add_argument("--safe-mode", action="store_true")
    args = parser.parse_args()
    if args.crop_size is not None:
        args.input_height = args.crop_size
        args.input_width = args.crop_size

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[*] Target device: {device}")

    model = PFNet(
        n_classes=args.n_classes,
        pretrained_path=None,
        return_aux=False,
        encoder_name=args.encoder_name,
        safe_mode=args.safe_mode,
    )
    state_dict = load_checkpoint_state(args.ckpt, model.state_dict())
    missing, unexpected = model.load_state_dict(state_dict, strict=False)
    model.to(device).eval()
    print(f"[*] Loaded weights from: {os.path.abspath(args.ckpt)}")
    print(f"[*] Missing keys: {len(missing)} | Unexpected keys: {len(unexpected)}")

    val_dataset = SUNDataset(
        args.data_root,
        mode="test",
        crop_size=(args.input_height, args.input_width),
        n_classes=args.n_classes,
        label_map_path=args.label_map,
        label_dir_name=args.label_dir_name,
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
        debug_stats=args.debug_val_stats,
    )
    print("\n" + "=" * 50)
    print(f"Final Validation mIoU: {miou:.4f}")
    print("=" * 50)


if __name__ == "__main__":
    main()
