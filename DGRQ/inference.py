import argparse
import os

import cv2
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F

from dataset import HHA_MEAN, INPUT_HEIGHT, INPUT_WIDTH, RGB_MEAN, normalize_modalities, resize_with_pad
from models import UGFLiteNet, remap_dgrq_state_dict

activations = {}


def _detach_output(output):
    if isinstance(output, tuple):
        return tuple(_detach_output(item) for item in output)
    if isinstance(output, list):
        return [_detach_output(item) for item in output]
    return output.detach() if torch.is_tensor(output) else output


def get_activation(name):
    def hook(_module, _inputs, output):
        activations[name] = _detach_output(output)
    return hook


def register_plot_hooks(model):
    hook_targets = {
        "rgb_encoder": model.rgb_encoder,
        "mueb": model.mueb,
        "bsde": model.bsde,
        "agfd": model.agfd,
        "magp": model.magp,
        "fdgpf": model.fdgpf,
        "dglsr_low": model.dglsr_layers[0],
        "dglsr_deep": model.dglsr_layers[-1],
        "aspp": model.aspp,
    }
    if hasattr(model, "simple_decoder"):
        hook_targets["simple_decoder"] = model.simple_decoder
    if hasattr(model, "qdhs_decoder"):
        hook_targets["qdhs_decoder"] = model.qdhs_decoder
    if hasattr(model, "msde_bdn"):
        hook_targets["msde_bdn"] = model.msde_bdn
    if hasattr(model, "se_cbrl"):
        hook_targets["se_cbrl"] = model.se_cbrl
    return [module.register_forward_hook(get_activation(name)) for name, module in hook_targets.items()]


def build_palette(n_classes):
    rng = np.random.default_rng(2026)
    colors = rng.integers(0, 256, size=(n_classes, 3), dtype=np.uint8)
    colors[0] = 0
    if n_classes >= 15:
        colors[:15] = np.array([
            [0, 0, 0], [255, 0, 0], [0, 255, 0], [0, 0, 255],
            [255, 255, 0], [255, 0, 255], [0, 255, 255], [128, 0, 0],
            [0, 128, 0], [0, 0, 128], [128, 128, 0], [128, 0, 128],
            [0, 128, 128], [192, 192, 192], [255, 165, 0],
        ], dtype=np.uint8)
    return colors


def decode_segmap(label_mask, n_classes):
    palette = build_palette(n_classes)
    safe_mask = np.clip(label_mask, 0, n_classes - 1)
    return palette[safe_mask] / 255.0


def normalize_map(array):
    array = np.nan_to_num(array.astype(np.float32), nan=0.0, posinf=0.0, neginf=0.0)
    array = array - array.min()
    vmax = np.percentile(array, 99)
    if vmax > 1e-6:
        array = array / vmax
    elif array.max() > 1e-6:
        array = array / array.max()
    return np.clip(array, 0.0, 1.0)


def overlay_heatmap(base_rgb, heatmap, cmap="jet", alpha=0.58):
    color = plt.get_cmap(cmap)(np.clip(heatmap, 0.0, 1.0))[..., :3]
    base = base_rgb.astype(np.float32) / 255.0
    return np.clip((1.0 - alpha) * base + alpha * color, 0.0, 1.0)


def overlay_segmentation(base_rgb, label_mask, n_classes, alpha=0.55):
    seg = decode_segmap(label_mask, n_classes)
    base = base_rgb.astype(np.float32) / 255.0
    return np.clip((1.0 - alpha) * base + alpha * seg, 0.0, 1.0)


def load_image(path, color=True):
    image = cv2.imread(path, cv2.IMREAD_COLOR if color else cv2.IMREAD_GRAYSCALE)
    if image is None:
        raise FileNotFoundError(f"Image not found: {path}")
    return cv2.cvtColor(image, cv2.COLOR_BGR2RGB) if color else image


def strip_module_prefix(state_dict):
    stripped = {}
    for key, value in state_dict.items():
        if key == "n_averaged":
            continue
        while key.startswith("module."):
            key = key[7:]
        stripped[key] = value
    return remap_dgrq_state_dict(stripped)


def extract_model_state(checkpoint, prefer_ema=True):
    if not isinstance(checkpoint, dict):
        return checkpoint
    if prefer_ema and checkpoint.get("ema_state_dict") is not None:
        return checkpoint["ema_state_dict"]
    for key in ("model_state_dict", "state_dict", "ema_state_dict"):
        if checkpoint.get(key) is not None:
            return checkpoint[key]
    return checkpoint


def filter_shape_mismatch(state_dict, model_state_dict):
    filtered = {}
    for key, value in state_dict.items():
        if key in model_state_dict and model_state_dict[key].shape != value.shape:
            print(
                f"[*] Skip shape-mismatch weight: {key} | "
                f"model={tuple(model_state_dict[key].shape)} ckpt={tuple(value.shape)}"
            )
            continue
        filtered[key] = value
    return filtered


def split_outputs(output):
    if isinstance(output, tuple):
        return output[0], (output[1] if len(output) > 1 else None)
    return output, None


def nested_tensor(output, *indices):
    current = output
    for idx in indices:
        if not isinstance(current, (list, tuple)) or len(current) == 0:
            return None
        if idx < 0:
            idx += len(current)
        if idx < 0 or idx >= len(current):
            return None
        current = current[idx]
    return current if torch.is_tensor(current) else None


def first_tensor(output, prefer_last=False):
    if torch.is_tensor(output):
        return output
    if isinstance(output, (list, tuple)):
        items = [first_tensor(item, prefer_last=prefer_last) for item in output]
        items = [item for item in items if item is not None]
        if not items:
            return None
        return items[-1] if prefer_last else items[0]
    return None


def pick_pyramid_feature(output, deep=False):
    if not isinstance(output, (list, tuple)):
        return output if torch.is_tensor(output) else None
    tensors = [
        item for item in output
        if torch.is_tensor(item) and item.ndim == 4 and item.shape[1] > 8
    ]
    if not tensors:
        return first_tensor(output, prefer_last=deep)
    return tensors[-1] if deep else tensors[0]


def tensor_to_map(tensor, output_size=None, reduction="max", absolute=True, apply_sigmoid=False, softmax_conf=False):
    if tensor is None:
        return None

    if tensor.ndim == 2:
        work = tensor.unsqueeze(0).unsqueeze(0)
    elif tensor.ndim == 3:
        work = tensor.unsqueeze(0)
    else:
        work = tensor
    work = work.detach().float().cpu()[0]

    if softmax_conf:
        heatmap = torch.softmax(work, dim=0).max(dim=0)[0].numpy()
    elif apply_sigmoid:
        sig = torch.sigmoid(work)
        heatmap = sig[0].numpy() if sig.shape[0] == 1 else sig.mean(dim=0).numpy()
    elif work.shape[0] == 1:
        heatmap = work[0].numpy()
    else:
        source = work.abs() if absolute else work
        if reduction == "mean":
            heatmap = source.mean(dim=0).numpy()
        elif reduction == "l2":
            heatmap = torch.sqrt((source ** 2).mean(dim=0) + 1e-6).numpy()
        else:
            heatmap = source.max(dim=0)[0].numpy()

    heatmap = normalize_map(heatmap)
    if output_size is not None:
        heatmap = cv2.resize(heatmap, output_size, interpolation=cv2.INTER_LINEAR)
    return heatmap


def logits_to_mask(logits, output_size=None):
    if logits is None:
        return None
    target_hw = output_size[::-1] if output_size is not None else None
    if target_hw is not None and logits.shape[2:] != target_hw:
        logits = F.interpolate(logits, size=target_hw, mode="bilinear", align_corners=False)
    return torch.argmax(logits, dim=1).squeeze(0).detach().cpu().numpy()


def seg_boundary_from_logits(seg_logits, output_size=None):
    if seg_logits is None:
        return None
    probs = F.softmax(seg_logits.detach(), dim=1)
    channels = probs.shape[1]
    kernel_x = torch.tensor(
        [[-1.0, 0.0, 1.0], [-2.0, 0.0, 2.0], [-1.0, 0.0, 1.0]],
        dtype=probs.dtype,
        device=probs.device,
    ).view(1, 1, 3, 3).repeat(channels, 1, 1, 1)
    kernel_y = torch.tensor(
        [[-1.0, -2.0, -1.0], [0.0, 0.0, 0.0], [1.0, 2.0, 1.0]],
        dtype=probs.dtype,
        device=probs.device,
    ).view(1, 1, 3, 3).repeat(channels, 1, 1, 1)
    grad_x = F.conv2d(probs, kernel_x, padding=1, groups=channels)
    grad_y = F.conv2d(probs, kernel_y, padding=1, groups=channels)
    boundary = (torch.abs(grad_x) + torch.abs(grad_y)).mean(dim=1, keepdim=True)
    boundary = boundary / boundary.flatten(1).amax(dim=1).view(-1, 1, 1, 1).clamp_min(1e-6)
    return tensor_to_map(boundary, output_size=output_size)


def show_overlay_panel(ax, base_rgb, heatmap, title, cmap="jet", alpha=0.58):
    ax.axis("off")
    ax.set_title(title, fontsize=12)
    if heatmap is None:
        ax.text(0.5, 0.5, "Unavailable", ha="center", va="center", fontsize=11)
        return
    ax.imshow(overlay_heatmap(base_rgb, heatmap, cmap=cmap, alpha=alpha))


def show_seg_panel(ax, base_rgb, mask, n_classes, title):
    ax.axis("off")
    ax.set_title(title, fontsize=12)
    if mask is None:
        ax.text(0.5, 0.5, "Unavailable", ha="center", va="center", fontsize=11)
        return
    ax.imshow(overlay_segmentation(base_rgb, mask, n_classes))


def compute_single_image_metrics(pred_mask, label_mask, n_classes, ignore_index=0, miou_start_class=1):
    if pred_mask is None or label_mask is None:
        return None

    valid = label_mask != ignore_index
    valid_pixels = int(valid.sum())
    total_pixels = int(label_mask.size)
    if valid_pixels == 0:
        return {
            "valid_pixels": 0,
            "total_pixels": total_pixels,
            "pixel_acc": 0.0,
            "mean_acc": 0.0,
            "miou": 0.0,
            "gt_classes": 0,
            "pred_classes_valid": 0,
        }

    pred_valid = pred_mask[valid]
    label_valid = label_mask[valid]
    pixel_acc = float((pred_valid == label_valid).sum()) / float(valid_pixels)

    ious = []
    accs = []
    gt_classes = []
    pred_classes_valid = np.unique(pred_valid)
    for cls in range(miou_start_class, n_classes):
        pred_c = pred_valid == cls
        label_c = label_valid == cls
        gt_count = int(label_c.sum())
        if gt_count > 0:
            gt_classes.append(cls)
            accs.append(float((pred_c & label_c).sum()) / float(gt_count))
        union = int((pred_c | label_c).sum())
        if union > 0:
            ious.append(float((pred_c & label_c).sum()) / float(union))

    return {
        "valid_pixels": valid_pixels,
        "total_pixels": total_pixels,
        "pixel_acc": pixel_acc,
        "mean_acc": float(np.mean(accs)) if accs else 0.0,
        "miou": float(np.mean(ious)) if ious else 0.0,
        "gt_classes": len(gt_classes),
        "pred_classes_valid": int(pred_classes_valid.size),
    }


def prediction_mode_text(args):
    use_multiscale = len(args.scales) > 1 or any(abs(scale - 1.0) > 1e-6 for scale in args.scales)
    if use_multiscale or args.use_flip:
        return "TTA averaged final output"
    return "single-scale final output"


def show_info_panel(ax, args, pred_mask, edge_prob, metrics=None):
    edge_mean = float(edge_prob.mean()) if edge_prob is not None else 0.0
    edge_max = float(edge_prob.max()) if edge_prob is not None else 0.0
    pred_classes = int(np.unique(pred_mask).size) if pred_mask is not None else 0
    lines = [
        "UGF-Lite Branch Diagnostics",
        f"checkpoint: {os.path.basename(args.ckpt)}",
        f"input: {args.input_height}x{args.input_width}",
        f"tta scales: {', '.join(str(s) for s in args.scales)}",
        f"flip tta: {'on' if args.use_flip else 'off'}",
        "",
        "heatmaps: single native forward",
        f"prediction: {prediction_mode_text(args)}",
        "",
        f"pred classes: {pred_classes}",
        f"edge mean/max: {edge_mean:.3f} / {edge_max:.3f}",
    ]
    if metrics is not None:
        valid_ratio = metrics["valid_pixels"] / float(max(metrics["total_pixels"], 1))
        lines.extend([
            "",
            "single-image metrics:",
            f"mIoU: {metrics['miou']:.4f}",
            f"pixel acc: {metrics['pixel_acc']:.4f}",
            f"mean acc: {metrics['mean_acc']:.4f}",
            f"valid pixels: {metrics['valid_pixels']}/{metrics['total_pixels']} ({valid_ratio:.3f})",
            f"gt/pred valid classes: {metrics['gt_classes']} / {metrics['pred_classes_valid']}",
        ])
    else:
        lines.extend(["", "single-image metrics: no label"])
    text = "\n".join(lines)
    ax.axis("off")
    ax.set_title("Run Summary", fontsize=12)
    ax.text(0.02, 0.98, text, ha="left", va="top", fontsize=11, family="monospace")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--rgb", type=str, required=True)
    parser.add_argument("--hha", type=str, required=True)
    parser.add_argument("--label", type=str, default="")
    parser.add_argument("--ckpt", type=str, required=True)
    parser.add_argument("--save-path", type=str, default="inference_result.png")
    parser.add_argument("--n-classes", type=int, default=41)
    parser.add_argument("--encoder-name", type=str, default="mit_b2")
    parser.add_argument("--input-height", type=int, default=INPUT_HEIGHT)
    parser.add_argument("--input-width", type=int, default=INPUT_WIDTH)
    parser.add_argument("--scales", type=float, nargs="+", default=[1.0], help="Test scales, e.g., 0.75 1.0 1.25")
    parser.add_argument("--use-flip", action="store_true", help="Enable horizontal flip TTA")
    parser.add_argument("--plot-features", action="store_true", help="Plot branch-level diagnostics")
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

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = UGFLiteNet(
        n_classes=args.n_classes,
        pretrained_path=None,
        return_aux=True,
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
    checkpoint = torch.load(args.ckpt, map_location="cpu")
    state_dict = extract_model_state(checkpoint, prefer_ema=True)
    state_dict = filter_shape_mismatch(strip_module_prefix(state_dict), model.state_dict())
    missing, unexpected = model.load_state_dict(state_dict, strict=False)
    if missing:
        print(f"[*] Missing keys while loading checkpoint: {len(missing)}")
    if unexpected:
        print(f"[*] Unexpected keys while loading checkpoint: {len(unexpected)}")
    model.to(device).eval()

    hook_handles = register_plot_hooks(model) if args.plot_features else []

    rgb = load_image(args.rgb, color=True)
    hha = load_image(args.hha, color=True)
    label = load_image(args.label, color=False) if args.label else None

    input_size = (args.input_height, args.input_width)
    rgb_pad_val = tuple(int(round(v * 255.0)) for v in RGB_MEAN)
    hha_pad_val = tuple(int(round(v * 255.0)) for v in HHA_MEAN)
    rgb_resized = resize_with_pad(rgb, input_size, cv2.INTER_LINEAR, rgb_pad_val)
    hha_resized = resize_with_pad(hha, input_size, cv2.INTER_LINEAR, hha_pad_val)

    label_resized = None
    if label is not None:
        label_resized = resize_with_pad(label, input_size, cv2.INTER_NEAREST, 0)
        label_resized[(label_resized < 0) | (label_resized >= args.n_classes)] = 0

    rgb_t, hha_t = normalize_modalities(rgb_resized, hha_resized, device=device)

    inspect_seg_logits = None
    inspect_edge_logits = None
    with torch.no_grad():
        final_seg_logits = torch.zeros((1, args.n_classes, args.input_height, args.input_width), device=device)
        final_edge_logits = torch.zeros((1, 1, args.input_height, args.input_width), device=device)
        tta_count = 0

        for scale in args.scales:
            new_h = int(args.input_height * scale)
            new_w = int(args.input_width * scale)
            rgb_s = F.interpolate(rgb_t, size=(new_h, new_w), mode="bilinear", align_corners=False)
            hha_s = F.interpolate(hha_t, size=(new_h, new_w), mode="bilinear", align_corners=False)

            seg_out, edge_out = split_outputs(model(rgb_s, hha_s))
            final_seg_logits += F.interpolate(seg_out, size=input_size, mode="bilinear", align_corners=False)
            if edge_out is not None:
                final_edge_logits += F.interpolate(edge_out, size=input_size, mode="bilinear", align_corners=False)
            tta_count += 1

            if args.use_flip:
                seg_flip, edge_flip = split_outputs(model(torch.flip(rgb_s, dims=[3]), torch.flip(hha_s, dims=[3])))
                final_seg_logits += F.interpolate(
                    torch.flip(seg_flip, dims=[3]),
                    size=input_size,
                    mode="bilinear",
                    align_corners=False,
                )
                if edge_flip is not None:
                    final_edge_logits += F.interpolate(
                        torch.flip(edge_flip, dims=[3]),
                        size=input_size,
                        mode="bilinear",
                        align_corners=False,
                    )
                tta_count += 1

        final_seg_logits /= max(tta_count, 1)
        final_edge_logits /= max(tta_count, 1)
        pred_mask = torch.argmax(final_seg_logits, dim=1).squeeze(0).cpu().numpy()
        edge_prob = torch.sigmoid(final_edge_logits).squeeze().cpu().numpy()

        if args.plot_features:
            activations.clear()
            inspect_seg_logits, inspect_edge_logits = split_outputs(model(rgb_t, hha_t))

    display_size = (args.input_width, args.input_height)
    has_gt = label_resized is not None
    single_image_metrics = compute_single_image_metrics(
        pred_mask,
        label_resized,
        args.n_classes,
        ignore_index=0,
        miou_start_class=1,
    ) if has_gt else None
    prediction_title = "Final Prediction (TTA)" if prediction_mode_text(args).startswith("TTA") else "Final Prediction"

    if args.plot_features:
        rgb_shallow = pick_pyramid_feature(activations.get("rgb_encoder"), deep=False)
        rgb_deep = pick_pyramid_feature(activations.get("rgb_encoder"), deep=True)
        hha_confidence = first_tensor(activations.get("mueb"))
        enhanced_hd = first_tensor(activations.get("bsde"))
        prompt_low = nested_tensor(activations.get("fdgpf"), 0)
        prompt_deep = nested_tensor(activations.get("fdgpf"), -1)
        angle_grad = first_tensor(activations.get("agfd"))
        side_low = nested_tensor(activations.get("magp"), 0)
        if side_low is None:
            side_low = pick_pyramid_feature(activations.get("magp"), deep=False)
        fusion_low = first_tensor(activations.get("dglsr_low"))
        fusion_deep = first_tensor(activations.get("dglsr_deep"))
        aspp_feat = first_tensor(activations.get("aspp"))
        decoder_seg_base = nested_tensor(activations.get("qdhs_decoder"), 0)
        detail_feat = nested_tensor(activations.get("qdhs_decoder"), 1)
        simple_decoder_logits = nested_tensor(activations.get("simple_decoder"), 0)
        simple_edge_logits = nested_tensor(activations.get("simple_decoder"), 1)
        detail_edge_logits = nested_tensor(activations.get("msde_bdn"), 1)
        detail_logits = nested_tensor(activations.get("msde_bdn"), 2)
        boundary_feat = nested_tensor(activations.get("msde_bdn"), 3)
        gated_logits = first_tensor(activations.get("se_cbrl"))
        decoder_logits = decoder_seg_base if decoder_seg_base is not None else simple_decoder_logits
        decoder_title = "Simple Decoder Confidence" if args.decoder_mode == "simple_mlp" else "QDHS-Decoder Confidence"

        fig, axes = plt.subplots(5, 4, figsize=(24, 28))

        axes[0, 0].imshow(rgb_resized)
        axes[0, 0].set_title("Input RGB", fontsize=12)
        axes[0, 0].axis("off")

        axes[0, 1].imshow(hha_resized)
        axes[0, 1].set_title("Input HHA", fontsize=12)
        axes[0, 1].axis("off")

        if has_gt:
            show_seg_panel(axes[0, 2], rgb_resized, label_resized, args.n_classes, "Ground Truth Overlay")
        else:
            axes[0, 2].imshow(rgb_resized)
            axes[0, 2].set_title("Ground Truth Unavailable", fontsize=12)
            axes[0, 2].axis("off")

        show_seg_panel(axes[0, 3], rgb_resized, pred_mask, args.n_classes, prediction_title)

        show_overlay_panel(axes[1, 0], rgb_resized, normalize_map(edge_prob), "Final Edge Probability", cmap="magma", alpha=0.65)
        show_overlay_panel(axes[1, 1], rgb_resized, tensor_to_map(angle_grad, display_size), "Angular Boundary Descriptor", cmap="magma")
        show_overlay_panel(axes[1, 2], rgb_resized, tensor_to_map(side_low, display_size, reduction="mean"), "Angular Guidance Pyramid")
        show_overlay_panel(axes[1, 3], rgb_resized, tensor_to_map(rgb_shallow, display_size, reduction="mean"), "RGB Shallow Feature")

        show_overlay_panel(axes[2, 0], rgb_resized, tensor_to_map(hha_confidence, display_size, reduction="mean"), "Geometry Reliability")
        show_overlay_panel(axes[2, 1], rgb_resized, tensor_to_map(fusion_low, display_size, reduction="mean"), "Local-Semantic Fusion")
        show_overlay_panel(axes[2, 2], rgb_resized, tensor_to_map(rgb_deep, display_size, reduction="mean"), "RGB Deep Feature")
        show_overlay_panel(axes[2, 3], rgb_resized, tensor_to_map(prompt_deep if prompt_deep is not None else enhanced_hd, display_size, reduction="mean"), "Geometry Prompt")

        show_overlay_panel(
            axes[3, 0],
            rgb_resized,
            tensor_to_map(aspp_feat if aspp_feat is not None else fusion_deep, display_size, reduction="mean"),
            "ASPP: Global Context Fusion",
        )
        show_overlay_panel(
            axes[3, 1],
            rgb_resized,
            tensor_to_map(decoder_logits if decoder_logits is not None else inspect_seg_logits, display_size, softmax_conf=True),
            decoder_title,
            cmap="viridis",
        )
        show_overlay_panel(
            axes[3, 2],
            rgb_resized,
            seg_boundary_from_logits(decoder_logits if decoder_logits is not None else inspect_seg_logits, display_size),
            "SE-CBRL Boundary Prior",
            cmap="magma",
            alpha=0.65,
        )
        show_overlay_panel(
            axes[3, 3],
            rgb_resized,
            tensor_to_map(boundary_feat if boundary_feat is not None else detail_feat, display_size, reduction="mean"),
            "MSDE-BDN Boundary Feature",
            cmap="magma",
        )

        show_overlay_panel(
            axes[4, 0],
            rgb_resized,
            tensor_to_map(detail_logits, display_size, softmax_conf=True),
            "MSDE-BDN Detail Confidence",
            cmap="viridis",
        )
        show_overlay_panel(
            axes[4, 1],
            rgb_resized,
            tensor_to_map(detail_edge_logits if detail_edge_logits is not None else simple_edge_logits if simple_edge_logits is not None else inspect_edge_logits, display_size, apply_sigmoid=True),
            "MSDE-BDN Edge Logits",
            cmap="magma",
            alpha=0.65,
        )
        show_overlay_panel(
            axes[4, 2],
            rgb_resized,
            tensor_to_map(gated_logits, display_size, softmax_conf=True),
            "SE-CBRL Refined Confidence",
            cmap="viridis",
        )
        show_info_panel(axes[4, 3], args, pred_mask, edge_prob, metrics=single_image_metrics)
    else:
        n_cols = 4 if has_gt else 3
        fig, axes = plt.subplots(1, n_cols, figsize=(5 * n_cols, 5))
        axes = np.atleast_1d(axes)

        axes[0].imshow(rgb_resized)
        axes[0].set_title("Input RGB")
        axes[0].axis("off")

        axes[1].imshow(edge_prob, cmap="magma")
        axes[1].set_title("Predicted Edge")
        axes[1].axis("off")

        next_idx = 2
        if has_gt:
            axes[next_idx].imshow(decode_segmap(label_resized, args.n_classes))
            axes[next_idx].set_title("Ground Truth")
            axes[next_idx].axis("off")
            next_idx += 1

        axes[next_idx].imshow(decode_segmap(pred_mask, args.n_classes))
        axes[next_idx].set_title(prediction_title)
        axes[next_idx].axis("off")

    plt.tight_layout()
    plt.savefig(args.save_path, dpi=300, bbox_inches="tight")

    for handle in hook_handles:
        handle.remove()

    print(f"Saved inference result to: {os.path.abspath(args.save_path)}")


if __name__ == "__main__":
    main()
