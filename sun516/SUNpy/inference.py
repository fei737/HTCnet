import argparse
import os

import cv2
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F

from models import PFNet


RGB_MEAN = (0.485, 0.456, 0.406)
RGB_STD = (0.229, 0.224, 0.225)
HHA_MEAN = (0.5, 0.5, 0.5)
HHA_STD = (0.5, 0.5, 0.5)

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
        "hd_encoder": model.hd_encoder,
        "angle_grad": model.angle_grad,
        "side_guide": model.side_guide,
        "fusion_low": model.fusion_layers[0],
        "fusion_deep": model.fusion_layers[-1],
        "aspp": model.aspp,
        "decoder": model.decoder,
        "detail_edge_head": model.detail_edge_head,
        "gated_edge": model.gated_edge,
    }
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


def resize_with_pad(image, target_size, interpolation, pad_value):
    target_h, target_w = target_size
    h, w = image.shape[:2]
    scale = min(target_h / max(h, 1), target_w / max(w, 1))
    new_h = max(int(round(h * scale)), 1)
    new_w = max(int(round(w * scale)), 1)
    resized = cv2.resize(image, (new_w, new_h), interpolation=interpolation)

    pad_h = target_h - new_h
    pad_w = target_w - new_w
    top, bottom = pad_h // 2, pad_h - (pad_h // 2)
    left, right = pad_w // 2, pad_w - (pad_w // 2)

    if image.ndim == 3 and np.isscalar(pad_value):
        pad_value = (pad_value,) * image.shape[2]

    return cv2.copyMakeBorder(
        resized,
        top,
        bottom,
        left,
        right,
        cv2.BORDER_CONSTANT,
        value=pad_value,
    )


def load_image(path, color=True):
    image = cv2.imread(path, cv2.IMREAD_COLOR if color else cv2.IMREAD_GRAYSCALE)
    if image is None:
        raise FileNotFoundError(f"Image not found: {path}")
    return cv2.cvtColor(image, cv2.COLOR_BGR2RGB) if color else image


def normalize_inputs(rgb, hha, device):
    rgb_t = torch.from_numpy(rgb).permute(2, 0, 1).float() / 255.0
    hha_t = torch.from_numpy(hha).permute(2, 0, 1).float() / 255.0
    rgb_t = (rgb_t - torch.tensor(RGB_MEAN).view(3, 1, 1)) / torch.tensor(RGB_STD).view(3, 1, 1)
    hha_t = (hha_t - torch.tensor(HHA_MEAN).view(3, 1, 1)) / torch.tensor(HHA_STD).view(3, 1, 1)
    return rgb_t.unsqueeze(0).to(device), hha_t.unsqueeze(0).to(device)


def strip_module_prefix(state_dict):
    return {k[7:] if k.startswith("module.") else k: v for k, v in state_dict.items()}


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


def show_info_panel(ax, args, pred_mask, edge_prob):
    edge_mean = float(edge_prob.mean()) if edge_prob is not None else 0.0
    edge_max = float(edge_prob.max()) if edge_prob is not None else 0.0
    pred_classes = int(np.unique(pred_mask).size) if pred_mask is not None else 0
    text = "\n".join([
        "PFNet Branch Diagnostics",
        f"checkpoint: {os.path.basename(args.ckpt)}",
        f"input: {args.input_height}x{args.input_width}",
        f"tta scales: {', '.join(str(s) for s in args.scales)}",
        f"flip tta: {'on' if args.use_flip else 'off'}",
        "",
        "heatmaps: single native forward",
        "prediction: TTA-averaged final output",
        "",
        f"pred classes: {pred_classes}",
        f"edge mean/max: {edge_mean:.3f} / {edge_max:.3f}",
    ])
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
    parser.add_argument("--input-height", type=int, default=480)
    parser.add_argument("--input-width", type=int, default=640)
    parser.add_argument("--scales", type=float, nargs="+", default=[1.0], help="Test scales, e.g., 0.75 1.0 1.25")
    parser.add_argument("--use-flip", action="store_true", help="Enable horizontal flip TTA")
    parser.add_argument("--plot-features", action="store_true", help="Plot branch-level diagnostics")
    parser.add_argument("--safe-mode", action="store_true")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = PFNet(
        n_classes=args.n_classes,
        pretrained_path=None,
        return_aux=True,
        encoder_name=args.encoder_name,
        safe_mode=args.safe_mode,
    )
    checkpoint = torch.load(args.ckpt, map_location="cpu")
    state_dict = checkpoint.get("ema_state_dict") or checkpoint.get("model_state_dict") or checkpoint
    state_dict = filter_shape_mismatch(strip_module_prefix(state_dict), model.state_dict())
    model.load_state_dict(state_dict, strict=False)
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

    rgb_t, hha_t = normalize_inputs(rgb_resized, hha_resized, device)

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

    if args.plot_features:
        rgb_shallow = pick_pyramid_feature(activations.get("rgb_encoder"), deep=False)
        rgb_deep = pick_pyramid_feature(activations.get("rgb_encoder"), deep=True)
        hd_shallow = pick_pyramid_feature(activations.get("hd_encoder"), deep=False)
        hd_deep = pick_pyramid_feature(activations.get("hd_encoder"), deep=True)
        angle_grad = first_tensor(activations.get("angle_grad"))
        side_low = nested_tensor(activations.get("side_guide"), 0)
        if side_low is None:
            side_low = pick_pyramid_feature(activations.get("side_guide"), deep=False)
        fusion_low = first_tensor(activations.get("fusion_low"))
        fusion_deep = first_tensor(activations.get("fusion_deep"))
        aspp_feat = first_tensor(activations.get("aspp"))
        decoder_seg_base = nested_tensor(activations.get("decoder"), 0)
        detail_feat = nested_tensor(activations.get("decoder"), 1)
        detail_edge_logits = nested_tensor(activations.get("detail_edge_head"), 1)
        detail_logits = nested_tensor(activations.get("detail_edge_head"), 2)
        boundary_feat = nested_tensor(activations.get("detail_edge_head"), 3)
        gated_logits = first_tensor(activations.get("gated_edge"))

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

        show_seg_panel(axes[0, 3], rgb_resized, pred_mask, args.n_classes, "Final Prediction (TTA)")

        show_overlay_panel(axes[1, 0], rgb_resized, normalize_map(edge_prob), "Final Edge Probability", cmap="magma", alpha=0.65)
        show_overlay_panel(axes[1, 1], rgb_resized, tensor_to_map(angle_grad, display_size), "Angle Gradient Branch", cmap="magma")
        show_overlay_panel(axes[1, 2], rgb_resized, tensor_to_map(side_low, display_size, reduction="mean"), "Low-Level Angle Guide")
        show_overlay_panel(axes[1, 3], rgb_resized, tensor_to_map(rgb_shallow, display_size, reduction="mean"), "RGB Shallow Feature")

        show_overlay_panel(axes[2, 0], rgb_resized, tensor_to_map(hd_shallow, display_size, reduction="mean"), "HD Shallow Feature")
        show_overlay_panel(axes[2, 1], rgb_resized, tensor_to_map(fusion_low, display_size, reduction="mean"), "Low-Level Fusion")
        show_overlay_panel(axes[2, 2], rgb_resized, tensor_to_map(rgb_deep, display_size, reduction="mean"), "RGB Deep Feature")
        show_overlay_panel(axes[2, 3], rgb_resized, tensor_to_map(hd_deep, display_size, reduction="mean"), "HD Deep Feature")

        show_overlay_panel(
            axes[3, 0],
            rgb_resized,
            tensor_to_map(aspp_feat if aspp_feat is not None else fusion_deep, display_size, reduction="mean"),
            "Global Context Fusion",
        )
        show_overlay_panel(
            axes[3, 1],
            rgb_resized,
            tensor_to_map(decoder_seg_base if decoder_seg_base is not None else inspect_seg_logits, display_size, softmax_conf=True),
            "Decoder Base Confidence",
            cmap="viridis",
        )
        show_overlay_panel(
            axes[3, 2],
            rgb_resized,
            seg_boundary_from_logits(decoder_seg_base if decoder_seg_base is not None else inspect_seg_logits, display_size),
            "Semantic Boundary Prior",
            cmap="magma",
            alpha=0.65,
        )
        show_overlay_panel(
            axes[3, 3],
            rgb_resized,
            tensor_to_map(boundary_feat if boundary_feat is not None else detail_feat, display_size, reduction="mean"),
            "Boundary Feature",
            cmap="magma",
        )

        show_overlay_panel(
            axes[4, 0],
            rgb_resized,
            tensor_to_map(detail_logits, display_size, softmax_conf=True),
            "Detail Branch Confidence",
            cmap="viridis",
        )
        show_overlay_panel(
            axes[4, 1],
            rgb_resized,
            tensor_to_map(detail_edge_logits if detail_edge_logits is not None else inspect_edge_logits, display_size, apply_sigmoid=True),
            "Detail Edge Logits",
            cmap="magma",
            alpha=0.65,
        )
        show_overlay_panel(
            axes[4, 2],
            rgb_resized,
            tensor_to_map(gated_logits, display_size, softmax_conf=True),
            "Gated Refine Confidence",
            cmap="viridis",
        )
        show_info_panel(axes[4, 3], args, pred_mask, edge_prob)
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
        axes[next_idx].set_title("Prediction")
        axes[next_idx].axis("off")

    plt.tight_layout()
    plt.savefig(args.save_path, dpi=300, bbox_inches="tight")

    for handle in hook_handles:
        handle.remove()

    print(f"Saved inference result to: {os.path.abspath(args.save_path)}")


if __name__ == "__main__":
    main()
