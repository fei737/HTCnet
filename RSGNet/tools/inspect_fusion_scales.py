"""Inspect learned residual scales and their gradient activity in RSGNet."""

import argparse

import torch

from rsgnet import RSGNet


SCALE_NAMES = (
    "local_scale",
    "semantic_scale",
    "residual_scale",
    "context_scale",
    "prompt_scales",
    "scale",
)


def is_scale_parameter(name):
    leaf = name.split(".")[-1]
    return any(
        name.endswith(scale_name)
        or f".{scale_name}." in name
        or leaf.startswith(scale_name)
        for scale_name in SCALE_NAMES
    )


def collect_scale_parameters(model):
    return {
        name: parameter
        for name, parameter in model.named_parameters()
        if is_scale_parameter(name)
    }


def run_gradient_probe(model, steps, device, n_classes, height=128, width=160):
    model.train()
    scale_parameters = collect_scale_parameters(model)
    gradient_means = {name: 0.0 for name in scale_parameters}
    optimizer = torch.optim.SGD(model.parameters(), lr=0.0)
    for _ in range(steps):
        rgb = torch.randn(2, 3, height, width, device=device)
        hha = torch.randn(2, 3, height, width, device=device)
        target = torch.randint(0, n_classes, (2, height, width), device=device)
        optimizer.zero_grad(set_to_none=True)
        output = model(rgb, hha)
        segmentation = output[0] if isinstance(output, (tuple, list)) else output
        loss = torch.nn.functional.cross_entropy(segmentation, target)
        loss.backward()
        for name, parameter in scale_parameters.items():
            if parameter.grad is not None:
                gradient_means[name] += float(parameter.grad.detach().abs().mean())
    for name in gradient_means:
        gradient_means[name] /= max(steps, 1)
    return gradient_means


def load_checkpoint(path, model):
    checkpoint = torch.load(path, map_location="cpu")
    if isinstance(checkpoint, dict):
        state_dict = (
            checkpoint.get("ema_state_dict")
            or checkpoint.get("model_state_dict")
            or checkpoint.get("state_dict")
            or checkpoint
        )
    else:
        state_dict = checkpoint
    state_dict = {
        key.removeprefix("module."): value
        for key, value in state_dict.items()
        if key != "n_averaged"
    }
    missing, unexpected = model.load_state_dict(state_dict, strict=False)
    print(f"Loaded {path}: {len(missing)} missing, {len(unexpected)} unexpected keys")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--ckpt", type=str, default="")
    parser.add_argument("--n-classes", type=int, default=37)
    parser.add_argument("--encoder-name", type=str, default="mit_b2")
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
    parser.add_argument("--steps", type=int, default=0)
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = RSGNet(
        n_classes=args.n_classes,
        pretrained_path=None,
        return_aux=True,
        encoder_name=args.encoder_name,
        fusion_mode=args.fusion_mode,
        use_reliability=not args.disable_reliability,
        architecture_variant=args.architecture_variant,
        geometry_encoding=args.geometry_encoding,
        geometry_channels=args.geometry_channels,
    ).to(device)

    initial_magnitudes = {
        name: float(parameter.detach().abs().mean())
        for name, parameter in collect_scale_parameters(model).items()
    }
    if args.ckpt:
        load_checkpoint(args.ckpt, model)
    gradients = (
        run_gradient_probe(model, args.steps, device, args.n_classes)
        if args.steps > 0
        else {}
    )

    print(f"{'parameter':72s} {'mean_abs':>11s} {'init_abs':>11s} {'ratio':>9s} {'grad_abs':>11s}")
    for name, parameter in sorted(collect_scale_parameters(model).items()):
        magnitude = float(parameter.detach().abs().mean())
        initial = max(initial_magnitudes.get(name, magnitude), 1e-12)
        gradient = gradients.get(name, float("nan"))
        print(
            f"{name:72s} {magnitude:11.4e} {initial:11.4e} "
            f"{magnitude / initial:9.3f} {gradient:11.4e}"
        )


if __name__ == "__main__":
    main()
