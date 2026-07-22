"""Profile RSGNet parameters, operator cost, latency, and peak memory."""

import argparse
import json
import os
import time

import torch
from thop import profile

from rsgnet import RSGNet


def _checkpoint_config(path):
    if not path:
        return {}, None
    checkpoint = torch.load(path, map_location="cpu")
    config = checkpoint.get("config", {}) if isinstance(checkpoint, dict) else {}
    state = checkpoint
    if isinstance(checkpoint, dict):
        state = (
            checkpoint.get("ema_state_dict")
            or checkpoint.get("model_state_dict")
            or checkpoint.get("state_dict")
            or checkpoint
        )
    state = {
        key.removeprefix("module."): value
        for key, value in state.items()
        if key != "n_averaged"
    }
    return config, state


def _value(config, name, fallback):
    return config.get(name, fallback)


def build_model(args, config, state):
    model = RSGNet(
        n_classes=int(_value(config, "n_classes", args.n_classes)),
        pretrained_path=None,
        return_aux=False,
        encoder_name=_value(config, "encoder_name", args.encoder_name),
        prompt_channels=int(_value(config, "prompt_channels", args.prompt_channels)),
        decoder_channels=int(_value(config, "decoder_channels", args.decoder_channels)),
        attention_tokens=int(_value(config, "attention_tokens", args.attention_tokens)),
        local_scale_init=float(_value(config, "local_scale_init", 0.10)),
        semantic_scale_init=float(_value(config, "semantic_scale_init", 0.10)),
        fusion_mode=_value(config, "fusion_mode", args.fusion_mode),
        use_reliability=not bool(
            _value(config, "disable_reliability", args.disable_reliability)
        ),
        architecture_variant=_value(config, "architecture_variant", "factorized"),
        geometry_encoding=_value(config, "geometry_encoding", args.geometry_encoding),
        geometry_channels=_value(config, "geometry_channels", "dha"),
    )
    if state is not None:
        compatible = {
            key: value
            for key, value in state.items()
            if key in model.state_dict() and model.state_dict()[key].shape == value.shape
        }
        model.load_state_dict(compatible, strict=False)
    return model.eval()


def inference_dtype(name):
    return {
        "fp32": (False, torch.float32),
        "fp16": (True, torch.float16),
        "bf16": (True, torch.bfloat16),
    }[name]


def benchmark(model, inputs, device, warmup, iterations, amp_name):
    amp_enabled, amp_dtype = inference_dtype(amp_name)
    use_amp = amp_enabled and device.type == "cuda"
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)
    with torch.no_grad():
        for _ in range(max(0, warmup)):
            with torch.autocast(device_type=device.type, dtype=amp_dtype, enabled=use_amp):
                model(*inputs)
        if device.type == "cuda":
            torch.cuda.synchronize(device)
        start = time.perf_counter()
        for _ in range(max(1, iterations)):
            with torch.autocast(device_type=device.type, dtype=amp_dtype, enabled=use_amp):
                model(*inputs)
        if device.type == "cuda":
            torch.cuda.synchronize(device)
        elapsed = time.perf_counter() - start
    latency_ms = 1000.0 * elapsed / max(1, iterations)
    peak_memory_mb = (
        torch.cuda.max_memory_allocated(device) / (1024.0 ** 2)
        if device.type == "cuda"
        else None
    )
    return latency_ms, 1000.0 / latency_ms, peak_memory_mb


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--ckpt", default="")
    parser.add_argument("--output", default="")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--height", type=int, default=480)
    parser.add_argument("--width", type=int, default=480)
    parser.add_argument("--warmup", type=int, default=20)
    parser.add_argument("--iterations", type=int, default=100)
    parser.add_argument("--amp-dtype", choices=("fp32", "fp16", "bf16"), default="fp32")
    parser.add_argument("--n-classes", type=int, default=37)
    parser.add_argument("--encoder-name", default="mit_b2")
    parser.add_argument("--prompt-channels", type=int, default=32)
    parser.add_argument("--decoder-channels", type=int, default=256)
    parser.add_argument("--attention-tokens", type=int, default=512)
    parser.add_argument("--fusion-mode", default="stagewise")
    parser.add_argument("--geometry-encoding", default="factorized_channel_routed")
    parser.add_argument("--disable-reliability", action="store_true")
    args = parser.parse_args()

    config, state = _checkpoint_config(args.ckpt)
    device = torch.device(args.device)
    model = build_model(args, config, state).to(device)
    rgb = torch.randn(1, 3, args.height, args.width, device=device)
    hha = torch.randn(1, 3, args.height, args.width, device=device).clamp(-1.0, 1.0)

    macs, thop_params = profile(model, inputs=(rgb, hha), verbose=False)
    parameter_count = sum(parameter.numel() for parameter in model.parameters())
    latency_ms, fps, peak_memory_mb = benchmark(
        model, (rgb, hha), device, args.warmup, args.iterations, args.amp_dtype
    )
    result = {
        "checkpoint": os.path.abspath(args.ckpt) if args.ckpt else None,
        "device": str(device),
        "device_name": torch.cuda.get_device_name(device) if device.type == "cuda" else "CPU",
        "input_size": [1, 3, args.height, args.width],
        "precision": args.amp_dtype,
        "parameters": int(parameter_count),
        "parameters_million": parameter_count / 1e6,
        "thop_parameters": int(thop_params),
        "gmacs": macs / 1e9,
        "gflops_approx": 2.0 * macs / 1e9,
        "latency_ms": latency_ms,
        "fps_batch1": fps,
        "peak_memory_mb": peak_memory_mb,
        "warmup": args.warmup,
        "iterations": args.iterations,
    }
    print(json.dumps(result, indent=2))
    if args.output:
        output_path = os.path.abspath(args.output)
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        with open(output_path, "w", encoding="utf-8") as file:
            json.dump(result, file, indent=2)


if __name__ == "__main__":
    main()
