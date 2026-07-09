"""Residual-scale diagnostics for UGF-Lite.

Purpose
-------
The network uses many near-identity residual gates initialized to ~1e-4/1e-5
(``layer_scale``, ``local_scale``, ``semantic_scale``, ``prompt_scales``,
``out_scale``, ``query_scale``, ``geometry_scale``, ``enhance_scale``,
``edge_scale``). A branch whose scale never moves away from its tiny init is
effectively a no-op: it costs parameters and compute but contributes almost
nothing. This tool measures, per scale parameter:

  * ``|weight|``        -- learned magnitude (how far it moved from init),
  * ``|grad|``          -- gradient magnitude over a few steps (is it learning?),
  * ``init``            -- the initialization magnitude for reference,
  * ``ratio = |w|/init``-- >> 1 means the branch became active.

Use it BEFORE deleting anything: rank branches by activity, then prune only
the consistently dead ones. This keeps the "simplification" claim
data-driven rather than guesswork.

Usage
-----
Inspect a trained checkpoint (no data needed)::

    python diagnose_residual_scales.py --ckpt /path/to/best.pth

Probe gradient flow on a fresh/loaded model with a few random steps::

    python diagnose_residual_scales.py --steps 5 --layout-mode ssm

Both can be combined; ``--steps`` runs forward/backward to populate grads.
"""

import argparse
import torch

from models import UGFLiteNet

# Parameter-name suffixes that denote residual / branch gates.
SCALE_SUFFIXES = (
    "layer_scale",
    "local_scale",
    "semantic_scale",
    "out_scale",
    "query_scale",
    "geometry_scale",
    "enhance_scale",
    "edge_scale",
    "prompt_scales",
    "scale_embed",
)


def is_scale_param(name):
    return any(name.endswith(sfx) or f".{sfx}." in name or name.split(".")[-1].startswith(sfx)
               for sfx in SCALE_SUFFIXES)


def collect_scale_params(model):
    return {name: p for name, p in model.named_parameters() if is_scale_param(name)}


@torch.no_grad()
def _init_magnitude(name, param):
    # Best-effort reference init magnitude based on known constructors.
    return float(param.detach().abs().mean())


def run_grad_probe(model, steps, device, n_classes, h=128, w=160):
    """Run a few random forward/backward steps and accumulate |grad| per scale."""
    model.train()
    scale_params = collect_scale_params(model)
    grad_accum = {name: 0.0 for name in scale_params}
    opt = torch.optim.SGD(model.parameters(), lr=0.0)  # zero LR: probe grads, don't move weights
    for _ in range(steps):
        rgb = torch.randn(1, 3, h, w, device=device)
        hha = torch.randn(1, 3, h, w, device=device)
        target = torch.randint(0, n_classes, (1, h * 4, w * 4), device=device)
        opt.zero_grad(set_to_none=True)
        out = model(rgb, hha)
        seg = out[0] if isinstance(out, (tuple, list)) else out
        if seg.shape[-2:] != target.shape[-2:]:
            target = torch.randint(0, n_classes, (1, *seg.shape[-2:]), device=device)
        loss = torch.nn.functional.cross_entropy(seg, target)
        loss.backward()
        for name, p in scale_params.items():
            if p.grad is not None:
                grad_accum[name] += float(p.grad.detach().abs().mean())
    for name in grad_accum:
        grad_accum[name] /= max(steps, 1)
    return grad_accum


def main():
    parser = argparse.ArgumentParser(description="Diagnose dead residual-scale branches in UGF-Lite.")
    parser.add_argument("--ckpt", type=str, default="", help="Optional checkpoint to load before inspection.")
    parser.add_argument("--n-classes", type=int, default=41)
    parser.add_argument("--encoder-name", type=str, default="mit_b2")
    parser.add_argument("--layout-mode", type=str, default="ssm", choices=("ssm", "conv", "avgpool"))
    parser.add_argument("--safe-mode", action="store_true")
    parser.add_argument("--steps", type=int, default=0, help="If >0, run this many random fwd/bwd steps to probe grads.")
    parser.add_argument("--dead-ratio", type=float, default=3.0,
                        help="A branch is flagged 'active' when |w|/init exceeds this; below it is a prune candidate.")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = UGFLiteNet(
        n_classes=args.n_classes,
        pretrained_path=None,
        return_aux=True,
        encoder_name=args.encoder_name,
        safe_mode=args.safe_mode,
        layout_mode=args.layout_mode,
    ).to(device)

    init_mag = {}
    for name, p in collect_scale_params(model).items():
        init_mag[name] = _init_magnitude(name, p)

    if args.ckpt:
        try:
            from utils import load_checkpoint_state
            sd = load_checkpoint_state(args.ckpt, model.state_dict())
        except Exception:
            ckpt = torch.load(args.ckpt, map_location="cpu")
            sd = ckpt.get("state_dict", ckpt.get("model", ckpt))
        missing, unexpected = model.load_state_dict(sd, strict=False)
        print(f"Loaded {args.ckpt}: {len(missing)} missing, {len(unexpected)} unexpected keys.")

    grads = run_grad_probe(model, args.steps, device, args.n_classes) if args.steps > 0 else {}

    scale_params = collect_scale_params(model)
    rows = []
    for name, p in scale_params.items():
        w = float(p.detach().abs().mean())
        init = init_mag.get(name, w) or 1e-12
        ratio = w / init if init > 0 else float("inf")
        g = grads.get(name, float("nan"))
        rows.append((name, w, init, ratio, g))

    rows.sort(key=lambda r: r[3])  # least active first
    header = f"{'branch':<60} {'|w|':>10} {'init':>10} {'|w|/init':>9} {'|grad|':>10}  status"
    print("\n" + header)
    print("-" * len(header))
    for name, w, init, ratio, g in rows:
        status = "DEAD?" if ratio < args.dead_ratio else "active"
        g_str = f"{g:.3e}" if g == g else "    n/a"  # nan check
        print(f"{name:<60} {w:.3e} {init:.3e} {ratio:8.2f} {g_str:>10}  {status}")

    dead = [r[0] for r in rows if r[3] < args.dead_ratio]
    print(f"\n{len(dead)}/{len(rows)} branches below ratio {args.dead_ratio} (prune candidates).")
    if args.steps == 0:
        print("Tip: pass --steps 5 to also see gradient flow, or --ckpt to inspect a trained model.")


if __name__ == "__main__":
    main()
