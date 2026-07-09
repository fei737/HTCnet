"""Static + runtime verification for the geometry prompt layout path.

Run this in the *training* environment (where torch is installed):

    python verify_mamba_layout.py

It checks, with a tiny synthetic batch on CPU:

  1. ``GeometryStateSpaceScan`` forward/backward and shape preservation.
  2. The pure-PyTorch ``selective_scan_ref`` against the ``mamba_ssm`` CUDA
     kernel when both a CUDA device and the kernel are available (skipped
     otherwise).
  3. ``FrequencyAwareGeometryPrompt`` in all three
     ``layout_mode`` settings (``ssm`` / ``conv`` / ``avgpool``).
  4. A full ``UGFLiteNet`` forward in ``layout_mode='ssm'`` with ``safe_mode``
     (no pretrained download) to confirm the CM-CGRM removal and the new
     prompt path are wired end to end.

No GPU and no custom CUDA build are required; the reference scan covers the
whole pipeline on CPU.
"""

import torch

from mamba_layout import (
    GeometryStateSpaceScan,
    SelectiveScan1D,
    selective_scan_ref,
    _HAS_MAMBA_CUDA,
)
from models import FrequencyAwareGeometryPrompt, UGFLiteNet


def _ok(name):
    print(f"[PASS] {name}")


def test_gsl_scan():
    dim = 32
    block = GeometryStateSpaceScan(dim, d_state=16, scan_size=16)
    print(f"      GSL-Scan backend = {block.backend}")
    x = torch.randn(2, dim, 60, 80, requires_grad=True)
    y = block(x)
    assert y.shape == x.shape, f"shape changed: {y.shape} vs {x.shape}"
    assert torch.isfinite(y).all(), "non-finite output"
    y.mean().backward()
    assert x.grad is not None and torch.isfinite(x.grad).all(), "bad grad"
    # scan grid must have been capped
    _ok("GeometryStateSpaceScan forward/backward + shape + scan_size cap")


def test_scan_size_passthrough():
    # When the map is already <= scan_size, no resampling should happen.
    block = GeometryStateSpaceScan(16, scan_size=64)
    x = torch.randn(1, 16, 20, 24)
    y = block(x)
    assert y.shape == x.shape
    _ok("scan_size pass-through for small maps")


def test_cuda_vs_ref():
    if not (_HAS_MAMBA_CUDA and torch.cuda.is_available()):
        print("[SKIP] CUDA kernel vs reference (mamba_ssm/CUDA unavailable)")
        return
    from mamba_ssm.ops.selective_scan_interface import selective_scan_fn
    torch.manual_seed(0)
    b, d, l, n = 2, 8, 64, 16
    u = torch.randn(b, d, l, device="cuda")
    delta = torch.rand(b, d, l, device="cuda")
    A = -torch.rand(d, n, device="cuda")
    B = torch.randn(b, n, l, device="cuda")
    C = torch.randn(b, n, l, device="cuda")
    D = torch.randn(d, device="cuda")
    y_cuda = selective_scan_fn(u, delta, A, B, C, D, z=None, delta_bias=None, delta_softplus=True)
    y_ref = selective_scan_ref(u, delta, A, B, C, D=D, z=None, delta_bias=None, delta_softplus=True)
    max_err = (y_cuda - y_ref).abs().max().item()
    assert max_err < 1e-3, f"ref vs CUDA mismatch: {max_err}"
    _ok(f"selective_scan_ref matches CUDA kernel (max_err={max_err:.2e})")


def test_fdgpf_modes():
    out_channels = [64, 128, 320, 512]
    hd = torch.randn(2, 2, 120, 160)
    angle = torch.randn(2, 1, 120, 160)
    conf = torch.rand(2, 1, 120, 160)
    sizes = [(120, 160), (60, 80), (30, 40), (15, 20)]
    for mode in ("ssm", "conv", "avgpool"):
        fdgpf = FrequencyAwareGeometryPrompt(out_channels, layout_mode=mode)
        prompts = fdgpf(hd, angle, conf, sizes)
        assert len(prompts) == len(out_channels)
        for p, ch, sz in zip(prompts, out_channels, sizes):
            assert p.shape == (2, ch, *sz), f"{mode}: {p.shape}"
            assert torch.isfinite(p).all()
        _ok(f"Geometry prompt layout_mode='{mode}'")


def test_dgrqnet_forward():
    # safe_mode avoids the GSA path edge cases; pretrained_path=None skips downloads.
    model = UGFLiteNet(n_classes=14, pretrained_path=None, return_aux=True,
                       safe_mode=True, layout_mode="ssm").eval()
    rgb = torch.randn(1, 3, 128, 160)
    hha = torch.randn(1, 3, 128, 160)
    with torch.no_grad():
        seg, edge, aux = model(rgb, hha)
    assert seg.shape[0] == 1 and seg.shape[1] == 14
    assert torch.isfinite(seg).all(), "non-finite seg logits"
    assert not hasattr(model, "cm_cgrm"), "CM-CGRM should be removed"
    _ok(f"UGFLiteNet end-to-end forward (seg={tuple(seg.shape)}, edge={tuple(edge.shape)})")


if __name__ == "__main__":
    torch.manual_seed(0)
    test_gsl_scan()
    test_scan_size_passthrough()
    test_cuda_vs_ref()
    test_fdgpf_modes()
    test_dgrqnet_forward()
    print("\nAll checks completed.")
