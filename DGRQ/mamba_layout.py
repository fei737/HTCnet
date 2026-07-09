"""Geometry State-Space Layout Scan (GSL-Scan).

This module implements the low-frequency *layout* stream of the
Frequency-Disentangled Geometric Prompt Field (FD-GPF). It replaces the
original local average-pooling stream, whose effective receptive field
(AvgPool 9x9 + a few 3x3 convs) was far too small to support the
manuscript claim of encoding *global room structure*.

Design rationale (paired with the unchanged convolutional boundary stream):

    high-frequency boundaries  -> local isotropic convolution
    low-frequency  layout      -> global selective state-space scan

A 2D selective scan (four-directional, SS2D-style) provides input-dependent
long-range aggregation with linear complexity in the number of pixels, which
matches the "global layout" role while staying cheap on the narrow
prompt-channel width used by FD-GPF.

Two backends are provided behind a single ``GeometryStateSpaceScan`` API:

* ``selective_scan_ref`` -- a pure-PyTorch reference scan (no custom CUDA).
  Correct and portable; used for verification, ablation, and CPU/odd-GPU
  environments. Slower because the recurrence is unrolled in Python.
* ``mamba_ssm`` CUDA kernel (``selective_scan_fn``) -- used automatically
  when importable, for full training speed.

The selected backend is recorded in ``GeometryStateSpaceScan.backend`` for
logging/reproducibility.
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F

# Optional fast CUDA kernel. Falls back to the pure-PyTorch reference scan
# when unavailable so the module imports and runs in any environment.
try:  # pragma: no cover - depends on local CUDA build
    from mamba_ssm.ops.selective_scan_interface import selective_scan_fn as _selective_scan_cuda

    _HAS_MAMBA_CUDA = True
except Exception:  # noqa: BLE001 - any import/runtime failure disables the fast path
    _selective_scan_cuda = None
    _HAS_MAMBA_CUDA = False


def selective_scan_ref(u, delta, A, B, C, D=None, z=None, delta_bias=None, delta_softplus=False):
    """Pure-PyTorch reference implementation of the selective scan recurrence.

    Mirrors the semantics of ``mamba_ssm``'s ``selective_scan_fn`` for the
    case used here (real-valued A, no complex states). Computes, per channel
    ``d`` and state ``n``:

        h_t = exp(delta_t * A) * h_{t-1} + (delta_t * B_t) * u_t
        y_t = sum_n C_t * h_t  + D * u_t

    Args:
        u:     (B, D, L) input sequence.
        delta: (B, D, L) positive timestep (pre-softplus if ``delta_softplus``).
        A:     (D, N) state-transition log-rates (negative).
        B:     (B, N, L) input projection.
        C:     (B, N, L) output projection.
        D:     (D,) optional skip connection.
        z:     (B, D, L) optional gating branch (SiLU gate, Mamba-style).
        delta_bias:     (D,) optional bias added to delta before softplus.
        delta_softplus: apply softplus to delta for positivity.

    Returns:
        (B, D, L) output sequence.
    """
    dtype_in = u.dtype
    u = u.float()
    delta = delta.float()
    A = A.float()
    B = B.float()
    C = C.float()
    if D is not None:
        D = D.float()
    if z is not None:
        z = z.float()
    if delta_bias is not None:
        delta = delta + delta_bias.view(1, -1, 1).float()
    if delta_softplus:
        delta = F.softplus(delta)

    batch, dim, length = u.shape
    n_state = A.shape[1]

    # deltaA: (B, D, L, N) exp(delta * A); deltaB_u: (B, D, L, N)
    deltaA = torch.exp(torch.einsum("bdl,dn->bdln", delta, A))
    deltaB_u = torch.einsum("bdl,bnl,bdl->bdln", delta, B, u)

    h = u.new_zeros((batch, dim, n_state))
    ys = []
    for t in range(length):
        h = deltaA[:, :, t] * h + deltaB_u[:, :, t]
        ys.append(torch.einsum("bdn,bnl->bd", h, C[:, :, t:t + 1]))
    y = torch.stack(ys, dim=-1)  # (B, D, L)

    if D is not None:
        y = y + u * D.view(1, -1, 1)
    if z is not None:
        y = y * F.silu(z)
    return y.to(dtype_in)


class SelectiveScan1D(nn.Module):
    """A single-direction selective SSM over a (B, C, L) sequence.

    Projects the input into ``(delta, B, C)`` and applies the selective scan.
    Uses the CUDA kernel when available, else the reference scan.
    """

    def __init__(self, dim, d_state=16, dt_rank="auto", conv_kernel=3):
        super().__init__()
        self.dim = dim
        self.d_state = d_state
        self.dt_rank = math.ceil(dim / 16) if dt_rank == "auto" else dt_rank

        # Short depthwise causal-ish conv mixes local context before the scan.
        self.in_conv = nn.Conv1d(
            dim, dim, kernel_size=conv_kernel, padding=conv_kernel // 2, groups=dim, bias=True
        )
        # x -> (dt_rank + 2 * d_state) for input-dependent (delta, B, C).
        self.x_proj = nn.Linear(dim, self.dt_rank + 2 * d_state, bias=False)
        self.dt_proj = nn.Linear(self.dt_rank, dim, bias=True)

        # A is parameterized in log space and kept negative via -exp(A_log).
        a = torch.arange(1, d_state + 1, dtype=torch.float32).repeat(dim, 1)
        self.A_log = nn.Parameter(torch.log(a))
        self.D = nn.Parameter(torch.ones(dim))

        # Initialize dt_proj so softplus(dt) starts in a reasonable range.
        dt_init_std = self.dt_rank ** -0.5
        nn.init.uniform_(self.dt_proj.weight, -dt_init_std, dt_init_std)
        dt = torch.exp(
            torch.rand(dim) * (math.log(0.1) - math.log(0.001)) + math.log(0.001)
        ).clamp_min(1e-4)
        inv_dt = dt + torch.log(-torch.expm1(-dt))
        with torch.no_grad():
            self.dt_proj.bias.copy_(inv_dt)

    def forward(self, x):
        # x: (B, C, L)
        x = self.in_conv(x)
        x_dbl = self.x_proj(x.transpose(1, 2))  # (B, L, dt_rank + 2*d_state)
        dt, B, C = torch.split(x_dbl, [self.dt_rank, self.d_state, self.d_state], dim=-1)
        dt = self.dt_proj(dt).transpose(1, 2).contiguous()  # (B, C_dim, L)
        B = B.transpose(1, 2).contiguous()  # (B, d_state, L)
        C = C.transpose(1, 2).contiguous()  # (B, d_state, L)
        A = -torch.exp(self.A_log.float())  # (C_dim, d_state)

        if _HAS_MAMBA_CUDA and x.is_cuda:
            y = _selective_scan_cuda(
                x, dt, A, B, C, self.D.float(), z=None,
                delta_bias=None, delta_softplus=True,
            )
        else:
            y = selective_scan_ref(
                x, dt, A, B, C, D=self.D, z=None,
                delta_bias=None, delta_softplus=True,
            )
        return y  # (B, C, L)


class GeometryStateSpaceScan(nn.Module):
    """Four-directional 2D selective scan for global layout aggregation.

    Scans the feature map left-right, right-left, top-bottom, and
    bottom-top, then merges the four directional outputs. This removes the
    direction bias of a single causal scan, which matters for dense
    segmentation, while keeping linear complexity in H*W.

    Input/output are NCHW to drop-in replace a convolutional stream.
    """

    def __init__(self, dim, d_state=16, conv_kernel=3, dropout=0.0, scan_size=32):
        super().__init__()
        self.dim = dim
        # The layout stream is low-frequency by construction, so the global scan
        # is performed on a coarse ``scan_size`` x ``scan_size`` grid and then
        # upsampled back. This (a) keeps the sequence length tractable for the
        # pure-PyTorch reference recurrence (e.g. 32*32=1024 steps instead of
        # 480*640~=3e5), (b) is consistent with the "global room layout" role,
        # and (c) bounds the cost regardless of input resolution.
        self.scan_size = scan_size
        self.norm = nn.GroupNorm(num_groups=min(4, dim), num_channels=dim)
        self.scan_h = SelectiveScan1D(dim, d_state=d_state, conv_kernel=conv_kernel)
        self.scan_v = SelectiveScan1D(dim, d_state=d_state, conv_kernel=conv_kernel)
        self.merge = nn.Sequential(
            nn.Conv2d(dim, dim, kernel_size=1, bias=False),
            nn.GroupNorm(num_groups=min(4, dim), num_channels=dim),
            nn.GELU(),
        )
        self.gate = nn.Sequential(
            nn.Conv2d(dim, dim, kernel_size=1),
            nn.Sigmoid(),
        )
        self.drop = nn.Dropout(dropout) if dropout > 0 else nn.Identity()
        self.out_scale = nn.Parameter(torch.ones(1, dim, 1, 1) * 0.5)

    @property
    def backend(self):
        return "mamba_ssm_cuda" if _HAS_MAMBA_CUDA else "torch_ref"

    def _scan_horizontal(self, x):
        b, c, h, w = x.shape
        seq = x.reshape(b, c, h * w)  # row-major, scans along width within each row
        fwd = self.scan_h(seq)
        bwd = self.scan_h(torch.flip(seq, dims=[-1]))
        bwd = torch.flip(bwd, dims=[-1])
        return (fwd + bwd).reshape(b, c, h, w)

    def _scan_vertical(self, x):
        b, c, h, w = x.shape
        # transpose H/W so the sequence runs along height
        seq = x.transpose(2, 3).reshape(b, c, w * h)
        fwd = self.scan_v(seq)
        bwd = self.scan_v(torch.flip(seq, dims=[-1]))
        bwd = torch.flip(bwd, dims=[-1])
        out = (fwd + bwd).reshape(b, c, w, h).transpose(2, 3)
        return out

    def _maybe_downsample(self, x):
        """Cap the scan grid to ``scan_size`` along the longer side.

        The layout stream is low-frequency by construction, so a global scan
        on a coarse grid loses no useful information while keeping the
        sequence length (and the unrolled reference recurrence) tractable at
        full input resolution (e.g. 480x640 ~ 3e5 positions would otherwise be
        infeasible). Returns the (possibly) downsampled map and the original
        spatial size for later upsampling.
        """
        h, w = x.shape[2:]
        long_side = max(h, w)
        if self.scan_size is None or long_side <= self.scan_size:
            return x, (h, w), False
        ratio = self.scan_size / float(long_side)
        new_h = max(1, int(round(h * ratio)))
        new_w = max(1, int(round(w * ratio)))
        x_ds = F.interpolate(x, size=(new_h, new_w), mode="bilinear", align_corners=False)
        return x_ds, (h, w), True

    def forward(self, x):
        residual = x
        x = self.norm(x)
        x_scan, orig_size, downsampled = self._maybe_downsample(x)
        scanned = self._scan_horizontal(x_scan) + self._scan_vertical(x_scan)
        scanned = self.merge(scanned)
        scanned = scanned * self.gate(scanned)
        if downsampled:
            scanned = F.interpolate(scanned, size=orig_size, mode="bilinear", align_corners=False)
        scanned = self.drop(scanned)
        return residual + self.out_scale * scanned
