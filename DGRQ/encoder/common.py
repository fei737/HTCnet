import os

import torch
import torch.nn as nn
import torch.nn.functional as F


def is_main_process():
    return int(os.environ.get("RANK", "0")) == 0


def local_contrast(x, kernel_size=3):
    padding = kernel_size // 2
    local_mean = F.avg_pool2d(x, kernel_size=kernel_size, stride=1, padding=padding)
    local_max = F.max_pool2d(x, kernel_size=kernel_size, stride=1, padding=padding)
    local_min = -F.max_pool2d(-x, kernel_size=kernel_size, stride=1, padding=padding)
    return 0.5 * ((x - local_mean).abs() + (local_max - local_min))


def make_group_count(channels, max_groups=32):
    groups = min(max_groups, channels)
    while channels % groups != 0:
        groups -= 1
    return groups


class LearnableGate(nn.Module):
    """Replaces a fixed affine remap ``lo + span * x`` with a learnable one.

    The classic pattern in this codebase is ``0.5 + 0.5 * conf`` or
    ``0.25 + 0.75 * reliability``: a hand-tuned floor and span applied to a
    gate in ``[0, 1]``. Here the floor and span are learned but kept inside
    ``[0, 1]`` through a sigmoid parameterization, so the output stays a valid
    gate while the exact mixing ratio adapts to the data.

    ``init_lo`` / ``init_span`` set the starting point so a fresh model begins
    near the original hand-tuned behaviour and drifts from there.
    """

    def __init__(self, init_lo=0.5, init_span=0.5, eps=1e-4):
        super().__init__()
        init_lo = float(min(max(init_lo, eps), 1.0 - eps))
        init_span = float(min(max(init_span, eps), 1.0 - eps))
        # Store pre-sigmoid logits so the learned values remain in (0, 1).
        self._lo = nn.Parameter(torch.logit(torch.tensor(init_lo)))
        self._span = nn.Parameter(torch.logit(torch.tensor(init_span)))

    def forward(self, x):
        lo = torch.sigmoid(self._lo)
        span = torch.sigmoid(self._span)
        return lo + span * x


class LearnableBlend(nn.Module):
    """Learns a per-pixel convex combination to replace a fixed one.

    Fixed blends such as ``0.65 * a + 0.35 * b`` bake a global prior into the
    fusion. This module predicts spatially-varying softmax weights from a
    context tensor (typically the concatenation of the signals being blended
    plus any evidence maps), so each location chooses its own mixing ratio.

    ``init_bias`` seeds the softmax logits so the starting blend matches the
    original hand-tuned ratio; e.g. ``init_bias=(0.65, 0.35)`` starts near the
    legacy weights before the router learns to deviate.
    """

    def __init__(self, n_inputs, ctx_channels, hidden_channels=None, init_bias=None):
        super().__init__()
        self.n_inputs = int(n_inputs)
        hidden_channels = hidden_channels or max(ctx_channels // 2, 8)
        self.router = nn.Sequential(
            nn.Conv2d(ctx_channels, hidden_channels, kernel_size=1, bias=False),
            nn.GELU(),
            nn.Conv2d(hidden_channels, self.n_inputs, kernel_size=1),
        )
        if init_bias is not None:
            assert len(init_bias) == self.n_inputs, "init_bias length must equal n_inputs"
            with torch.no_grad():
                probs = torch.tensor(init_bias, dtype=torch.float32).clamp_min(1e-6)
                probs = probs / probs.sum()
                nn.init.zeros_(self.router[-1].weight)
                self.router[-1].bias.copy_(torch.log(probs))

    def forward(self, inputs, context):
        """``inputs``: list of ``n_inputs`` tensors [B, C, H, W] to blend.

        ``context``: [B, ctx_channels, H, W] used to predict the weights.
        Returns the per-pixel weighted sum of ``inputs``.
        """
        weights = torch.softmax(self.router(context), dim=1)
        out = 0.0
        for i, feat in enumerate(inputs):
            out = out + weights[:, i:i + 1] * feat
        return out


def set_encoder_drop_path(encoder, drop_path_rate):
    """Enable stochastic depth on a SegFormer/MiT-style encoder."""
    if drop_path_rate <= 0:
        return 0
    dp_modules = [m for m in encoder.modules() if "DropPath" in type(m).__name__]
    n = len(dp_modules)
    if n == 0:
        return 0
    for i, module in enumerate(dp_modules):
        rate = drop_path_rate * float(i) / float(max(1, n - 1))
        if hasattr(module, "drop_prob"):
            module.drop_prob = rate
        elif hasattr(module, "p"):
            module.p = rate
    return n


class ConvBNAct(nn.Sequential):
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, padding=None,
                 groups=1, bias=False, activation=nn.ReLU):
        if padding is None:
            padding = kernel_size // 2
        layers = [
            nn.Conv2d(
                in_channels,
                out_channels,
                kernel_size=kernel_size,
                stride=stride,
                padding=padding,
                groups=groups,
                bias=bias,
            ),
            nn.BatchNorm2d(out_channels),
        ]
        if activation is not None:
            try:
                layers.append(activation(inplace=True))
            except TypeError:
                layers.append(activation())
        super().__init__(*layers)


class ConvGNAct(nn.Sequential):
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, padding=None,
                 groups=1, bias=False, activation=nn.ReLU, num_groups=32):
        if padding is None:
            padding = kernel_size // 2
        gn_groups = min(num_groups, out_channels)
        while out_channels % gn_groups != 0:
            gn_groups -= 1
        layers = [
            nn.Conv2d(
                in_channels,
                out_channels,
                kernel_size=kernel_size,
                stride=stride,
                padding=padding,
                groups=groups,
                bias=bias,
            ),
            nn.GroupNorm(gn_groups, out_channels),
        ]
        if activation is not None:
            try:
                layers.append(activation(inplace=True))
            except TypeError:
                layers.append(activation())
        super().__init__(*layers)


class FixedBoundaryExtractor(nn.Module):
    def __init__(self, out_channels):
        super().__init__()
        hidden_channels = max(out_channels // 2, 16)
        self.smooth = nn.AvgPool2d(kernel_size=3, stride=1, padding=1)
        self.proj = nn.Sequential(
            ConvBNAct(3, hidden_channels, kernel_size=3, activation=nn.GELU),
            ConvBNAct(hidden_channels, out_channels, kernel_size=1, activation=nn.GELU),
        )
        sobel_x = torch.tensor([[-1.0, 0.0, 1.0], [-2.0, 0.0, 2.0], [-1.0, 0.0, 1.0]], dtype=torch.float32)
        sobel_y = torch.tensor([[-1.0, -2.0, -1.0], [0.0, 0.0, 0.0], [1.0, 2.0, 1.0]], dtype=torch.float32)
        laplacian = torch.tensor([[0.0, -1.0, 0.0], [-1.0, 4.0, -1.0], [0.0, -1.0, 0.0]], dtype=torch.float32)
        self.register_buffer("sobel_x", sobel_x.view(1, 1, 3, 3))
        self.register_buffer("sobel_y", sobel_y.view(1, 1, 3, 3))
        self.register_buffer("laplacian", laplacian.view(1, 1, 3, 3))

    def forward(self, x):
        x_mean = self.smooth(x.mean(dim=1, keepdim=True))
        sobel_x = self.sobel_x.to(device=x.device, dtype=x.dtype)
        sobel_y = self.sobel_y.to(device=x.device, dtype=x.dtype)
        laplacian = self.laplacian.to(device=x.device, dtype=x.dtype)
        grad_x = torch.abs(F.conv2d(x_mean, sobel_x, padding=1))
        grad_y = torch.abs(F.conv2d(x_mean, sobel_y, padding=1))
        lap = torch.abs(F.conv2d(x_mean, laplacian, padding=1))
        boundary = torch.cat([grad_x, grad_y, lap], dim=1)
        normalizer = boundary.flatten(2).amax(dim=2, keepdim=True).unsqueeze(-1).clamp_min(1e-6)
        boundary = boundary / normalizer
        return self.proj(boundary)
