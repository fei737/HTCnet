import os

import torch
import torch.nn as nn


def is_main_process():
    return int(os.environ.get("RANK", "0")) == 0


def make_group_count(channels, max_groups=32):
    groups = min(max_groups, channels)
    while channels % groups != 0:
        groups -= 1
    return groups


class LearnableGate(nn.Module):
    """Monotonic affine gate constrained to the interval [0, 1]."""

    def __init__(self, init_lo=0.5, init_span=0.5, eps=1e-4):
        super().__init__()
        init_lo = float(min(max(init_lo, eps), 1.0 - 2.0 * eps))
        init_hi = float(min(max(init_lo + init_span, init_lo + eps), 1.0 - eps))
        fraction = (init_hi - init_lo) / max(1.0 - init_lo, eps)
        fraction = float(min(max(fraction, eps), 1.0 - eps))
        self._lo = nn.Parameter(torch.logit(torch.tensor(init_lo)))
        self._span = nn.Parameter(torch.logit(torch.tensor(fraction)))

    def forward(self, x):
        lo = torch.sigmoid(self._lo)
        hi = lo + (1.0 - lo) * torch.sigmoid(self._span)
        return lo + (hi - lo) * x.clamp(0.0, 1.0)


def set_encoder_drop_path(encoder, drop_path_rate):
    if drop_path_rate <= 0:
        return 0
    modules = [module for module in encoder.modules() if "DropPath" in type(module).__name__]
    for index, module in enumerate(modules):
        rate = drop_path_rate * float(index) / float(max(1, len(modules) - 1))
        if hasattr(module, "drop_prob"):
            module.drop_prob = rate
        elif hasattr(module, "p"):
            module.p = rate
    return len(modules)


class ConvBNAct(nn.Sequential):
    def __init__(
        self,
        in_channels,
        out_channels,
        kernel_size=3,
        stride=1,
        padding=None,
        groups=1,
        bias=False,
        activation=nn.ReLU,
    ):
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
    def __init__(
        self,
        in_channels,
        out_channels,
        kernel_size=3,
        stride=1,
        padding=None,
        groups=1,
        bias=False,
        activation=nn.ReLU,
        num_groups=32,
    ):
        if padding is None:
            padding = kernel_size // 2
        norm_groups = make_group_count(out_channels, max_groups=num_groups)
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
            nn.GroupNorm(norm_groups, out_channels),
        ]
        if activation is not None:
            try:
                layers.append(activation(inplace=True))
            except TypeError:
                layers.append(activation())
        super().__init__(*layers)
