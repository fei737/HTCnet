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
