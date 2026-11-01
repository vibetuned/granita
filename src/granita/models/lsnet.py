"""LSNet: 'See Large, Focus Small' lightweight vision backbone for music glyphs.

Implements the core design from "LSNet: See Large, Focus Small" (CVPR 2025, THU-MIG):
- Large-Kernel Perception (LKP): 7x7 depthwise context modeling ("See Large") that
  predicts spatially-varying dynamic aggregation weights for local groups.
- Small-Kernel Aggregation (SKA): 3x3 dynamic grouped local aggregation ("Focus Small")
  that fuses fine-grained local glyph strokes using the LKP weights.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from granita.config import DEFAULT_CLASSES, DEFAULT_IN_CHANNELS
from granita.models.regla import LayerScale, conv_norm_act, make_norm

LSNET_VARIANTS: dict[str, tuple[tuple[int, int, int, int], tuple[int, int, int, int]]] = {
    "lsnet": ((2, 2, 4, 2), (32, 64, 128, 256)),
    "lsnet_t": ((2, 2, 4, 2), (32, 64, 128, 256)),
    "lsnet_s": ((2, 2, 6, 2), (48, 96, 192, 384)),
}


class LSConv(nn.Module):
    """Large-Small (LS) Convolution combining LKP ('See Large') and SKA ('Focus Small')."""

    def __init__(
        self,
        dim: int,
        large_kernel: int = 7,
        small_kernel: int = 3,
        group_size: int = 8,
        norm: str = "ln",
    ) -> None:
        super().__init__()
        self.dim = dim
        self.small_kernel = small_kernel
        self.groups = max(dim // group_size, 1)
        self.group_channels = dim // self.groups
        k2 = small_kernel * small_kernel

        # Large-Kernel Perception (LKP): 7x7 depthwise conv -> norm -> 1x1 dynamic weights
        self.lkp = nn.Sequential(
            nn.Conv2d(
                dim,
                dim,
                kernel_size=large_kernel,
                padding=large_kernel // 2,
                groups=dim,
                bias=False,
            ),
            make_norm(norm, dim),
            nn.GELU(),
            nn.Conv2d(dim, self.groups * k2, kernel_size=1),
        )

        # Small-Kernel Aggregation (SKA) static anchor + dynamic unfold mixer
        self.ska_static = nn.Conv2d(
            dim,
            dim,
            kernel_size=small_kernel,
            padding=small_kernel // 2,
            groups=dim,
            bias=False,
        )
        self.post_norm = make_norm(norm, dim)
        self.proj = nn.Conv2d(dim, dim, kernel_size=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b, c, h, w = x.shape
        k = self.small_kernel
        k2 = k * k
        g = self.groups
        gc = self.group_channels

        # 1. See Large: generate unconstrained dynamic (B, G, 1, K^2, H*W) kernel weights
        dyn_weights = self.lkp(x).view(b, g, 1, k2, h * w)

        # 2. Focus Small: unfold local 3x3 patches and aggregate with dynamic weights
        patches = F.unfold(x, kernel_size=k, padding=k // 2)  # (B, C * K^2, H*W)
        patches = patches.view(b, g, gc, k2, h * w)
        ska_dyn = (patches * dyn_weights).sum(dim=3).view(b, c, h, w)

        out = ska_dyn + self.ska_static(x)
        return self.proj(self.post_norm(out))


class LSBlock(nn.Module):
    """LSNet block: pre-norm -> LSConv -> residual + pre-norm -> FFN -> residual."""

    def __init__(
        self,
        dim: int,
        expansion: int = 4,
        norm: str = "ln",
        layer_scale: float = 1e-2,
    ) -> None:
        super().__init__()
        hidden = dim * expansion
        self.norm1 = make_norm(norm, dim)
        self.ls_conv = LSConv(dim=dim, norm=norm)
        self.scale1 = LayerScale(dim, layer_scale)

        self.norm2 = make_norm(norm, dim)
        self.ffn = nn.Sequential(
            nn.Conv2d(dim, hidden, kernel_size=1),
            nn.GELU(),
            nn.Conv2d(hidden, dim, kernel_size=1),
        )
        self.scale2 = LayerScale(dim, layer_scale)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.scale1(self.ls_conv(self.norm1(x)))
        x = x + self.scale2(self.ffn(self.norm2(x)))
        return x


class LSStage(nn.Module):
    """Single hierarchical stage of LSNet."""

    def __init__(
        self,
        c_in: int,
        c_out: int,
        depth: int,
        norm: str,
        downsample: bool,
        layer_scale: float,
    ) -> None:
        super().__init__()
        self.downsample: nn.Module = (
            conv_norm_act(c_in, c_out, 3, 2, norm, act=False)
            if downsample
            else (
                conv_norm_act(c_in, c_out, 1, 1, norm, act=False)
                if c_in != c_out
                else nn.Identity()
            )
        )
        self.blocks = nn.Sequential(
            *(LSBlock(c_out, norm=norm, layer_scale=layer_scale) for _ in range(depth))
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.blocks(self.downsample(x))


class GranitaLSNet(nn.Module):
    """LSNet classifier adapted for 32x32 isolated music glyphs."""

    def __init__(
        self,
        variant: str = "lsnet_t",
        num_classes: int = len(DEFAULT_CLASSES),
        in_channels: int = DEFAULT_IN_CHANNELS,
        norm: str = "ln",
        layer_scale: float = 0.1,
        dropout: float = 0.2,
    ) -> None:
        super().__init__()
        if variant not in LSNET_VARIANTS:
            raise ValueError(f"Unknown LSNet variant {variant!r}: {list(LSNET_VARIANTS)}")

        depths, dims = LSNET_VARIANTS[variant]
        self.num_classes = num_classes
        self.in_channels = in_channels

        # Stem: 32x32 -> 16x16
        self.stem = conv_norm_act(in_channels, dims[0], 3, 2, norm, act=True)

        stage_downsamples = (False, True, False, True)
        self.stages = nn.ModuleList(
            [
                LSStage(
                    c_in=dims[max(i - 1, 0)],
                    c_out=dims[i],
                    depth=depths[i],
                    norm=norm,
                    downsample=stage_downsamples[i],
                    layer_scale=layer_scale,
                )
                for i in range(4)
            ]
        )

        self.head_norm = make_norm(norm, dims[-1])
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.head = nn.Sequential(
            nn.Flatten(),
            nn.Dropout(p=dropout),
            nn.Linear(dims[-1], num_classes),
        )
        self.apply(self._init_weights)

    @staticmethod
    def _init_weights(m: nn.Module) -> None:
        if isinstance(m, (nn.Conv2d, nn.Linear)):
            nn.init.trunc_normal_(m.weight, std=0.02)
            if m.bias is not None:
                nn.init.zeros_(m.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = self.stem(x)
        for stage in self.stages:
            out = stage(out)
        out = self.pool(self.head_norm(out))
        return self.head(out)
