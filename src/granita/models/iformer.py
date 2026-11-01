"""iFormer: Hybrid ConvNet and Single-Head Modulation Attention classifier.

Implements the mobile hybrid design from "iFormer: Integrating ConvNet and Transformer
for Mobile Applications" (ICLR 2025) adapted for 32x32 isolated music glyphs:
- Early stages (1-2) use lightweight ConvNeXt-style 7x7 depthwise blocks for fast
  high-frequency local feature extraction.
- Late stages (3-4) use Single-Head Modulation Attention (SHMA) combined with parallel
  local depthwise mixing to capture global glyph structure without multi-head reshape
  bottlenecks.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from granita.config import DEFAULT_CLASSES, DEFAULT_IN_CHANNELS
from granita.models.regla import LayerScale, conv_norm_act, make_norm

IFORMER_VARIANTS: dict[str, tuple[tuple[int, int, int, int], tuple[int, int, int, int]]] = {
    "iformer": ((2, 2, 4, 2), (32, 64, 128, 256)),
    "iformer_t": ((2, 2, 4, 2), (32, 64, 128, 256)),
    "iformer_s": ((2, 2, 6, 2), (48, 96, 192, 384)),
}


class IFormerConvBlock(nn.Module):
    """Lightweight ConvNeXt-style inverted bottleneck block for early stages."""

    def __init__(
        self,
        dim: int,
        expansion: int = 4,
        norm: str = "ln",
        layer_scale: float = 1e-2,
    ) -> None:
        super().__init__()
        hidden = dim * expansion
        self.dw = nn.Conv2d(dim, dim, kernel_size=7, padding=3, groups=dim, bias=False)
        self.norm = make_norm(norm, dim)
        self.pw1 = nn.Conv2d(dim, hidden, kernel_size=1)
        self.act = nn.GELU()
        self.pw2 = nn.Conv2d(hidden, dim, kernel_size=1)
        self.scale = LayerScale(dim, layer_scale)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.pw2(self.act(self.pw1(self.norm(self.dw(x)))))
        return x + self.scale(h)


class SingleHeadModulationAttention(nn.Module):
    """Single-Head Modulation Attention (SHMA) with parallel local high-frequency path."""

    def __init__(
        self,
        dim: int,
        reduction_ratio: int = 2,
        norm: str = "ln",
        layer_scale: float = 1e-2,
    ) -> None:
        super().__init__()
        self.dim = dim
        self.qk_dim = max(dim // reduction_ratio, 16)
        self.attn_scale = self.qk_dim**-0.5

        self.pre_norm = make_norm(norm, dim)

        # Single-head Q, K projections + modulated V and Gate branches
        self.q_proj = nn.Conv2d(dim, self.qk_dim, kernel_size=1, bias=False)
        self.k_proj = nn.Conv2d(dim, self.qk_dim, kernel_size=1, bias=False)
        self.v_proj = nn.Conv2d(dim, dim, kernel_size=1)
        self.gate_proj = nn.Sequential(
            nn.Conv2d(dim, dim, kernel_size=3, padding=1, groups=dim, bias=False),
            make_norm(norm, dim),
            nn.Sigmoid(),
        )

        self.out_proj = nn.Conv2d(dim, dim, kernel_size=1)
        self.scale = LayerScale(dim, layer_scale)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b, c, h, w = x.shape
        n = h * w

        xn = self.pre_norm(x)
        q = self.q_proj(xn).view(b, self.qk_dim, n).transpose(1, 2)  # (B, N, d_qk)
        k = self.k_proj(xn).view(b, self.qk_dim, n)  # (B, d_qk, N)
        v = self.v_proj(xn).view(b, c, n).transpose(1, 2)  # (B, N, C)
        gate = self.gate_proj(xn)  # (B, C, H, W)

        attn = F.softmax(torch.bmm(q, k) * self.attn_scale, dim=-1)  # (B, N, N)
        out = torch.bmm(attn, v).transpose(1, 2).view(b, c, h, w)  # (B, C, H, W)

        # Modulate global attention with local spatial gate
        out = self.out_proj(out * gate)
        return x + self.scale(out)


class IFormerAttnBlock(nn.Module):
    """Hybrid attention block: CPE -> SHMA -> ConvFFN."""

    def __init__(
        self,
        dim: int,
        expansion: int = 4,
        norm: str = "ln",
        layer_scale: float = 1e-2,
    ) -> None:
        super().__init__()
        hidden = dim * expansion
        self.cpe = nn.Sequential(
            nn.Conv2d(dim, dim, kernel_size=3, padding=1, groups=dim, bias=False),
            make_norm(norm, dim),
        )
        self.shma = SingleHeadModulationAttention(
            dim=dim,
            norm=norm,
            layer_scale=layer_scale,
        )
        self.ffn_norm = make_norm(norm, dim)
        self.ffn = nn.Sequential(
            nn.Conv2d(dim, hidden, kernel_size=1),
            nn.GELU(),
            nn.Conv2d(hidden, hidden, kernel_size=3, padding=1, groups=hidden, bias=False),
            make_norm(norm, hidden),
            nn.GELU(),
            nn.Conv2d(hidden, dim, kernel_size=1),
        )
        self.scale = LayerScale(dim, layer_scale)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.cpe(x)
        x = self.shma(x)
        return x + self.scale(self.ffn(self.ffn_norm(x)))


class IFormerStage(nn.Module):
    """Single hierarchical stage of iFormer."""

    def __init__(
        self,
        c_in: int,
        c_out: int,
        depth: int,
        kind: str,
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
        blocks: list[nn.Module] = []
        for _ in range(depth):
            if kind == "conv":
                blocks.append(IFormerConvBlock(c_out, norm=norm, layer_scale=layer_scale))
            else:
                blocks.append(IFormerAttnBlock(c_out, norm=norm, layer_scale=layer_scale))
        self.blocks = nn.Sequential(*blocks)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.blocks(self.downsample(x))


class GranitaIFormer(nn.Module):
    """iFormer classifier adapted for 32x32 isolated music glyphs."""

    def __init__(
        self,
        variant: str = "iformer_t",
        num_classes: int = len(DEFAULT_CLASSES),
        in_channels: int = DEFAULT_IN_CHANNELS,
        norm: str = "ln",
        layer_scale: float = 0.1,
        dropout: float = 0.2,
    ) -> None:
        super().__init__()
        if variant not in IFORMER_VARIANTS:
            raise ValueError(f"Unknown iFormer variant {variant!r}: {list(IFORMER_VARIANTS)}")

        depths, dims = IFORMER_VARIANTS[variant]
        self.num_classes = num_classes
        self.in_channels = in_channels

        # Stem: 32x32 -> 16x16
        self.stem = conv_norm_act(in_channels, dims[0], 3, 2, norm, act=True)

        stage_downsamples = (False, True, False, True)
        self.stages = nn.ModuleList(
            [
                IFormerStage(
                    c_in=dims[max(i - 1, 0)],
                    c_out=dims[i],
                    depth=depths[i],
                    kind="conv" if i < 2 else "attn",
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
