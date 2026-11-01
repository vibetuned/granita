"""ReGLA classifier backbone adapted from kakigori for 32x32 music glyphs.

Based on Li et al., "ReGLA: Efficient Receptive-Field Modeling with Gated Linear
Attention Network" (arXiv 2602.05262, 2026) and kakigori's vision backbone:
- Stages 1-2 stack ELRF blocks (cascaded DW3x3 and DW5x5 inside an inverted bottleneck)
- Stages 3-4 stack ReLU Gated Modulated Attention (RGMA) blocks with CPE and separable
  local mixing.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from granita.config import DEFAULT_CLASSES, DEFAULT_IN_CHANNELS

REGLA_VARIANTS: dict[str, tuple[tuple[int, int, int, int], tuple[int, int, int, int]]] = {
    "regla": ((2, 2, 4, 2), (32, 64, 128, 256)),
    "regla_t": ((2, 2, 4, 2), (32, 64, 128, 256)),
    "regla_s": ((2, 2, 6, 2), (48, 96, 192, 384)),
}


def _exporting() -> bool:
    """Detect whether the model is currently being traced or exported (TorchScript / ONNX)."""
    return bool(
        torch.jit.is_tracing()  # type: ignore[no-untyped-call]
        or torch.compiler.is_exporting()
        or torch.onnx.is_in_onnx_export()
    )


class LayerNorm2d(nn.Module):
    """LayerNorm over the channels of an NCHW feature map."""

    def __init__(self, channels: int, eps: float = 1e-6) -> None:
        super().__init__()
        self.weight = nn.Parameter(torch.ones(1, channels, 1, 1))
        self.bias = nn.Parameter(torch.zeros(1, channels, 1, 1))
        self.eps = eps

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if _exporting():
            mean = x.mean(dim=1, keepdim=True)
            centered = x - mean
            var = centered.square().mean(dim=1, keepdim=True)
            return centered * torch.rsqrt(var + self.eps) * self.weight + self.bias
        y = F.layer_norm(
            x.permute(0, 2, 3, 1),
            (x.shape[1],),
            self.weight.view(-1),
            self.bias.view(-1),
            self.eps,
        )
        return y.permute(0, 3, 1, 2)


def make_norm(kind: str, channels: int) -> nn.Module:
    """Create channel LayerNorm ('ln') or BatchNorm ('bn')."""
    if kind == "ln":
        return LayerNorm2d(channels, eps=1e-6)
    if kind == "bn":
        return nn.BatchNorm2d(channels)
    raise ValueError(f"Unknown norm {kind!r} (expected 'ln' or 'bn')")


class LayerScale(nn.Module):
    """Per-channel learnable residual scaling."""

    def __init__(self, channels: int, init: float = 1e-2) -> None:
        super().__init__()
        self.gamma = nn.Parameter(torch.full((channels, 1, 1), init))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x * self.gamma


def conv_norm_act(
    c_in: int,
    c_out: int,
    k: int,
    stride: int,
    norm: str,
    act: bool = True,
) -> nn.Sequential:
    """Convolution followed by normalization and optional GELU activation."""
    layers: list[nn.Module] = [
        nn.Conv2d(c_in, c_out, k, stride=stride, padding=k // 2, bias=False),
        make_norm(norm, c_out),
    ]
    if act:
        layers.append(nn.GELU())
    return nn.Sequential(*layers)


class ELRF(nn.Module):
    """Efficient Large Receptive Field block: DW3x3 -> FFN with DW5x5 inside."""

    def __init__(
        self,
        dim: int,
        norm: str = "ln",
        expansion: int = 4,
        layer_scale: float = 1e-2,
    ) -> None:
        super().__init__()
        hidden = dim * expansion
        self.dw3 = nn.Conv2d(dim, dim, 3, padding=1, groups=dim, bias=False)
        self.norm1 = make_norm(norm, dim)
        self.fc1 = nn.Conv2d(dim, hidden, 1)
        self.act1 = nn.GELU()
        self.dw5 = nn.Conv2d(hidden, hidden, 5, padding=2, groups=hidden, bias=False)
        self.norm2 = make_norm(norm, hidden)
        self.act2 = nn.GELU()
        self.fc2 = nn.Conv2d(hidden, dim, 1)
        self.scale = LayerScale(dim, layer_scale)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.act1(self.fc1(self.norm1(self.dw3(x))))
        h = self.fc2(self.act2(self.norm2(self.dw5(h))))
        return x + self.scale(h)


class RGMA(nn.Module):
    """ReLU Gated Modulated Attention with pre-norm, FP16 scale safety, and output projection."""

    def __init__(
        self,
        dim: int,
        norm: str = "ln",
        heads: int = 1,
        qk_dim: int | None = None,
        layer_scale: float = 1e-2,
        eps: float = 1e-6,
    ) -> None:
        super().__init__()
        resolved_qk = qk_dim or dim
        if dim % heads != 0 or resolved_qk % heads != 0:
            raise ValueError(f"dim {dim} and qk_dim {resolved_qk} must be divisible by {heads}")
        self.heads = heads
        self.qk_dim = resolved_qk
        self.eps = eps
        self.pre_norm = make_norm(norm, dim)
        self.v = nn.Conv2d(dim, dim, 1)
        self.qk = nn.Conv2d(dim, 2 * resolved_qk, 1)
        self.post_norm = make_norm(norm, dim)
        self.proj = nn.Conv2d(dim, dim, 1)
        self.scale = LayerScale(dim, layer_scale)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b, c, h, w = x.shape
        n = h * w
        heads = self.heads
        d = self.qk_dim // heads
        c_h = c // heads

        xn = self.pre_norm(x)
        v = torch.sigmoid(self.v(xn))
        q, k = F.relu(self.qk(xn)).chunk(2, dim=1)

        inv_sqrt_n = n**-0.5
        q_r = q.reshape(b * heads, d, n) * inv_sqrt_n
        k_r = k.reshape(b * heads, d, n) * inv_sqrt_n
        v_r = v.reshape(b * heads, c_h, n)

        with torch.autocast(device_type=x.device.type, enabled=False):
            qf, kf, vf = q_r.float(), k_r.float(), v_r.float()
            kv = kf @ vf.transpose(-2, -1)  # (B*H, d, c_h)
            num = kv.transpose(-2, -1) @ qf  # (B*H, c_h, N)
            k_sum = kf.sum(dim=-1, keepdim=True)  # (B*H, d, 1)
            den = (k_sum.transpose(-2, -1) @ qf) + self.eps  # (B*H, 1, N)
            attn = (num / den).reshape(b, c, h, w).to(x.dtype)

        out = self.proj(self.post_norm(attn * v))
        return x + self.scale(out)


class AttentionBlock(nn.Module):
    """CPE (DW3x3 + norm, residual) -> RGMA -> Separable Conv3x3 (residual)."""

    def __init__(
        self,
        dim: int,
        norm: str = "ln",
        heads: int = 1,
        layer_scale: float = 1e-2,
    ) -> None:
        super().__init__()
        self.cpe = nn.Sequential(
            nn.Conv2d(dim, dim, 3, padding=1, groups=dim, bias=False),
            make_norm(norm, dim),
        )
        self.rgma = RGMA(dim, norm=norm, heads=heads, layer_scale=layer_scale)
        self.local = nn.Sequential(
            nn.Conv2d(dim, dim, 3, padding=1, groups=dim, bias=False),
            make_norm(norm, dim),
            nn.GELU(),
            nn.Conv2d(dim, dim, 1),
        )
        self.scale = LayerScale(dim, layer_scale)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.cpe(x)
        x = self.rgma(x)
        return x + self.scale(self.local(x))


class ReGLAStage(nn.Module):
    """Single hierarchical stage of ReGLA."""

    def __init__(
        self,
        c_in: int,
        c_out: int,
        depth: int,
        kind: str,
        norm: str,
        downsample: bool,
        heads: int,
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
            if kind == "elrf":
                blocks.append(ELRF(c_out, norm=norm, layer_scale=layer_scale))
            else:
                blocks.append(
                    AttentionBlock(c_out, norm=norm, heads=heads, layer_scale=layer_scale)
                )
        self.blocks = nn.Sequential(*blocks)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.blocks(self.downsample(x))


class GranitaReGLA(nn.Module):
    """ReGLA classifier adapted for 32x32 isolated music glyphs.

    Spatial progression on 32x32 input:
    - Stem: 32x32 -> 16x16 (C0)
    - Stage 0 (ELRF): 16x16 (C0)
    - Stage 1 (ELRF): 16x16 -> 8x8 (C1)
    - Stage 2 (RGMA Attention): 8x8 (C2, 64 spatial tokens)
    - Stage 3 (RGMA Attention): 8x8 -> 4x4 (C3, 16 spatial tokens)
    - Global Pool & Classifier Head -> num_classes
    """

    def __init__(
        self,
        variant: str = "regla_t",
        num_classes: int = len(DEFAULT_CLASSES),
        in_channels: int = DEFAULT_IN_CHANNELS,
        norm: str = "ln",
        heads: int = 1,
        layer_scale: float = 0.1,
        dropout: float = 0.2,
    ) -> None:
        super().__init__()
        if variant not in REGLA_VARIANTS:
            raise ValueError(f"Unknown ReGLA variant {variant!r}: {list(REGLA_VARIANTS)}")

        depths, dims = REGLA_VARIANTS[variant]
        self.num_classes = num_classes
        self.in_channels = in_channels

        # Stem: 32x32 -> 16x16
        self.stem = conv_norm_act(in_channels, dims[0], 3, 2, norm, act=True)

        # Stages: downsample at stage 1 (16->8) and stage 3 (8->4), keeping stage 2 at 8x8
        stage_downsamples = (False, True, False, True)
        self.stages = nn.ModuleList(
            [
                ReGLAStage(
                    c_in=dims[max(i - 1, 0)],
                    c_out=dims[i],
                    depth=depths[i],
                    kind="elrf" if i < 2 else "attention",
                    norm=norm,
                    downsample=stage_downsamples[i],
                    heads=heads,
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

    def forward(self, x: torch.Tensor, context: torch.Tensor | None = None) -> torch.Tensor:
        out = self.stem(x)
        for stage in self.stages:
            out = stage(out)
        out = self.pool(self.head_norm(out))
        return self.head(out)
