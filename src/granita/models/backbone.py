"""Model architectures tailored for music glyph and symbol classification."""

from __future__ import annotations

from typing import Final

import torch
import torch.nn as nn

from granita.config import DEFAULT_CLASSES, DEFAULT_IN_CHANNELS
from granita.models.iformer import GranitaIFormer
from granita.models.lsnet import GranitaLSNet
from granita.models.regla import GranitaReGLA

AVAILABLE_MODELS: Final[tuple[str, ...]] = (
    "granitanet",
    "tiny",
    "regla",
    "regla_t",
    "regla_s",
    "iformer",
    "iformer_t",
    "iformer_s",
    "lsnet",
    "lsnet_t",
    "lsnet_s",
)

MODEL_DESCRIPTIONS: Final[dict[str, str]] = {
    "granitanet": "Residual ConvNet with Squeeze-and-Excitation channel attention",
    "tiny": "Ultra-compact 3-stage CNN (~78k params) for microsecond latency",
    "regla": "ReGLA (kakigori backbone): ELRF local convs + ReLU Gated Modulated Attention",
    "regla_t": "ReGLA-Tiny (depths 2-2-4-2, dims 32-64-128-256)",
    "regla_s": "ReGLA-Small (depths 2-2-6-2, dims 48-96-192-384)",
    "iformer": "iFormer (ICLR 2025): Mobile ConvNeXt + Single-Head Modulation Attention (SHMA)",
    "iformer_t": "iFormer-Tiny (depths 2-2-4-2, dims 32-64-128-256)",
    "iformer_s": "iFormer-Small (depths 2-2-6-2, dims 48-96-192-384)",
    "lsnet": "LSNet (CVPR 2025): 'See Large, Focus Small' LKP + dynamic SKA convolutions",
    "lsnet_t": "LSNet-Tiny (depths 2-2-4-2, dims 32-64-128-256)",
    "lsnet_s": "LSNet-Small (depths 2-2-6-2, dims 48-96-192-384)",
}


def count_parameters(model: nn.Module) -> int:
    """Return the number of trainable parameters in a model."""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


class SqueezeExcitation(nn.Module):
    """Squeeze-and-Excitation channel attention block."""

    def __init__(self, channels: int, reduction: int = 8) -> None:
        super().__init__()
        reduced = max(channels // reduction, 8)
        self.fc = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(channels, reduced),
            nn.ReLU(inplace=True),
            nn.Linear(reduced, channels),
            nn.Sigmoid(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b, c, _, _ = x.shape
        scale = self.fc(x).view(b, c, 1, 1)
        return x * scale


class ResidualBlock(nn.Module):
    """Residual block with batch normalization and squeeze-and-excitation."""

    def __init__(self, channels: int, use_se: bool = True) -> None:
        super().__init__()
        self.conv1 = nn.Conv2d(channels, channels, kernel_size=3, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(channels)
        self.act = nn.GELU()
        self.conv2 = nn.Conv2d(channels, channels, kernel_size=3, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(channels)
        self.se = SqueezeExcitation(channels) if use_se else nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = x
        out = self.act(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out = self.se(out)
        return self.act(out + residual)


class DownsampleBlock(nn.Module):
    """Strided convolution block for spatial downsampling."""

    def __init__(self, in_channels: int, out_channels: int) -> None:
        super().__init__()
        self.conv = nn.Conv2d(
            in_channels, out_channels, kernel_size=3, stride=2, padding=1, bias=False
        )
        self.bn = nn.BatchNorm2d(out_channels)
        self.act = nn.GELU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.act(self.bn(self.conv(x)))


class GranitaNet(nn.Module):
    """Modern lightweight convolutional network for 32x32 music glyphs.

    Combines stem convolution, multi-stage residual blocks with channel
    attention (Squeeze-and-Excitation), and global average pooling head.
    """

    def __init__(
        self,
        num_classes: int = len(DEFAULT_CLASSES),
        in_channels: int = DEFAULT_IN_CHANNELS,
        dropout: float = 0.2,
        context_dim: int = 0,
        context_hidden: int = 64,
    ) -> None:
        super().__init__()
        self.num_classes = num_classes
        self.in_channels = in_channels
        self.context_dim = context_dim

        # Stem: (B, 1, 32, 32) -> (B, 32, 32, 32)
        self.stem = nn.Sequential(
            nn.Conv2d(in_channels, 32, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(32),
            nn.GELU(),
        )

        # Stage 1: 32x32 -> 16x16 (64 channels)
        self.stage1 = nn.Sequential(
            DownsampleBlock(32, 64),
            ResidualBlock(64),
        )

        # Stage 2: 16x16 -> 8x8 (128 channels)
        self.stage2 = nn.Sequential(
            DownsampleBlock(64, 128),
            ResidualBlock(128),
        )

        # Stage 3: 8x8 -> 4x4 (256 channels)
        self.stage3 = nn.Sequential(
            DownsampleBlock(128, 256),
            ResidualBlock(256),
        )

        # Conditioning vector (where the symbol sits in the score), joined to the pooled features
        self.context: nn.Module | None = (
            nn.Sequential(
                nn.Linear(context_dim, context_hidden),
                nn.GELU(),
                nn.Linear(context_hidden, context_hidden),
                nn.GELU(),
            )
            if context_dim > 0
            else None
        )

        # Classifier head
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.head = nn.Sequential(
            nn.Dropout(p=dropout),
            nn.Linear(256 + (context_hidden if context_dim > 0 else 0), 128),
            nn.GELU(),
            nn.Dropout(p=dropout * 0.5),
            nn.Linear(128, num_classes),
        )

    def forward(self, x: torch.Tensor, context: torch.Tensor | None = None) -> torch.Tensor:
        out = self.stem(x)
        out = self.stage1(out)
        out = self.stage2(out)
        out = self.stage3(out)
        feats = self.pool(out).flatten(1)
        if self.context is not None:
            if context is None:
                context = feats.new_zeros(feats.shape[0], self.context_dim)
            feats = torch.cat([feats, self.context(context)], dim=1)
        return self.head(feats)


class GranitaTiny(nn.Module):
    """Ultra-compact ConvNet (~78k parameters) for low-latency glyph inference."""

    def __init__(
        self,
        num_classes: int = len(DEFAULT_CLASSES),
        in_channels: int = DEFAULT_IN_CHANNELS,
    ) -> None:
        super().__init__()
        self.num_classes = num_classes
        self.in_channels = in_channels

        self.features = nn.Sequential(
            nn.Conv2d(in_channels, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),  # 16x16
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),  # 8x8
            nn.Conv2d(64, 96, kernel_size=3, padding=1),
            nn.BatchNorm2d(96),
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool2d(1),
        )
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Dropout(0.2),
            nn.Linear(96, num_classes),
        )

    def forward(self, x: torch.Tensor, context: torch.Tensor | None = None) -> torch.Tensor:
        return self.classifier(self.features(x))


def create_model(
    model_name: str = "granitanet",
    num_classes: int = len(DEFAULT_CLASSES),
    in_channels: int = DEFAULT_IN_CHANNELS,
    dropout: float = 0.2,
    context_dim: int = 0,
) -> nn.Module:
    """Factory function for creating Granita classification models.

    ``context_dim > 0`` adds the conditioning-vector input; only ``granitanet``
    uses it, the other backbones accept and ignore the argument.
    """
    name = model_name.lower()
    if name in ("granitanet", "default"):
        return GranitaNet(
            num_classes=num_classes,
            in_channels=in_channels,
            dropout=dropout,
            context_dim=context_dim,
        )
    if name in ("granitatiny", "tiny", "lenet"):
        return GranitaTiny(num_classes=num_classes, in_channels=in_channels)
    if name in ("regla", "regla_t", "regla_s"):
        return GranitaReGLA(
            variant=name,
            num_classes=num_classes,
            in_channels=in_channels,
            dropout=dropout,
        )
    if name in ("iformer", "iformer_t", "iformer_s"):
        return GranitaIFormer(
            variant=name,
            num_classes=num_classes,
            in_channels=in_channels,
            dropout=dropout,
        )
    if name in ("lsnet", "lsnet_t", "lsnet_s"):
        return GranitaLSNet(
            variant=name,
            num_classes=num_classes,
            in_channels=in_channels,
            dropout=dropout,
        )
    raise ValueError(
        f"Unknown model architecture: '{model_name}'. Choose from {list(AVAILABLE_MODELS)}."
    )
