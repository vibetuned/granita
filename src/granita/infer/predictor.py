"""Inference engine and high-level Python API for Granita glyph classification."""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image

from granita.config import CLASS_GROUPS, DEFAULT_CLASSES
from granita.data.transforms import get_eval_transforms
from granita.models.backbone import create_model

if TYPE_CHECKING:
    import torch.nn as nn


@dataclass
class Prediction:
    """Inference result for a single music glyph."""

    label: str
    confidence: float
    group: str
    class_idx: int
    top_k: list[tuple[str, float]]

    def __repr__(self) -> str:
        return (
            f"Prediction(label='{self.label}', "
            f"confidence={self.confidence:.4f}, group='{self.group}')"
        )


class GranitaClassifier:
    """High-level classifier for predicting musical glyphs and symbols."""

    def __init__(
        self,
        model: nn.Module,
        classes: list[str],
        device: str | torch.device = "cpu",
    ) -> None:
        self.device = torch.device(device)
        self.model = model.to(self.device).eval()
        self.classes = list(classes)
        self.class_to_idx = {name: i for i, name in enumerate(self.classes)}
        self.transform = get_eval_transforms()

    @classmethod
    def from_checkpoint(
        cls,
        checkpoint_path: str | Path,
        device: str = "auto",
    ) -> GranitaClassifier:
        """Load a Granita classifier from a checkpoint file (.pt)."""
        path = Path(checkpoint_path)
        if not path.is_file():
            raise FileNotFoundError(f"Checkpoint not found at: {path}")

        target_device = (
            ("cuda" if torch.cuda.is_available() else "cpu") if device == "auto" else device
        )

        ckpt = torch.load(path, map_location=target_device, weights_only=False)
        classes = list(ckpt.get("classes", DEFAULT_CLASSES))
        model_name = ckpt.get("model_name", "granitanet")

        model = create_model(model_name=model_name, num_classes=len(classes))
        model.load_state_dict(ckpt["model_state_dict"])

        return cls(model=model, classes=classes, device=target_device)

    def _prepare_tensor(self, image: Image.Image | np.ndarray | torch.Tensor) -> torch.Tensor:
        """Convert input image into a normalized (1, C, H, W) tensor."""
        if isinstance(image, Image.Image):
            gray = image.convert("L")
            arr = np.array(gray, dtype=np.uint8)
            tensor = torch.from_numpy(arr).unsqueeze(0)
        elif isinstance(image, np.ndarray):
            if image.ndim == 2:
                tensor = torch.from_numpy(image.astype(np.uint8)).unsqueeze(0)
            elif image.ndim == 3 and image.shape[0] in (1, 3):
                tensor = torch.from_numpy(image.astype(np.uint8))
            else:
                tensor = torch.from_numpy(image.astype(np.uint8)).permute(2, 0, 1)
        elif isinstance(image, torch.Tensor):
            if image.ndim == 2:
                tensor = image.unsqueeze(0)
            elif image.ndim == 3:
                tensor = image
            else:
                return image.to(self.device)
        else:
            raise TypeError(f"Unsupported image type: {type(image)}")

        transformed = self.transform(tensor)
        if transformed.ndim == 3:
            transformed = transformed.unsqueeze(0)
        return transformed.to(self.device)

    @torch.no_grad()
    def predict(
        self,
        image: Image.Image | np.ndarray | torch.Tensor,
        top_k: int = 5,
    ) -> Prediction:
        """Predict the symbol class for a single glyph image."""
        tensor = self._prepare_tensor(image)
        logits = self.model(tensor)
        probs = F.softmax(logits, dim=-1).squeeze(0)

        k = min(top_k, len(self.classes))
        top_probs, top_indices = probs.topk(k)

        best_idx = int(top_indices[0].item())
        best_label = self.classes[best_idx]
        best_conf = float(top_probs[0].item())

        top_k_list = [
            (self.classes[int(idx.item())], float(p.item()))
            for idx, p in zip(top_indices, top_probs, strict=True)
        ]

        return Prediction(
            label=best_label,
            confidence=best_conf,
            group=CLASS_GROUPS.get(best_label, "other"),
            class_idx=best_idx,
            top_k=top_k_list,
        )

    @torch.no_grad()
    def predict_batch(
        self,
        images: list[Image.Image | np.ndarray | torch.Tensor],
        top_k: int = 5,
    ) -> list[Prediction]:
        """Predict classes for a batch of images."""
        if not images:
            return []

        tensors = [self._prepare_tensor(img) for img in images]
        batch = torch.cat(tensors, dim=0)

        logits = self.model(batch)
        probs = F.softmax(logits, dim=-1)

        k = min(top_k, len(self.classes))
        top_probs, top_indices = probs.topk(k, dim=-1)

        results: list[Prediction] = []
        for i in range(len(images)):
            best_idx = int(top_indices[i, 0].item())
            best_label = self.classes[best_idx]
            best_conf = float(top_probs[i, 0].item())

            top_k_list = [
                (self.classes[int(idx.item())], float(p.item()))
                for idx, p in zip(top_indices[i], top_probs[i], strict=True)
            ]

            results.append(
                Prediction(
                    label=best_label,
                    confidence=best_conf,
                    group=CLASS_GROUPS.get(best_label, "other"),
                    class_idx=best_idx,
                    top_k=top_k_list,
                )
            )
        return results


def main() -> None:
    parser = argparse.ArgumentParser(description="Classify a music symbol with Granita")
    parser.add_argument("--checkpoint", type=str, required=True, help="Path to checkpoint (.pt)")
    parser.add_argument("--image", type=str, required=True, help="Path to input glyph image")
    parser.add_argument("--device", type=str, default="auto")
    parser.add_argument("--top-k", type=int, default=5)
    args = parser.parse_args()

    classifier = GranitaClassifier.from_checkpoint(args.checkpoint, device=args.device)
    with Image.open(args.image) as img:
        pred = classifier.predict(img, top_k=args.top_k)

    conf_pct = pred.confidence * 100
    print(f"Predicted: {pred.label} (confidence: {conf_pct:.2f}%, group: {pred.group})")
    print("\nTop Candidates:")
    for label, conf in pred.top_k:
        print(f"  - {label:<24}: {conf * 100:.2f}%")


if __name__ == "__main__":
    main()
