"""Rome32 OOD dataset loader.

Loads 32x32 PNG patches from `export32/{class}/` for five Rome scene classes
plus an `_ood` bucket. Images are already sized; the loader only applies the
shared corruption pipeline used elsewhere in the repo.
"""
from __future__ import annotations

from pathlib import Path
import warnings

import numpy as np
import torch
import torchvision.transforms as T
from PIL import Image

warnings.filterwarnings("ignore", category=UserWarning)

from imagecorruptions import corrupt  # noqa: E402

CLASSES: tuple[str, ...] = (
    "affreschi", "fontanelle", "monete", "pasta", "statue", "_ood",
)
_IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".bmp", ".webp", ".tif", ".tiff"}


def _collect_image_paths(folder: str) -> list[Path]:
    """Collect images under `folder/{class}/*` for the fixed class allowlist.

    Stable, reproducible order: sorted by (class index, filename).
    """
    root = Path(folder)
    paths: list[Path] = []
    for cls in CLASSES:
        cls_dir = root / cls
        if not cls_dir.is_dir():
            continue
        cls_paths = sorted(
            p for p in cls_dir.iterdir()
            if p.is_file()
            and p.suffix.lower() in _IMAGE_SUFFIXES
            and p.stat().st_size > 0  # archive contains 0-byte placeholder PNGs
        )
        paths.extend(cls_paths)
    if not paths:
        raise FileNotFoundError(
            f"No Rome32 images found under {folder!r} for classes {CLASSES}"
        )
    return paths


def load_rome32_c(
    folder: str,
    corruption: str,
    severity: int = 5,
    indices: list[int] | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Load Rome32 export32 images and apply the requested CIFAR-10-C corruption.

    Returns (x: Tensor[N, 3, 32, 32], y: Tensor[N]) with y set to zeros
    (csOOD has no class labels in this protocol).
    """
    image_paths = _collect_image_paths(folder)

    if indices is not None:
        image_paths = [image_paths[i] for i in indices]

    to_tensor = T.ToTensor()

    imgs = []
    for path in image_paths:
        with Image.open(path) as image:
            image = image.convert("RGB")
            img_tensor = to_tensor(image)

        if img_tensor.shape[-1] != 32 or img_tensor.shape[-2] != 32:
            img_tensor = T.functional.resize(img_tensor, [32, 32], antialias=True)

        img_np = (img_tensor.permute(1, 2, 0).numpy() * 255).astype(np.uint8)
        img_corrupted = corrupt(img_np, corruption_name=corruption, severity=severity)
        imgs.append(torch.from_numpy(img_corrupted).permute(2, 0, 1).float() / 255.0)

    x = torch.stack(imgs)
    y = torch.zeros(len(imgs), dtype=torch.long)
    return x, y
