"""Rome32 OOD dataset loader.

Rome32 is capped to the SVHN test-set size by default so the loader does not
touch extra images that downstream experiments would never use.
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

SVHN_TEST_SIZE = 26032

_IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".bmp", ".webp", ".tif", ".tiff"}


def _collect_image_paths(folder: str) -> list[Path]:
    root = Path(folder)
    paths = sorted(
        path for path in root.rglob("*")
        if path.is_file() and path.suffix.lower() in _IMAGE_SUFFIXES
    )
    if not paths:
        raise FileNotFoundError(f"No image files found under {folder!r}")
    return paths


def load_rome32_c(
    folder: str,
    corruption: str,
    severity: int = 5,
    indices: list[int] | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Load images from folder, resize to 32x32, apply corruption.

    The loader keeps at most the same number of samples as SVHN's test split,
    so Rome32 work is bounded to the sample budget used elsewhere in the repo.
    Returns (x: Tensor[N, 3, 32, 32], y: Tensor[N]) with y set to zeros.
    """
    image_paths = _collect_image_paths(folder)
    limit = min(len(image_paths), SVHN_TEST_SIZE)
    image_paths = image_paths[:limit]

    if indices is not None:
        image_paths = [image_paths[i] for i in indices]

    resize = T.Resize((32, 32))
    to_tensor = T.ToTensor()

    imgs = []
    for path in image_paths:
        with Image.open(path) as image:
            image = image.convert("RGB")
            image = resize(image)
            img_tensor = to_tensor(image)

        img_np = (img_tensor.permute(1, 2, 0).numpy() * 255).astype(np.uint8)
        img_corrupted = corrupt(img_np, corruption_name=corruption, severity=severity)
        imgs.append(torch.from_numpy(img_corrupted).permute(2, 0, 1).float() / 255.0)

    x = torch.stack(imgs)
    y = torch.zeros(len(imgs), dtype=torch.long)
    return x, y
