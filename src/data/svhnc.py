import warnings
import numpy as np
import torch
import torchvision.transforms as T
from torchvision.datasets import SVHN

warnings.filterwarnings("ignore", category=UserWarning)

from imagecorruptions import corrupt  # noqa: E402


def load_svhn_c(
    corruption: str,
    severity: int = 5,
    data_dir: str = "./data",
    indices: list[int] | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Load SVHN test set, apply corruption, return (x, y).

    x: Tensor[N, 3, 32, 32] in [0, 1].
    y: Tensor[N] with class labels (0-9).
    """
    dataset = SVHN(root=data_dir, split="test", download=True,
                   transform=T.ToTensor())

    if indices is not None:
        subset = torch.utils.data.Subset(dataset, indices)
    else:
        subset = dataset

    imgs, labels = [], []
    for img_tensor, label in subset:
        img_np = (img_tensor.permute(1, 2, 0).numpy() * 255).astype(np.uint8)
        img_corrupted = corrupt(img_np, corruption_name=corruption, severity=severity)
        imgs.append(torch.from_numpy(img_corrupted).permute(2, 0, 1).float() / 255.0)
        labels.append(label)

    x = torch.stack(imgs)
    y = torch.tensor(labels, dtype=torch.long)
    return x, y
