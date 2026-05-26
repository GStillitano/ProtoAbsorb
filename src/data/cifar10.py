"""Clean CIFAR-10 test set loader.

Returns float tensors in [0, 1] CHW — identical preprocessing to load_cifar10c_data
(which does uint8 / 255, CHW transpose). Consistent with Hendrycks2020AugMix, which
normalises internally.

Image order matches CIFAR-10-C: both use the standard CIFAR-10 test set.
"""
import torch
from torchvision import datasets, transforms


def load_cifar10_data(
    n_examples: int = 10000,
    data_dir: str = "./data",
) -> tuple[torch.Tensor, torch.Tensor]:
    ds = datasets.CIFAR10(
        root=data_dir,
        train=False,
        download=True,
        transform=transforms.ToTensor(),
    )
    x = torch.stack([ds[i][0] for i in range(n_examples)])
    y = torch.tensor([ds[i][1] for i in range(n_examples)])
    return x, y
