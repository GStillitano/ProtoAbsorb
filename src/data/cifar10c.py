import warnings
import torch

warnings.filterwarnings("ignore", category=UserWarning, module="robustbench")

from robustbench.data import load_cifar10c  # noqa: E402

CORRUPTIONS = [
    "gaussian_noise", "shot_noise", "impulse_noise", "defocus_blur", "glass_blur",
    "motion_blur", "zoom_blur", "snow", "frost", "fog", "brightness", "contrast",
    "elastic_transform", "pixelate", "jpeg_compression",
]


def load_cifar10c_data(
    corruption: str,
    severity: int = 5,
    n_examples: int = 10000,
    data_dir: str = "./data",
) -> tuple[torch.Tensor, torch.Tensor]:
    x, y = load_cifar10c(
        n_examples=n_examples,
        corruptions=[corruption],
        severity=severity,
        data_dir=data_dir,
    )
    return x, y
