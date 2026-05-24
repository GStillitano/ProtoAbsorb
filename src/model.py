import warnings
import torch
import torch.nn as nn
import yaml
from pathlib import Path

warnings.filterwarnings("ignore", category=UserWarning, module="robustbench")
from robustbench.utils import load_model as _rb_load_model


def load_model(config_path: str | Path = "configs/model.yaml", data_dir: str = "./data") -> nn.Module:
    """Load pretrained model. BN always uses batch statistics (train mode, no running stats)."""
    cfg = yaml.safe_load(Path(config_path).read_text())
    model = _rb_load_model(
        model_name=cfg["robustbench_name"],
        dataset=cfg["robustbench_dataset"],
        threat_model=cfg["robustbench_threat"],
        model_dir=data_dir,
    )
    model.train()
    for m in model.modules():
        if isinstance(m, nn.BatchNorm2d):
            m.track_running_stats = False
            m.running_mean = None
            m.running_var = None
    return model


def get_embeddings(
    model: nn.Module,
    x: torch.Tensor,
    device: str = "cpu",
) -> tuple[torch.Tensor, torch.Tensor]:
    """Forward pass → (features [N, d], logits [N, K]). BN uses x's own batch stats."""
    activations = {}
    linear_layers = [m for m in model.modules() if isinstance(m, nn.Linear)]
    assert linear_layers, "No Linear layer found in model"
    handle = linear_layers[-1].register_forward_hook(
        lambda _, inp, __: activations.update({"f": inp[0].detach()})
    )
    with torch.no_grad():
        logits = model(x.to(device))
    handle.remove()
    return activations["f"].cpu(), logits.cpu()


def classifier_weights(model: nn.Module) -> torch.Tensor:
    """Return final linear layer weight matrix [K, d] (frozen during TTA)."""
    linear_layers = [m for m in model.modules() if isinstance(m, nn.Linear)]
    assert linear_layers, "No Linear layer found in model"
    return linear_layers[-1].weight.detach().cpu()
