import os
import random

import numpy as np
import torch


def set_seed(seed: int) -> None:
    """Seed every RNG that can affect a run, for reproducible checkpoints.

    Covers Python `random`, NumPy, and Torch (CPU + CUDA + MPS), plus cuDNN
    determinism. sklearn estimators must still be passed `random_state=seed`
    explicitly (see GmmScorer).
    """
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    if torch.backends.mps.is_available():
        torch.mps.manual_seed(seed)
