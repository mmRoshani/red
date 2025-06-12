import random
import numpy as np
import torch
from .log import Log


def set_random_seeds(seed: int, log: "Log") -> None:
    """Set random seeds for reproducibility"""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    log.warn(
        f"Random seed set for random, numpy, torch (manual_seed, cuda (manual_seed, manual_seed_all), cudnn (deterministic, benchmark)): {seed}"
    )
