from torch.nn import  Module, CrossEntropyLoss

from constants.loss_constants import LOSS_CROSS_ENTROPY, LOSS_MASKED_CROSS_ENTROPY, LOSS_SMOOTHED_CROSS_ENTROPY

from optimizer.masked_cross_entropy_loss import MaskedCrossEntropyLoss
from optimizer.smooth_cross_entropy_loss import SmoothCrossEntropyLoss
from typing import List


def loss_factory(loss_func: str, loader_mask: List[int])-> Module:
    if loss_func == LOSS_CROSS_ENTROPY:
        return CrossEntropyLoss()
    elif loss_func == LOSS_MASKED_CROSS_ENTROPY:
        return MaskedCrossEntropyLoss(loader_mask)
    elif loss_func == LOSS_SMOOTHED_CROSS_ENTROPY:
        return SmoothCrossEntropyLoss(smoothing=0.1)
    else:
        raise ValueError (f"unsupported optimizer input, got: {loss_func}")
