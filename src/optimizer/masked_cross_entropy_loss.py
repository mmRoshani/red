import torch
import torch.nn as nn
import torch.nn.functional as F


class MaskedCrossEntropyLoss(nn.Module):
    def __init__(self, class_mask):
        """
        Args:
            class_mask (list or 1D tensor): Multi-hot mask of classes to include, e.g., [0,1,0,0,0,1,0,0,0,0]
        """
        super().__init__()
        self.register_buffer(
            "class_indices",
            torch.nonzero(torch.tensor(class_mask), as_tuple=False).squeeze(1),
        )

    def forward(self, logits, targets):
        logits_masked = logits[:, self.class_indices]

        target_map = {orig.item(): new for new, orig in enumerate(self.class_indices)}
        mapped_targets = torch.tensor(
            [target_map.get(t.item(), -100) for t in targets], device=targets.device
        )

        loss = F.cross_entropy(logits_masked, mapped_targets, ignore_index=-100)
        return loss
