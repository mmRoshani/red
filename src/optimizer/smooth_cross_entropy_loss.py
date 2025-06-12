import torch
import torch.nn as nn
import torch.nn.functional as F


class SmoothCrossEntropyLoss(nn.Module):
    """
    Label Smoothing Cross Entropy Loss
    
    This implementation applies label smoothing to the CrossEntropy loss.
    Label smoothing is a regularization technique that prevents the model
    from becoming overconfident in its predictions.
    """
    
    def __init__(self, smoothing=0.1, reduction='mean'):
        """
        Args:
            smoothing (float): Label smoothing factor (0.0 to 1.0)
            reduction (str): Specifies the reduction to apply to the output
        """
        super(SmoothCrossEntropyLoss, self).__init__()
        self.smoothing = smoothing
        self.reduction = reduction
    
    def forward(self, input, target):
        """
        Args:
            input (Tensor): Predicted logits of shape (N, C) where N is batch size and C is number of classes
            target (Tensor): Ground truth labels of shape (N,)
        
        Returns:
            Tensor: Computed loss
        """
        if input.dim() != 2:
            raise ValueError(f"Expected input to have 2 dimensions, got {input.dim()}")
        
        if target.dim() != 1:
            raise ValueError(f"Expected target to have 1 dimension, got {target.dim()}")
        
        if input.size(0) != target.size(0):
            raise ValueError(f"Expected input and target to have same batch size, "
                           f"got {input.size(0)} and {target.size(0)}")
        
        num_classes = input.size(1)
        
        target_one_hot = torch.zeros_like(input).scatter(1, target.unsqueeze(1), 1)
        
        smoothed_target = target_one_hot * (1.0 - self.smoothing) + self.smoothing / num_classes
        
        log_probs = F.log_softmax(input, dim=1)
        
        loss = -(smoothed_target * log_probs).sum(dim=1)
        
        if self.reduction == 'mean':
            return loss.mean()
        elif self.reduction == 'sum':
            return loss.sum()
        else: 
            return loss 