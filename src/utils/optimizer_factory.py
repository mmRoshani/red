import torch
import torch.optim as optim
from constants.optimizer_constants import (
    OPTIMIZER_ADAM,
    OPTIMIZER_ADAM_W,
    OPTIMIZER_SGD,
    OPTIMIZER_DSGD,
)
from validators.config_validator import ConfigValidator


def create_optimizer(model_parameters, config: ConfigValidator):
    """
    Creates an optimizer based on the configuration settings.
    
    Args:
        model_parameters: The model parameters to optimize
        config: Configuration object containing optimizer settings
        
    Returns:
        torch.optim.Optimizer: The configured optimizer
    """
    optimizer_type = config.OPTIMIZER.lower()
    learning_rate = config.LEARNING_RATE
    weight_decay = config.WEIGHT_DECAY if config.WEIGHT_DECAY is not None else 1e-4
    
    if optimizer_type == OPTIMIZER_ADAM:
        return optim.Adam(
            model_parameters,
            lr=learning_rate,
            betas=(0.9, 0.999),
            eps=1e-08,
            weight_decay=weight_decay,
        )
    elif optimizer_type == OPTIMIZER_ADAM_W:
        return optim.AdamW(
            model_parameters,
            lr=learning_rate,
            betas=(0.9, 0.999),
            eps=1e-08,
            weight_decay=weight_decay,
        )
    elif optimizer_type == OPTIMIZER_SGD:
        return optim.SGD(
            model_parameters,
            lr=learning_rate,
            momentum=0.9,
            weight_decay=weight_decay,
        )
    elif optimizer_type == OPTIMIZER_DSGD:
        return optim.SGD(
            model_parameters,
            lr=learning_rate,
            momentum=0.9,
            weight_decay=weight_decay,
        )
    else:
        raise ValueError(f"Unsupported optimizer type: {optimizer_type}")


def create_optimizer_lambda(config: ConfigValidator):
    """
    Creates a lambda function that returns an optimizer for given model parameters.
    This is useful for client initialization where we need a factory function.
    
    Args:
        config: Configuration object containing optimizer settings
        
    Returns:
        callable: A lambda function that takes model parameters and returns an optimizer
    """
    return lambda model_parameters: create_optimizer(model_parameters, config) 