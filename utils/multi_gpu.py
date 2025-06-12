import torch
import torch.nn as nn
from src.utils.log import Log


def setup_model_for_gpu(model, config, log: Log):
    """
    Setup model for single or multi-GPU training based on config.
    
    Args:
        model: The neural network model
        config: Configuration object with GPU settings
        log: Logger instance
    
    Returns:
        model: Model configured for appropriate GPU setup
    """
    if config.MULTI_GPU and len(config.GPU_DEVICE_IDS) > 1:
        log.info(f"Setting up model for multi-GPU training on devices: {config.GPU_DEVICE_IDS}")
        
        # Move model to primary device first
        primary_device = f"cuda:{config.GPU_DEVICE_IDS[0]}"
        model = model.to(primary_device)
        
        # Wrap model with DataParallel for multi-GPU training
        model = nn.DataParallel(model, device_ids=config.GPU_DEVICE_IDS)
        
        log.info(f"Model wrapped with DataParallel across GPUs: {config.GPU_DEVICE_IDS}")
        
    else:
        device = config.DEVICE
        model = model.to(device)
        log.info(f"Model moved to single device: {device}")
    
    return model


def move_batch_to_device(batch, config):
    """
    Move batch data to appropriate device(s) based on config.
    
    Args:
        batch: Input batch (can be tuple, dict, or tensor)
        config: Configuration object with GPU settings
    
    Returns:
        batch: Batch moved to appropriate device
    """
    if config.MULTI_GPU and len(config.GPU_DEVICE_IDS) > 1:
        device = f"cuda:{config.GPU_DEVICE_IDS[0]}"
    else:
        device = config.DEVICE
    
    if isinstance(batch, dict):
        return {key: value.to(device) if hasattr(value, 'to') else value 
                for key, value in batch.items()}
    elif isinstance(batch, (list, tuple)):
        return [item.to(device) if hasattr(item, 'to') else item for item in batch]
    else:
        return batch.to(device) if hasattr(batch, 'to') else batch


def get_effective_batch_size(config):
    """
    Calculate effective batch size for multi-GPU training.
    
    Args:
        config: Configuration object
    
    Returns:
        int: Effective batch size per GPU
    """
    if config.MULTI_GPU and len(config.GPU_DEVICE_IDS) > 1:
        return config.TRAIN_BATCH_SIZE // len(config.GPU_DEVICE_IDS)
    else:
        return config.TRAIN_BATCH_SIZE 