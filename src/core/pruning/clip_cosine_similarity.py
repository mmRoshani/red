import torch

def clip_cosine_similarity(base_weights, model_weights):
    return torch.nan_to_num(
        torch.clip(
            torch.dot(base_weights, model_weights)
            / (torch.linalg.norm(base_weights) * torch.linalg.norm(model_weights)),
            -1,
            1,
        ),
        0,
    )

# RvQ