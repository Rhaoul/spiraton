import torch


def dextro_mask(inputs: torch.Tensor) -> torch.Tensor:
    """
    True = dextrogyre
    False = levogyre
    """
    return inputs.mean(dim=-1) >= 0.0
