import torch


def additive(inputs: torch.Tensor, w: torch.Tensor) -> torch.Tensor:
    return torch.sum(inputs * w, dim=-1)


def subtractive(inputs: torch.Tensor, w: torch.Tensor) -> torch.Tensor:
    return torch.sum(inputs - w, dim=-1)


def multiplicative(inputs: torch.Tensor, w: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    log_abs = torch.log(torch.clamp(inputs.abs(), min=eps))
    return torch.sum(w * log_abs, dim=-1)


def divisive(inputs: torch.Tensor, w: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    log_abs = torch.log(torch.clamp(inputs.abs(), min=eps))
    return -torch.sum(w * log_abs, dim=-1)
