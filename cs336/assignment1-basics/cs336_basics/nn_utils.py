import torch

def softmax(x: torch.Tensor, dim: int) -> torch.Tensor:
    exp_x = torch.exp(x - torch.max(x, dim=dim, keepdim=True).values)
    exp_x_sum = torch.sum(exp_x, dim=dim, keepdim=True)

    return exp_x / exp_x_sum