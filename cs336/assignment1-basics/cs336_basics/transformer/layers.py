from torch import nn
from einops import rearrange, einsum

import torch
import math

"""
    y           =       W               x
(d_out x 1)        (d_out x d_in)     (d_in x 1)
"""
class Linear(nn.Module):
    def __init__(
        self,
        in_features: int,
        out_features: int,
        device: torch.device | None = None,
        dtype: torch.dtype | None = None,
    ):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.device = device
        self.dtype = dtype

        dt = torch.empty(out_features, in_features, device=device, dtype=dtype)

        mean = 0
        std = math.sqrt(2 / (in_features + out_features))

        nn.init.trunc_normal_(dt, mean, std, -3 * std, 3 * std)

        self.weights = nn.Parameter(dt)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        y = einsum(x, self.weights, "... d_in, d_out d_in -> ... d_out")        
        return y
