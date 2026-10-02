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

class Embedding(nn.Module):
    def __init__(
        self,
        num_embeddings: int,
        embedding_dim: int,
        device: torch.device | None = None,
        dtype: torch.dtype | None = None,
    ):
        super().__init__()
        self.num_embeddings = num_embeddings
        self.embedding_dim = embedding_dim
        self.device = device
        self.dtype = dtype

        dt = torch.empty(num_embeddings, embedding_dim, device=device, dtype=dtype)

        mean = 0
        std = 1
        nn.init.trunc_normal_(dt, mean, std, -3, 3)

        self.weights = nn.Parameter(dt)


    def forward(self, token_ids: torch.Tensor) -> torch.Tensor:
        # token_ids: (batch_size, sequence_length)
        # output: (batch_size, sequence_length, d_model)
        return self.weights[token_ids]
