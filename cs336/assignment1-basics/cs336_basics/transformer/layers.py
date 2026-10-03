from torch import nn
from einops import reduce, einsum
from math import sqrt

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


class RMSNorm(nn.Module):
    def __init__(
        self,
        d_model: int,
        eps: float = 1e-5,
        device: torch.device | None = None,
        dtype: torch.dtype | None = None,
    ):
        super().__init__()

        self.d_model = d_model
        self.eps = eps
        self.device = device
        self.dtype = dtype

        dt = torch.ones(size=(d_model,), device=device, dtype=dtype)
        self.weights = nn.Parameter(dt)

    # (batch_size, sequence_length, d_model) -> (batch_size, sequence_length, d_model)
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        in_dtype = x.dtype
        x = x.to(torch.float32)

        # RMS: (batch_size, sequence_length, 1)
        # 1. 平方
        # 2. 求平均
        # 3. 加上 eps
        # 4. 开平方, 得到每个token对应的 RMS
        rms = reduce(x.square(), "... d -> ... 1", 'mean')
        rms += self.eps
        rms = rms.sqrt()

        result = x / rms * self.weights

        return result.to(in_dtype)
