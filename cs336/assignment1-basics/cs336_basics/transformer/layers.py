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


class SwiGLU(nn.Module):
    def __init__(
        self,
        d_model: int,
        d_ff: int,
        device: torch.device | None = None,
        dtype: torch.dtype | None = None,
    ):
        super().__init__()

        # 1. W_1: 激活分支, d_model -> d_ff
        # 2. W_3: 另一条分支, d_model -> d_ff
        # 3. W_2: 输出变换, d_ff -> d_model
        self.w1 = Linear(d_model, d_ff, device, dtype)
        self.w3 = Linear(d_model, d_ff, device, dtype)
        self.w2 = Linear(d_ff, d_model, device, dtype)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (batch, sequence, dmodel)
        # W1x, W3x: (batch, sequence, dff)
        # y: (batch, sequence, dmodel)
        # FFN(x) = W2(SiLU(W1x) x W3x)
        w1x = self.w1(x)
        w1x_silu = torch.sigmoid(w1x) * w1x
        w3x = self.w3(x)

        return self.w2(w1x_silu * w3x)

class RotaryPositionalEmbedding(nn.Module):
    cos_cached: torch.Tensor
    sin_cached: torch.Tensor

    def __init__(
        self,
        theta: float,
        d_k: int,
        max_seq_len: int,
        device: torch.device | None = None,
    ):
        super().__init__()

        positions = torch.arange(max_seq_len, dtype=torch.float32, device=device) # (max_seq_len,)
        positions = positions.unsqueeze(-1) # (max_seq_len, 1)
        exponents = torch.arange(0, d_k, 2, device=device).float() / d_k
        denominators = theta ** exponents # (d_k // 2)
        angles = positions / denominators

        self.register_buffer(
            'cos_cached',       # 注册后的属性名
            angles.cos(),  # 要保存的张量
            persistent=False,   # 不写入 state_dict
        )

        self.register_buffer(
            'sin_cached',
            angles.sin(),
            persistent=False
        )


    def forward(
        self,
        x: torch.Tensor,
        token_positions: torch.Tensor,
    ) -> torch.Tensor:
        """
        输入 query/key 向量及其位置, 输出旋转后的向量
        """

        # x:                (..., seq_len, d_k)
        # token_positions:  (..., seq_len,)

        cos = self.cos_cached[token_positions]
        sin = self.sin_cached[token_positions]

        # 将相邻特征分成两组
        x_even = x[..., 0::2]
        x_odd = x[..., 1::2]

        # 每对特征做二维旋转
        rotated_event = x_even * cos - x_odd * sin
        rotated_odd = x_even * sin + x_odd * cos

        # [..., a', c'] 和 [..., b', d']
        # -> [..., [a', b'], [c', d']]
        paired = torch.stack(
            (rotated_event, rotated_odd), dim=-1
        )

        # 合并最后两个维度, 恢复相邻特征的排列
        # -> [..., a', b', c', d']
        output = paired.flatten(start_dim=-2)

        return output.to(x.dtype)
