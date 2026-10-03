from einops import einsum, rearrange
from cs336_basics.nn_utils import softmax
import math

import torch

def scaled_dot_product_attention(
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    mask: torch.Tensor,
) -> torch.Tensor:
    """
    query: (..., queries, d_k)
    key: (..., keys, d_k)
    value: (..., keys, d_v)
    mask: (queries, keys)
    result: (..., queries, d_v)
    """
    dk_scale = math.sqrt(query.shape[-1])
    # (..., queries, keys)
    qk_scale = einsum(query, key, "... q d, ... k d -> ... q k") / dk_scale
    qk_scale = qk_scale.masked_fill(~mask, float("-inf"))
    # (..., queries, keys), s[..., i, j]: 第 i 个 query 分配给第 j 个 key 的注意力权重
    s = softmax(qk_scale, -1)

    return einsum(s, value, "... q k, ... k dv -> ... q dv")
