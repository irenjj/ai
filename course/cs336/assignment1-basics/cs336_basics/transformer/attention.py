from torch import nn
from einops import einsum, rearrange
from cs336_basics.nn_utils import softmax
from cs336_basics.transformer.layers import Linear, RotaryPositionalEmbedding
import math

import torch

def scaled_dot_product_attention(
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    mask: torch.Tensor | None = None,
) -> torch.Tensor:
    """
    query: (..., queries, d_k)
    key: (..., keys, d_k)
    value: (..., keys, d_v)
    mask: 可选，可广播到 (..., queries, keys) 的布尔张量
    result: (..., queries, d_v)
    """
    dk_scale = math.sqrt(query.shape[-1])
    # (..., queries, keys)
    qk_scale = einsum(query, key, "... q d, ... k d -> ... q k") / dk_scale
    if mask is not None:
        qk_scale = qk_scale.masked_fill(~mask, float("-inf"))
    # (..., queries, keys), s[..., i, j]: 第 i 个 query 分配给第 j 个 key 的注意力权重
    s = softmax(qk_scale, -1)

    return einsum(s, value, "... q k, ... k dv -> ... q dv")

class MultiHeadSelfAttention(nn.Module):
    def __init__(
        self,
        d_model: int,
        num_heads: int,
        theta: float | None = None,
        max_seq_len: int | None = None,
        use_rope: bool = False,
        device: torch.device | None = None,
        dtype: torch.dtype | None = None,
    ):
        super().__init__()

        if d_model <= 0 or num_heads <= 0 or d_model % num_heads != 0:
            raise ValueError("d_model and num_heads must be positive, and d_model must be divisible by num_heads")
        if use_rope:
            if (d_model // num_heads) % 2 != 0:
                raise ValueError("RoPE requires an even head dimension")
            if theta is None or not math.isfinite(theta) or theta <= 0:
                raise ValueError("RoPE requires a finite, positive theta")
            if max_seq_len is None or max_seq_len <= 0:
                raise ValueError("RoPE requires a positive max_seq_len")

        self.d_model = d_model
        self.num_heads = num_heads
        self.theta = theta
        self.max_seq_len = max_seq_len

        self.wk = Linear(d_model, d_model, device, dtype)
        self.wv = Linear(d_model, d_model, device, dtype)
        self.wq = Linear(d_model, d_model, device, dtype)
        self.wo = Linear(d_model, d_model, device, dtype)

        self.use_rope = use_rope
        if self.use_rope:
            self.rope = RotaryPositionalEmbedding(
                theta=theta,
                d_k=self.d_model // self.num_heads,
                max_seq_len=max_seq_len,
                device=device
            )


    def forward(
        self,
        x: torch.Tensor,
        token_positions: torch.Tensor | None = None,
    ) -> torch.Tensor:
        # x: (..., seq_len, d_model)
        # k,v,q: (..., seq_len, d_model)
        k = self.wk(x)
        v = self.wv(x)
        q = self.wq(x)

        # (..., num_heads, seq_len, d_model / num_heads)
        # example:
        #              4 个 query 特征
        # token 0：    [1,  2,  3,  4]
        # token 1：    [5,  6,  7,  8]
        # token 2：    [9, 10, 11, 12]
        #
        #              头 0       头 1
        # token 0：    [1, 2]     [3, 4]
        # token 1：    [5, 6]     [7, 8]
        # token 2：    [9, 10]    [11, 12]
        head_k = rearrange(k, "... s (h d) -> ... h s d", h=self.num_heads)
        head_v = rearrange(v, "... s (h d) -> ... h s d", h=self.num_heads)
        head_q = rearrange(q, "... s (h d) -> ... h s d", h=self.num_heads)

        if self.use_rope:
            if token_positions is None:
                token_positions = torch.arange(
                    x.shape[-2], device=x.device, dtype=torch.long
                )
            rope_position = rearrange(token_positions, "... s -> ... 1 s")
            head_k = self.rope(head_k, rope_position)
            head_q = self.rope(head_q, rope_position)

        seq_len = x.shape[-2]
        mask = torch.tril(
            torch.ones(
                seq_len,
                seq_len,
                dtype=torch.bool,
                device=x.device
            )
        )

        # (..., num_heads, seq_len, d_model / num_heads)
        # -> (..., seq_len, d_model)
        concat = rearrange(
            scaled_dot_product_attention(head_q, head_k, head_v, mask),
            "... h s d -> ... s (h d)",
            h = self.num_heads
        )

        return self.wo(concat)
