from torch import nn

from cs336_basics.transformer.attention import MultiHeadSelfAttention
from cs336_basics.transformer.layers import RMSNorm, SwiGLU

import torch

class TransformerBlock(nn.Module):
    def __init__(
        self,
        d_model: int,
        num_heads: int,
        d_ff: int,
        theta: float | None = None,
        max_seq_len: int | None = None,
        use_rope: bool = True,
        eps: float = 1e-5,
        device: torch.device | None = None,
        dtype: torch.dtype | None = None,
        use_rmsnorm: bool = True,
    ):
        super().__init__()

        self.mha = MultiHeadSelfAttention(
            d_model,
            num_heads,
            theta,
            max_seq_len,
            use_rope,
            device, dtype)
        self.mha_rms_norm = RMSNorm(
            d_model,
            eps,
            device,
            dtype) if use_rmsnorm else nn.Identity()

        self.swiglu = SwiGLU(d_model, d_ff, device, dtype)
        self.swiglu_norm = RMSNorm(
            d_model,
            eps,
            device,
            dtype
        ) if use_rmsnorm else nn.Identity()

    def forward(
        self,
        x: torch.Tensor,
        token_positions: torch.Tensor | None = None,
    ) -> torch.Tensor:
        z = x + self.mha(self.mha_rms_norm(x), token_positions=token_positions)
        y = z + self.swiglu(self.swiglu_norm(z))

        return y
