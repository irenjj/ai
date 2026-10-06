from torch import nn
from cs336_basics.transformer.layers import Embedding, RMSNorm, Linear
from cs336_basics.transformer.block import TransformerBlock

import torch

class TransformerLM(nn.Module):
    def __init__(
        self,
        d_model: int,
        num_heads: int,
        d_ff: int,
        vocab_size: int,
        context_length: int,
        num_layers: int,
        theta: float,
        use_rope: bool = True,
        eps: float = 1e-5,
        device: torch.device | None = None,
        dtype: torch.dtype | None = None,
        use_rmsnorm: bool = True,
    ):
        super().__init__()

        self.token_embedding = Embedding(vocab_size, d_model, device, dtype)
        self.layers = nn.ModuleList([
            TransformerBlock(
                d_model=d_model,
                num_heads=num_heads,
                d_ff=d_ff,
                theta=theta,
                max_seq_len=context_length,
                use_rope=use_rope,
                eps=eps,
                device=device,
                dtype=dtype,
                use_rmsnorm=use_rmsnorm,
            )
            for _ in range(num_layers)
        ])
        self.norm = RMSNorm(d_model, eps, device, dtype) if use_rmsnorm else nn.Identity()
        self.linear = Linear(d_model, vocab_size, device, dtype)

        # Transformer Block 层数
        self.num_layers = num_layers

    def forward(
        self,
        token_ids: torch.Tensor,
        token_positions: torch.Tensor | None = None,
    ) -> torch.Tensor:
        # (batch_size, seq_len)
        # -> (batch_size, seq_len, d_model)
        token_embedding_output = self.token_embedding(token_ids)

        # (batch_size, seq_len, d_model)
        # -> (batch_size, seq_len, d_model)
        block_output = token_embedding_output
        for layer in self.layers:
            block_output = layer(block_output, token_positions=token_positions)

        # (batch_size, seq_len, d_model)
        # -> (batch_size, seq_len, d_model)
        norm_output = self.norm(block_output)

        # (batch_size, seq_len, d_model)
        # -> (batch_size, seq_len, vocab_size)
        linear_output = self.linear(norm_output)

        return linear_output
