from einops import einsum, rearrange, reduce

import torch

def softmax(x: torch.Tensor, dim: int) -> torch.Tensor:
    exp_x = torch.exp(x - torch.max(x, dim=dim, keepdim=True).values)
    exp_x_sum = torch.sum(exp_x, dim=dim, keepdim=True)

    return exp_x / exp_x_sum

def cross_entropy(logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
    """
    logits: (..., V)
    targets: (...)

    output: ()，所有位置的平均损失
    """

    sub_max_logits = logits - torch.max(logits, dim=-1, keepdim=True).values

    # (B, S) -> (B, S, 1)
    indices = rearrange(targets, "... -> ... 1")

    # 沿最后一维, 每行取出对应 target 的值
    # (B, S, V) -> (B, S, 1)
    selected = sub_max_logits.gather(dim=-1, index=indices)
    selected = rearrange(selected, "... 1 -> ...")

    exp_sub_max_logits = torch.exp(sub_max_logits)
    sum_logits = reduce(exp_sub_max_logits, "... v -> ...", "sum")

    losses = torch.log(sum_logits) - selected
    return losses.mean()
