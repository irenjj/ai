from einops import rearrange, reduce
from math import cos
from collections.abc import Iterable
from torch import nn, Tensor
from numpy import ndarray

import torch
import math
import numpy as np

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


def learning_rate_schedule(
    t: int,
    tw: int,
    tc: int,
    max_lr: float,
    min_lr: float,
) -> float:
    lr: float
    if t < tw:
        lr = t / tw * max_lr
    elif tw <= t <= tc:
        lr = min_lr

        delta = (max_lr - min_lr) * (1 + cos((t - tw) / (tc - tw) * math.pi)) / 2

        lr += delta
    else:
        lr = min_lr

    return lr


@torch.no_grad()
def gradient_clipping(parameters: Iterable[nn.Parameter], max_l2_norm: float) -> None:
    """按全局 L2 范数原地裁剪梯度，跳过没有梯度的参数。"""
    grads = [p.grad for p in parameters if p.grad is not None]
    if not grads:
        return

    total_sq = sum(reduce(grad.square(), "... ->", "sum") for grad in grads)
    total_norm = torch.sqrt(total_sq)
    if total_norm >= max_l2_norm:
        scale = max_l2_norm / (total_norm + 1e-6)
        for grad in grads:
            grad.mul_(scale)

def get_batch(
    dataset: ndarray,
    batch_size: int,
    context_length: int,
    device: str
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    (
      (batch_size, context_length), (batch_size, context_length)
    )
    """
    n = dataset.__len__()

    inputs = []
    targets = []
    for _ in range(0, batch_size):
        i = np.random.randint(0, n - context_length)
        input = torch.tensor(
            dataset[i : i + context_length],
            dtype=torch.long,
            device=device,
        )
        target = torch.tensor(
            dataset[i + 1 : i + context_length + 1],
            dtype=torch.long,
            device=device,
        )

        inputs.append(input)
        targets.append(target)

    return (
        rearrange(inputs, "b s -> b s"),
        rearrange(targets, "b s -> b s")
    )
