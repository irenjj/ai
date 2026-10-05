from collections.abc import Callable
from typing import Optional
import torch
import math


class AdamW(torch.optim.Optimizer):
    def __init__(
        self,
        params,
        lr=1e-3,
        betas=(0.9, 0.999),
        eps=1e-8,
        weight_decay=0.01
    ):
        if lr < 0:
            raise ValueError(f"Invalid learning rate: {lr}")
        defaults = {"lr": lr,
                    "betas": betas,
                    "eps": eps,
                    "weight_decay": weight_decay}

        super().__init__(params, defaults)

    @torch.no_grad()
    def step(self, closure: Optional[Callable] = None):
        # 按手册实现 AdamW，保留单参数组的简化接口。
        if len(self.param_groups) != 1:
            raise ValueError("This simplified optimizer supports one parameter group")
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()
        # 所有权重放在同一组，使用相同的学习率。
        group = self.param_groups[0]
        lr = group["lr"]
        (beta1, beta2) = group["betas"]
        eps = group["eps"]
        weight_decay = group["weight_decay"]
        for p in group["params"]:
            if p.grad is None:
                continue

            state = self.state[p]

            # 只在这个参数第一次更新时初始化
            if len(state) == 0:
                state["m"] = torch.zeros_like(p)
                state["v"] = torch.zeros_like(p)
                state["t"] = 0

            m = state["m"]
            v = state["v"]
            g = p.grad
            t = state["t"] + 1

            adjusted_lr = lr * math.sqrt(1 - beta2 ** t) / (1 - beta1 ** t)
            # 权重衰减使用原始学习率，与梯度更新解耦。
            p.mul_(1 - lr * weight_decay)

            mt = beta1 * m + (1 - beta1) * g
            vt = beta2 * v + (1 - beta2) * torch.square(g)
            delta = mt / (torch.sqrt(vt) + eps)

            p.add_(delta, alpha=-adjusted_lr)

            state["m"] = mt
            state["v"] = vt
            state["t"] = t
        return loss
