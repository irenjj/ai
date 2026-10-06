"""训练采样步的只读诊断；不保留计算图或完整激活张量。"""

import math
from contextlib import contextmanager

import torch


@torch.no_grad()
def tensor_stats(tensor):
    value = tensor.detach().float()
    l2 = torch.linalg.vector_norm(value).item()
    finite = math.isfinite(l2)
    return {
        "shape": list(tensor.shape),
        "numel": tensor.numel(),
        "l2": l2 if finite else None,
        "rms": l2 / math.sqrt(tensor.numel()) if finite and tensor.numel() else None,
        "finite": finite,
    }


def parameter_stats(model, *, gradients=False):
    """同时报告逐参数统计和整个模型的统计；缺失梯度单独列出。"""
    per_parameter = {}
    missing = []
    for name, parameter in model.named_parameters():
        value = parameter.grad if gradients else parameter
        if value is None:
            missing.append(name)
        else:
            per_parameter[name] = tensor_stats(value)
    count = sum(item["numel"] for item in per_parameter.values())
    finite = all(item["finite"] for item in per_parameter.values())
    l2 = math.hypot(*(item["l2"] for item in per_parameter.values())) if finite else None
    return {
        "l2": l2,
        "rms": l2 / math.sqrt(count) if finite and count else None,
        "finite": finite,
        "missing_gradients": missing,
        "per_parameter": per_parameter,
    }


@contextmanager
def capture_activations(model, enabled=True):
    """仅在本次 forward 中采样模块输出；共享模块多次调用会分别记录。"""
    records = {}
    handles = []

    def hook_for(name):
        def hook(module, inputs, output):
            if isinstance(output, torch.Tensor):
                records.setdefault(name, []).append(tensor_stats(output))
        return hook

    try:
        if enabled:
            for name, module in model.named_modules():
                if name and not isinstance(module, torch.nn.ModuleList):
                    handles.append(module.register_forward_hook(hook_for(name)))
        yield records
    finally:
        for handle in handles:
            handle.remove()
