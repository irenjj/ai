"""训练入口：uv run python -m cs336_basics.train --help。"""

import argparse
import json
import math
from pathlib import Path

import numpy as np
import torch

from cs336_basics.nn_utils import (
    cross_entropy,
    get_batch,
    gradient_clipping,
    learning_rate_schedule,
    load_checkpoint,
    save_checkpoint,
)
from cs336_basics.optimizer import AdamW
from cs336_basics.transformer.model import TransformerLM


def load_dataset(path: Path, dtype: str, context_length: int, vocab_size: int) -> np.ndarray:
    """只读映射 .npy 或原始二进制 token 文件，分块校验，避免加载整个数组。"""
    if path.suffix == ".npy":
        data = np.load(path, mmap_mode="r", allow_pickle=False)
    else:
        data = np.memmap(path, mode="r", dtype=np.dtype(dtype))
    if data.ndim != 1 or not np.issubdtype(data.dtype, np.integer):
        raise ValueError(f"{path}: expected a one-dimensional integer array")
    if len(data) <= context_length:
        raise ValueError(f"{path}: need at least context_length + 1 tokens")
    for start in range(0, len(data), 1_000_000):
        chunk = data[start : start + 1_000_000]
        if chunk.min() < 0 or chunk.max() >= vocab_size:
            raise ValueError(f"{path}: token IDs must be in [0, vocab_size)")
    return data


@torch.no_grad()
def evaluate(model, dataset, batch_size, context_length, device, num_batches) -> float:
    """验证不改变参数，也不推进训练采样所使用的 NumPy 随机状态。"""
    was_training = model.training
    rng_state = np.random.get_state()
    model.eval()
    try:
        total = 0.0
        for _ in range(num_batches):
            inputs, targets = get_batch(dataset, batch_size, context_length, device)
            total += cross_entropy(model(inputs), targets).item()
        return total / num_batches
    finally:
        np.random.set_state(rng_state)
        model.train(was_training)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train a CS336 Transformer LM on token IDs")
    parser.add_argument("--train-data", type=Path, required=True)
    parser.add_argument("--val-data", type=Path, required=True)
    parser.add_argument("--data-dtype", default="uint16", help="Raw file dtype; .npy uses its own header")
    parser.add_argument("--checkpoint", type=Path, required=True, help="Output checkpoint (latest state)")
    parser.add_argument("--resume", type=Path, help="Restore weights, optimizer and completed steps")
    parser.add_argument("--vocab-size", type=int, required=True)
    parser.add_argument("--context-length", type=int, default=256)
    parser.add_argument("--d-model", type=int, default=256)
    parser.add_argument("--num-layers", type=int, default=4)
    parser.add_argument("--num-heads", type=int, default=4)
    parser.add_argument("--d-ff", type=int, default=704)
    parser.add_argument("--theta", type=float, default=10000.0)
    parser.add_argument("--norm-eps", type=float, default=1e-5)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--steps", type=int, default=10000, help="Total updates, including resumed updates")
    parser.add_argument("--max-lr", type=float, default=3e-4)
    parser.add_argument("--min-lr", type=float, default=3e-5)
    parser.add_argument("--warmup-steps", type=int, default=100)
    parser.add_argument("--decay-steps", type=int, default=10000, help="Absolute cosine decay end step")
    parser.add_argument("--betas", type=float, nargs=2, default=(0.9, 0.95))
    parser.add_argument("--eps", type=float, default=1e-8)
    parser.add_argument("--weight-decay", type=float, default=0.1)
    parser.add_argument("--max-grad-norm", type=float, default=1.0)
    parser.add_argument("--log-every", type=int, default=10)
    parser.add_argument("--eval-every", type=int, default=100)
    parser.add_argument("--eval-batches", type=int, default=10)
    parser.add_argument("--save-every", type=int, default=1000)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    for name in ("vocab_size", "context_length", "d_model", "num_layers", "num_heads", "d_ff",
                 "batch_size", "steps", "log_every", "eval_every", "eval_batches", "save_every"):
        if getattr(args, name) <= 0:
            parser.error(f"{name} must be positive")
    if not 0 <= args.warmup_steps < args.decay_steps:
        parser.error("require 0 <= warmup_steps < decay_steps")
    if not 0 <= args.min_lr <= args.max_lr or not math.isfinite(args.max_lr):
        parser.error("require finite 0 <= min_lr <= max_lr")
    if not all(0 <= beta < 1 for beta in args.betas):
        parser.error("betas must be in [0, 1)")
    for name in ("eps", "norm_eps", "max_grad_norm", "theta"):
        if not math.isfinite(getattr(args, name)) or getattr(args, name) <= 0:
            parser.error(f"{name} must be finite and positive")
    if not math.isfinite(args.weight_decay) or args.weight_decay < 0:
        parser.error("weight_decay must be finite and nonnegative")
    return args


def main() -> None:
    args = parse_args()
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    device = torch.device(args.device)
    train_data = load_dataset(args.train_data, args.data_dtype, args.context_length, args.vocab_size)
    val_data = load_dataset(args.val_data, args.data_dtype, args.context_length, args.vocab_size)
    model = TransformerLM(
        d_model=args.d_model, num_heads=args.num_heads, d_ff=args.d_ff,
        vocab_size=args.vocab_size, context_length=args.context_length,
        num_layers=args.num_layers, theta=args.theta, eps=args.norm_eps,
        device=device, dtype=torch.float32,
    )
    optimizer = AdamW(model.parameters(), lr=args.max_lr, betas=tuple(args.betas),
                      eps=args.eps, weight_decay=args.weight_decay)
    completed = load_checkpoint(args.resume, model, optimizer) if args.resume else 0
    if not isinstance(completed, int) or not 0 <= completed <= args.steps:
        raise ValueError("Checkpoint iteration must be between zero and --steps")
    args.checkpoint.parent.mkdir(parents=True, exist_ok=True)
    # 将运行配置随日志输出，便于复现实验及恢复时提供相同的模型/调度配置。
    print(json.dumps({"config": vars(args), "resumed_steps": completed}, default=str), flush=True)
    model.train()
    loss_sum = 0.0
    logged_steps = 0
    for step in range(completed, args.steps):
        lr = learning_rate_schedule(step, args.warmup_steps, args.decay_steps, args.max_lr, args.min_lr)
        optimizer.param_groups[0]["lr"] = lr
        inputs, targets = get_batch(train_data, args.batch_size, args.context_length, args.device)
        optimizer.zero_grad(set_to_none=True)
        loss = cross_entropy(model(inputs), targets)
        if not torch.isfinite(loss):
            raise FloatingPointError(f"Non-finite loss at step {step}")
        loss.backward()
        gradient_clipping(model.parameters(), args.max_grad_norm)
        optimizer.step()
        completed = step + 1
        loss_sum += loss.item()
        logged_steps += 1
        if completed % args.log_every == 0 or completed == args.steps:
            print(json.dumps({"step": completed, "lr": lr, "train_loss": loss_sum / logged_steps}), flush=True)
            loss_sum, logged_steps = 0.0, 0
        if completed % args.eval_every == 0 or completed == args.steps:
            val_loss = evaluate(model, val_data, args.batch_size, args.context_length, args.device, args.eval_batches)
            print(json.dumps({"step": completed, "val_loss": val_loss}), flush=True)
        if completed % args.save_every == 0 or completed == args.steps:
            # 先写临时文件，再替换，避免中途退出破坏已有检查点。
            temporary = args.checkpoint.with_name(args.checkpoint.name + ".tmp")
            save_checkpoint(model, optimizer, completed, temporary)
            temporary.replace(args.checkpoint)


if __name__ == "__main__":
    main()
