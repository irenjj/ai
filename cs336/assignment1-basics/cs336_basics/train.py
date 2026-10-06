"""训练入口：uv run python -m cs336_basics.train --help。"""

import argparse
import json
import math
import time
from datetime import UTC, datetime
from pathlib import Path
from uuid import uuid4

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
from cs336_basics.training_diagnostics import capture_activations, parameter_stats
from cs336_basics.transformer.model import TransformerLM


class ExperimentLogger:
    """每次启动单独保存 JSONL；耗时含验证和保存，不含数据/模型初始化。"""

    def __init__(self, log_dir: Path, run_id: str, device: torch.device):
        self.run_id = run_id
        self.session_id = datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ") + "-" + uuid4().hex[:8]
        directory = log_dir / run_id
        directory.mkdir(parents=True, exist_ok=True)
        self.path = directory / f"{self.session_id}.jsonl"
        self.device = device
        self.file = self.path.open("x", encoding="utf-8")
        if device.type == "cuda":
            torch.cuda.synchronize(device)
        self.started = time.perf_counter()

    def log(self, event: str, step: int, **values) -> None:
        if self.device.type == "cuda":
            torch.cuda.synchronize(self.device)
        record = {
            "run_id": self.run_id,
            "session_id": self.session_id,
            "event": event,
            "timestamp": datetime.now(UTC).isoformat(),
            "step": step,
            "elapsed_seconds": time.perf_counter() - self.started,
            **values,
        }
        line = json.dumps(record, default=str, ensure_ascii=False)
        self.file.write(line + "\n")
        self.file.flush()
        print(line, flush=True)

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        self.file.close()


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
    parser.add_argument("--no-rmsnorm", action="store_true",
                        help="Ablation: remove all block and final RMSNorm layers")
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--overfit-one-batch", action="store_true",
                        help="Debug training by reusing one sampled minibatch for every update")
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
    parser.add_argument("--run-id", help="Experiment name; reuse on resume to group session logs")
    parser.add_argument("--log-dir", type=Path, default=Path("runs"))
    parser.add_argument("--notes", default="", help="Experiment purpose or changes from the baseline")
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
    if args.run_id is None:
        args.run_id = datetime.now(UTC).strftime("run-%Y%m%dT%H%M%SZ") + "-" + uuid4().hex[:8]
    if not args.run_id or args.run_id in (".", "..") or any(c in args.run_id for c in "/\\"):
        parser.error("run-id must be a nonempty directory name without path separators")
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
        use_rmsnorm=not args.no_rmsnorm,
        device=device, dtype=torch.float32,
    )
    optimizer = AdamW(model.parameters(), lr=args.max_lr, betas=tuple(args.betas),
                      eps=args.eps, weight_decay=args.weight_decay)
    completed = load_checkpoint(args.resume, model, optimizer) if args.resume else 0
    if not isinstance(completed, int) or not 0 <= completed <= args.steps:
        raise ValueError("Checkpoint iteration must be between zero and --steps")
    args.checkpoint.parent.mkdir(parents=True, exist_ok=True)
    with ExperimentLogger(args.log_dir, args.run_id, device) as logger:
        logger.log(
            "start", completed, config=vars(args), resumed_steps=completed,
            parameter_count=sum(p.numel() for p in model.parameters()),
            optimizer_config={k: v for k, v in optimizer.param_groups[0].items() if k != "params"},
            log_file=str(logger.path.resolve()),
            timing_scope="Current session only; includes validation/checkpoints, excludes initialization and downtime",
        )
        model.train()
        loss_sum = 0.0
        logged_steps = 0
        final_val_loss = None
        session_start_step = completed
        fixed_batch = (get_batch(train_data, args.batch_size, args.context_length, args.device)
                       if args.overfit_one_batch else None)
        for step in range(completed, args.steps):
            lr = learning_rate_schedule(step, args.warmup_steps, args.decay_steps, args.max_lr, args.min_lr)
            optimizer.param_groups[0]["lr"] = lr
            inputs, targets = (fixed_batch if fixed_batch is not None else
                               get_batch(train_data, args.batch_size, args.context_length, args.device))
            optimizer.zero_grad(set_to_none=True)
            sample_norms = (step + 1) % args.log_every == 0 or step + 1 == args.steps
            with capture_activations(model, enabled=sample_norms) as activations:
                loss = cross_entropy(model(inputs), targets)
            if not torch.isfinite(loss):
                logger.log("nonfinite_loss", step + 1, activations=activations)
                raise FloatingPointError(f"Non-finite loss at step {step}")
            loss.backward()
            if sample_norms:
                weights = parameter_stats(model)
                gradients_before = parameter_stats(model, gradients=True)
            gradient_clipping(model.parameters(), args.max_grad_norm)
            if sample_norms:
                logger.log(
                    "diagnostics", step + 1,
                    timing="Current training batch; weights before optimizer update",
                    activations=activations, weights=weights,
                    gradients_before_clip=gradients_before,
                    gradients_after_clip=parameter_stats(model, gradients=True),
                )
            optimizer.step()
            completed = step + 1
            loss_sum += loss.item()
            logged_steps += 1
            if completed % args.log_every == 0 or completed == args.steps:
                logger.log("train", completed, lr=lr, train_loss=loss_sum / logged_steps, averaged_steps=logged_steps)
                loss_sum, logged_steps = 0.0, 0
            if completed % args.eval_every == 0 or completed == args.steps:
                val_loss = evaluate(model, val_data, args.batch_size, args.context_length, args.device, args.eval_batches)
                final_val_loss = val_loss
                logger.log("validation", completed, val_loss=val_loss, eval_batches=args.eval_batches)
            if completed % args.save_every == 0 or completed == args.steps:
                # 先写临时文件，再替换，避免中途退出破坏已有检查点。
                temporary = args.checkpoint.with_name(args.checkpoint.name + ".tmp")
                save_checkpoint(model, optimizer, completed, temporary)
                temporary.replace(args.checkpoint)
                logger.log("checkpoint", completed, checkpoint=str(args.checkpoint.resolve()))

        logger.log("finished", completed, completed_this_session=completed - session_start_step,
                   val_loss=final_val_loss)


if __name__ == "__main__":
    main()
