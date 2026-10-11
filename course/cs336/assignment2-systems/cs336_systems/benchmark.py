import argparse
import torch
import torch.nn.functional as F
import timeit
import statistics
from contextlib import contextmanager, nullcontext
import cs336_basics.model

from cs336_basics.model import BasicsTransformerLM
from cs336_basics.optimizer import AdamW
from einops import rearrange
import torch.cuda.nvtx as nvtx

cs336_basics.model.scaled_dot_product_attention = cs336_basics.model.annotated_scaled_dot_product_attention

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--vocab-size", type=int, required=True)
    parser.add_argument("--context-length", type=int, default=256)
    parser.add_argument("--d-model", type=int, default=256)
    parser.add_argument("--num-layers", type=int, default=4)
    parser.add_argument("--num-heads", type=int, default=4)
    parser.add_argument("--d-ff", type=int, default=704)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--rope-theta", type=float, default=10000.0)
    parser.add_argument("--betas", type=float, nargs=2, default=(0.9, 0.95))
    parser.add_argument("--eps", type=float, default=1e-8)
    parser.add_argument("--weight-decay", type=float, default=0.1)
    parser.add_argument("--warmup-steps", type=int, default=5)
    parser.add_argument("--measure-steps", type=int, default=10)
    parser.add_argument("--max-lr", type=float, default=3e-4)
    # 1: forward, 2: forward + loss + backward, 3: forward + loss + backward + 优化器更新
    parser.add_argument("--run-mode", type=int, choices=(1, 2, 3), default=1)
    parser.add_argument("--mixed-precision", action="store_true",
                        help="Use CUDA BF16 autocast for forward and loss; keep parameters in FP32")


    args = parser.parse_args()
    if args.warmup_steps < 0:
        parser.error("--warmup-steps must be >= 0")
    if args.measure_steps < 1:
        parser.error("--measure-steps must be >= 1")

    return args

def benchmark(
    model,
    train_data,
    targets,
    mode=1,
    warmup_steps=5,
    measure_steps=10,
    optimizer=None,
    mixed_precision=False,
):
    if mode not in (1, 2, 3):
        raise ValueError("mode must be 1, 2, or 3")
    if warmup_steps < 0 or measure_steps < 1:
        raise ValueError("warmup_steps must be >= 0 and measure_steps must be >= 1")
    if mode == 3 and optimizer is None:
        raise ValueError("full 模式需要 optimizer")

    device = train_data.device
    if mixed_precision:
        if device.type != "cuda":
            raise ValueError("--mixed-precision requires a CUDA device")
        with torch.cuda.device(device):
            if not torch.cuda.is_bf16_supported(including_emulation=False):
                raise ValueError("The selected CUDA device does not support native BF16")
    model.train()

    def synchronize():
        if device.type == "cuda":
            torch.cuda.synchronize(device)

    stage_times_ms = {name: [] for name in ("forward", "loss", "backward", "optimizer")}
    step_times_ms = []

    def nvtx_range(name):
        return nvtx.range(name) if device.type == "cuda" else nullcontext()

    @contextmanager
    def timed_stage(name, record):
        # 排空此前的 GPU 工作；阶段结束后等待 GPU 完成再停止计时。
        # 预热也执行相同的同步，计时包含 CPU 发起操作和末尾同步的开销。
        synchronize()
        with nvtx_range("optimize" if name == "optimizer" else name):
            start = timeit.default_timer() if record else None
            yield
            synchronize()
            if record:
                elapsed_ms = (timeit.default_timer() - start) * 1000
        if record:
            stage_times_ms[name].append(elapsed_ms)

    def run_step(record):
        # 每步单独进入上下文，预热和测量使用相同精度；反向与更新在上下文外。
        autocast_context = (
            torch.autocast(device_type="cuda", dtype=torch.bfloat16)
            if mixed_precision else nullcontext()
        )
        with autocast_context:
            with timed_stage("forward", record):
                logits = model(train_data)

            if mode != 1:
                with timed_stage("loss", record):
                    loss = F.cross_entropy(
                        rearrange(logits, "b c v -> (b c) v"),
                        rearrange(targets, "b c -> (b c)"),
                    )

        if mode != 1:
            with timed_stage("backward", record):
                loss.backward()

        if mode == 3:
            with timed_stage("optimizer", record):
                optimizer.step()

    with torch.enable_grad():
        # 预热执行相同的阶段和同步，但不加入正式测量样本。
        synchronize()
        warmup_start = timeit.default_timer()
        with nvtx_range("warmup"):
            for _ in range(warmup_steps):
                model.zero_grad(set_to_none=True)
                run_step(record=False)
                synchronize()
        warmup_total_ms = (timeit.default_timer() - warmup_start) * 1000

        synchronize()
        measurement_start = timeit.default_timer()
        with nvtx_range("measurement"):
            for _ in range(measure_steps):
                model.zero_grad(set_to_none=True)
                synchronize()
                step_start = timeit.default_timer()
                run_step(record=True)
                synchronize()
                step_times_ms.append((timeit.default_timer() - step_start) * 1000)
        measurement_total_ms = (timeit.default_timer() - measurement_start) * 1000

    def summarize(times_ms):
        if not times_ms:
            return None
        return {
            "count": len(times_ms),
            "total_ms": sum(times_ms),
            "mean_ms": statistics.mean(times_ms),
            # 与 nsys nvtx_sum 的样本标准差口径一致；单个样本记为 0。
            "std_ms": statistics.stdev(times_ms) if len(times_ms) > 1 else 0.0,
            #"times_ms": times_ms,
        }

    return {
        "mode": mode,
        "precision": "BF16 autocast (FP32 parameters)" if mixed_precision else "FP32",
        "timing_scope": "stages=GPU-synchronized wall time (includes CPU launch and end-sync overhead); step=stage-synchronized, excludes zero_grad",
        "warmup_total_ms": warmup_total_ms,
        "measurement_total_ms": measurement_total_ms,
        "step": summarize(step_times_ms),
        **{name: summarize(times) for name, times in stage_times_ms.items()},
    }

def print_metrics(metrics):
    print(f"Mode: {metrics['mode']}")
    print(f"Precision: {metrics['precision']}")
    print(f"Timing scope: {metrics['timing_scope']}")
    print(f"Warmup total: {metrics['warmup_total_ms']:.3f} ms")
    print()
    header = f"{'Stage':<14} {'Total (ms)':>14} {'Count':>7} {'Mean (ms)':>14} {'StdDev (ms)':>14}"
    print(header)
    print('-' * len(header))
    print(f"{'measurement':<14} {metrics['measurement_total_ms']:>14.3f} {1:>7} {'—':>14} {'—':>14}")
    for name in ("step", "forward", "loss", "backward", "optimizer"):
        stats = metrics[name]
        if stats is None:
            continue
        print(
            f"{name:<14} {stats['total_ms']:>14.3f} {stats['count']:>7} "
            f"{stats['mean_ms']:>14.3f} {stats['std_ms']:>14.3f}"
        )


def main() -> None:
    args = parse_args()
    torch.manual_seed(args.seed)
    device = torch.device(args.device)

    n = args.batch_size * args.context_length
    tokens = torch.randint(
        low=0,
        high=args.vocab_size,
        size=(n+1,),
        dtype=torch.long,
        device=device
    )
    train_data = rearrange(tokens[:-1], "(b c) -> b c", b=args.batch_size)
    targets = rearrange(tokens[1:], "(b c) -> b c", b=args.batch_size)

    model = BasicsTransformerLM(
        args.vocab_size, args.context_length, args.d_model,
        args.num_layers, args.num_heads, args.d_ff, args.rope_theta,
    ).to(device)

    optimizer = AdamW(
        model.parameters(), args.max_lr, args.betas, args.eps, args.weight_decay)


    metrics = benchmark(
        model,
        train_data,
        targets,
        args.run_mode,
        args.warmup_steps,
        args.measure_steps,
        optimizer,
        mixed_precision=args.mixed_precision)

    print_metrics(metrics)
    
if __name__ == "__main__":
    main()
