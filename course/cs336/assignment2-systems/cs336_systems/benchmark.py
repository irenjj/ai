"""CS336 benchmark functions shared by the notebook and CLI."""

import torch
import torch.cuda.nvtx as nvtx
from contextlib import contextmanager, nullcontext
from cs336_basics.model import annotated_scaled_dot_product_attention

def cleanup_cuda_memory():
    import sys
    import traceback
    import gc
    import torch

    # 等待旧 CUDA 工作结束，再释放异常引用及显存。
    if torch.cuda.is_available():
        torch.cuda.synchronize()

    tb = getattr(sys, "last_traceback", None)
    if tb is not None:
        traceback.clear_frames(tb)
    del tb

    for name in ("last_traceback", "last_value", "last_type", "last_exc"):
        if hasattr(sys, name):
            setattr(sys, name, None)

    gc.collect()

    if torch.cuda.is_available():
        torch.cuda.synchronize()
        torch.cuda.empty_cache()
        print(f"活跃张量：{torch.cuda.memory_allocated() / 2**30:.2f} GiB")
        print(f"缓存池：  {torch.cuda.memory_reserved() / 2**30:.2f} GiB")

import timeit
import statistics
from contextlib import contextmanager, nullcontext

from cs336_basics import model, optimizer, nn_utils
from einops import rearrange

def run(
    m,
    adamw,
    mode: int,
    input: torch.Tensor,
    target: torch.Tensor,
    record: bool,
    precision: str = "fp32",
) -> dict[str, float] | None:
    if mode not in (1, 2, 3):
        raise ValueError("mode 必须是 1、2 或 3")

    # 梯度清理不计时；mode 1 是 no_grad forward，mode 2/3 保留 autograd。
    if mode != 1:
        adamw.zero_grad(set_to_none=True)

    if precision not in ("fp32", "bf16"):
        raise ValueError("precision 必须是 fp32 或 bf16")

    def autocast_context():
        return torch.autocast(device_type="cuda", dtype=torch.bfloat16) if precision == "bf16" else nullcontext()

    timings = {}

    @contextmanager
    def measure(name):
        # 上个阶段已同步；预热也采用相同的同步边界。
        # 仅测量阶段添加 NVTX 标记，区间包含等待 GPU 完成的时间。
        if record:
            with nvtx.range(name):
                start = timeit.default_timer()
                yield
                torch.cuda.synchronize(input.device)
                timings[name] = timeit.default_timer() - start
        else:
            yield
            torch.cuda.synchronize(input.device)

    torch.cuda.synchronize(input.device)
    if record:
        total_start = timeit.default_timer()

    with measure("forward"):
        with torch.no_grad() if mode == 1 else nullcontext(), autocast_context():
            res = m(input)

    if mode != 1:
        with measure("loss"):
            with autocast_context():
                loss = nn_utils.cross_entropy(
                    rearrange(res, "b t v -> (b t) v"),
                    rearrange(target, "b t -> (b t)"),
                ).mean()
        with measure("backward"):
            loss.backward()
        if mode == 3:
            with measure("step"):
                adamw.step()

    if record:
        # 包含阶段间同步和计时开销，不等于无阶段同步的端到端时间。
        timings["total"] = timeit.default_timer() - total_start
        return timings
    return None


def benchmark(
    d_model: int = 1024,
    d_ff: int = 4096,
    num_layers: int = 24,
    num_heads: int = 16,
    vocab_size: int = 10000,
    context_length: int = 512,
    batch_size: int = 4,
    mode: int = 1,
    warmup_steps: int = 5,
    measurement_steps: int = 10,
    precision: str = "fp32",
    memory_profile: bool = False,
    memory_snapshot_path: str = "memory_snapshot.pickle",
):
    if mode not in (1, 2, 3):
        raise ValueError("mode 必须是 1、2 或 3")
    if warmup_steps < 0 or measurement_steps < 1:
        raise ValueError("warmup_steps 必须 >= 0，measurement_steps 必须 >= 1")

    if precision not in ("fp32", "bf16"):
        raise ValueError("precision 必须是 fp32 或 bf16")
    if precision == "bf16" and not torch.cuda.is_bf16_supported():
        raise RuntimeError("当前 CUDA 设备不支持 BF16")

    m = model.BasicsTransformerLM(vocab_size, context_length, d_model, num_layers, num_heads, d_ff)
    m.to("cuda")
    adamw = optimizer.AdamW(m.parameters())
    origin = torch.randint(
        0, vocab_size,
        (batch_size * context_length + 1,),
        device="cuda",
        dtype=torch.long,
    )
    input = rearrange(origin[:-1], "(b c) -> b c", b=batch_size)
    target = rearrange(origin[1:], "(b c) -> b c", b=batch_size)

    for _ in range(warmup_steps):
        run(m, adamw, mode, input, target, record=False, precision=precision)

    torch.cuda.synchronize()
    if memory_profile:
        torch.cuda.reset_peak_memory_stats()
        memory_start = torch.cuda.memory_allocated()
    if memory_profile:
        torch.cuda.memory._record_memory_history(max_entries=1_000_000)

    samples = []
    try: 
        for _ in range(measurement_steps):
            samples.append(run(m, adamw, mode, input, target, record=True, precision=precision))

        if memory_profile:
            torch.cuda.synchronize()
            torch.cuda.memory._dump_snapshot(memory_snapshot_path)
    finally:
        if memory_profile:
            torch.cuda.memory._record_memory_history(enabled=None)

    summary = {}
    for name in samples[0]:
        values_ms = [sample[name] * 1000 for sample in samples]
        summary[name] = {
            "mean_ms": statistics.mean(values_ms),
            "std_ms": statistics.pstdev(values_ms),
        }

    print(f"precision={precision}, mode={mode}, warmup={warmup_steps}, measurements={measurement_steps}")
    print("细粒度同步计时；total 包含阶段间同步开销；std 为总体标准差（ddof=0）。")
    print(f"{'阶段':<12} {'均值(ms)':>12} {'标准差(ms)':>12}")
    for name, stats in summary.items():
        print(f"{name:<12} {stats['mean_ms']:>12.3f} {stats['std_ms']:>12.3f}")

    if memory_profile:
        summary["memory"] = {
            "start_allocated_bytes": memory_start,
            "peak_allocated_bytes": torch.cuda.max_memory_allocated(),
            "peak_reserved_bytes": torch.cuda.max_memory_reserved(),
            "end_allocated_bytes": torch.cuda.memory_allocated(),
            "snapshot_path": str(memory_snapshot_path),
        }
    return summary

def main(argv=None):
    import argparse

    parser = argparse.ArgumentParser(description="Transformer 分阶段 CUDA 基准测试")
    parser.add_argument("--d-model", type=int, default=1024)
    parser.add_argument("--d-ff", type=int, default=4096)
    parser.add_argument("--num-layers", type=int, default=24)
    parser.add_argument("--num-heads", type=int, default=16)
    parser.add_argument("--vocab-size", type=int, default=10000)
    parser.add_argument("--context-length", type=int, default=512)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--mode", type=int, choices=[1, 2, 3], default=1)
    parser.add_argument("--warmup-steps", type=int, default=5)
    parser.add_argument("--measurement-steps", type=int, default=10)
    parser.add_argument("--precision", choices=["fp32", "bf16"], default="fp32")
    parser.add_argument("--replace", action="store_true")
    parser.add_argument("--memory-profile", action="store_true")
    parser.add_argument("--memory-snapshot-path", default="memory_snapshot.pickle")
    args = parser.parse_args(argv)
    kwargs = vars(args)
    replace = kwargs.pop("replace")

    if replace:
        model.scaled_dot_product_attention = annotated_scaled_dot_product_attention


    return benchmark(**vars(args))


if __name__ == "__main__":
    main()
