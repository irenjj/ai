import torch
import statistics
import timeit
from itertools import product

from cs336_basics.model import scaled_dot_product_attention

def benchmark_attention(
    d_model: int,
    seq_len: int,
    warmup_steps: int = 5,
    measurement_steps: int = 100,
):
    device = torch.device("cuda")
    batch_size = 8

    # 自注意力: query 数量和 key 数量都等于 seq_len
    # 本题取 d_k = d_v = d_model
    shape = (batch_size, seq_len, d_model)

    q = torch.randn(
        shape, device=device, dtype=torch.float32, requires_grad=True
    )
    k = torch.randn(
        shape, device=device, dtype=torch.float32, requires_grad=True
    )
    v = torch.randn(
        shape, device=device, dtype=torch.float32, requires_grad=True
    )

    # attention 的输出不是标量, 需要提供同形状的上游梯度
    grad_output = torch.randn(shape, device=device, dtype=torch.float32)

    def clear_grads():
        q.grad = None
        k.grad = None
        v.grad = None

    # 1. 预热 forward 和 backward
    for _ in range(warmup_steps):
        clear_grads()

        output = scaled_dot_product_attention(q, k, v)
        torch.cuda.synchronize(device)

        output.backward(grad_output)
        torch.cuda.synchronize(device)

        del output

    clear_grads()
    torch.cuda.synchronize(device)

    # 包含 Q, K, V 和 grad_output 的基础占用
    baseline_bytes = torch.cuda.memory_allocated(device)

    # 2. 单独测 forward
    # 保留 autograd, 因为后续要研究为 backward 保存的显存.
    forward_times = []

    for _ in range(measurement_steps):
        start = timeit.default_timer()

        output = scaled_dot_product_attention(q, k, v)
        torch.cuda.synchronize(device)

        forward_times.append(timeit.default_timer() - start)

        # 及时释放本轮计算图, 避免多轮图同时驻留
        del output

    # 3. 单独测 backward
    backward_times = []
    before_backward_bytes = []

    for _ in range(measurement_steps):
        clear_grads()

        # 每次 backward 都重新创建计算图
        # 这次 forward 不计入 backward 时间
        output = scaled_dot_product_attention(q, k, v)
        torch.cuda.synchronize(device)

        before_backward_bytes.append(
            torch.cuda.memory_allocated(device)
        )

        start = timeit.default_timer()

        output.backward(grad_output)
        torch.cuda.synchronize(device)

        backward_times.append(timeit.default_timer() - start)

        del output

    assert all(t.grad is not None for t in (q, k, v))

    mib = 1024 ** 2

    return {
        "d_model": d_model,
        "seq_len": seq_len,
        "status": "ok",
        "forward_mean_ms": statistics.mean(forward_times) * 1000,
        "forward_std_ms": statistics.pstdev(forward_times) * 1000,
        "backward_mean_ms": statistics.mean(backward_times) * 1000,
        "backward_std_ms": statistics.pstdev(backward_times) * 1000,
        "before_backward_mib": max(before_backward_bytes) / mib,
        # 该增量包含输出及其计算图保存的张量，不等于纯 saved tensors。
        "forward_live_increment_mib": (
            max(before_backward_bytes) - baseline_bytes
        ) / mib,
    }

def run_case(d_model, seq_len):
    # 在这一层捕获异常, 使 benchmark 的局部张量随后可以被释放
    try:
        return benchmark_attention(d_model, seq_len)
    except torch.cuda.OutOfMemoryError:
        return {
            "d_model": d_model,
            "seq_len": seq_len,
            "status": "OOM",
        }

def main():
    torch.manual_seed(42)

    d_models = [16, 32, 64, 128]
    seq_lengths = [256, 1024, 4096, 8192, 16384]
    results = []

    print(
        "| d_model | seq_len | forward ms | backward ms "
        "| backward 前 MiB | 状态 |"
    )
    print("|---:|---:|---:|---:|---:|:---|")

    for d_model, seq_len in product(d_models, seq_lengths):
        result = run_case(d_model,seq_len)
        results.append(result)

        # 清理放在单个配置结束之后, 不放进计时区域
        gc.collect()
        torch.cuda.empty_cache()

        if result["status"] == "OOM":
            print(f"| {d_model} | {seq_len} | — | — | — | OOM |")
        else:
            print(
                f"| {d_model} | {seq_len} "
                f"| {result['forward_mean_ms']:.3f} "
                f"| {result['backward_mean_ms']:.3f} "
                f"| {result['before_backward_mib']:.2f} "
                f"| ok |"
            )

        return results

if __name__ == "__main__":
    main()