import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint

from cs336_basics.model import TransformerBlock, RotaryEmbedding

class ToyModel(nn.Module):
    def __init__(self, in_features: int, out_features: int):
        super().__init__()
        self.fc1 = nn.Linear(in_features, 10, bias=False)
        self.ln = nn.LayerNorm(10)
        self.fc2 = nn.Linear(10, out_features, bias=False)
        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.fc1(x)
        print(f"fc1: {x.dtype}")
        print(f"fc1.weight: {self.fc1.weight.dtype}")

        x = self.relu(x)
        print(f"relu: {x.dtype}")
        x = self.ln(x)
        print(f"ln: {x.dtype}")
        x = self.fc2(x)
        print(f"fc2: {x.dtype}")
        return x

class ToyRMSNorm(nn.Module):
    def __init__(
        self,
        hidden_size: int,
        eps: float = 1e-5,
        device=None,
    ):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(hidden_size, device=device))
        self.eps = eps

    def forward(self, x):
        rms = torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)
        x = x * rms
        return self.weight * x


def test_save_tensor_rmsnorm():
    def pack_hook(t):
        shape, dtype, grad_fn = t.shape, t.dtype, t.grad_fn
        print(f"Saving residual: {shape=}, {dtype=}, {grad_fn=}")
        return t

    def unpack_hook(t):
        shape, dtype, grad_fn = t.shape, t.dtype, t.grad_fn
        print(f"Loading residual: {shape=}, {dtype=}, {grad_fn=}")
        return t

    x = torch.randn((4, 512, 2560), requires_grad=True)
    # ln = ToyRMSNorm(x.shape[-1])
    ln = torch.compile(ToyRMSNorm(x.shape[-1]))
    with torch.autograd.graph.saved_tensors_hooks(pack_hook, unpack_hook):
        y = ln(x)
        y.sum().backward()


def activation_checkpointing_example(compile_block=False):
    """讲义 3.2：四次 block 调用，对比每两个 block 设置一个检查点。

    使用小尺寸 CPU 输入；与讲义一样，四次调用共享同一个 block 的参数。
    compile_block=True 可以观察编译与 checkpoint 组合后的保存行为。
    """
    torch.manual_seed(42)
    batch_size, context_length, d_model = 2, 64, 128
    block = TransformerBlock(
        d_model=d_model, d_ff=512, num_heads=4,
        positional_encoder=RotaryEmbedding(
            dim=32, context_length=context_length),
    )
    run_block = torch.compile(block, fullgraph=True) if compile_block else block
    original_x = torch.randn(batch_size, context_length, d_model)

    def run(use_checkpoint):
        block.zero_grad(set_to_none=True)
        x = original_x.clone().requires_grad_(True)
        phase = "forward"
        calls = {"forward": 0, "backward": 0}
        saved_bytes = 0
        saved_count = 0
        name = "checkpoint: 2 + 2 blocks" if use_checkpoint else "normal: 4 blocks"
        print(f"\n=== {name}; compile={compile_block} ===")

        def two_blocks(value):
            # 非重入 checkpoint 可能在所需张量重算完后提前停止，故在入口计数。
            calls[phase] += 1
            print(f"[{phase}] enter two_blocks #{calls[phase]}")
            return run_block(run_block(value))

        def pack_hook(t):
            nonlocal saved_bytes, saved_count
            if isinstance(t, nn.Parameter):
                return t
            saved_count += 1
            saved_bytes += t.numel() * t.element_size()
            print(f"Saving residual: shape={tuple(t.shape)}, dtype={t.dtype}")
            return t

        def unpack_hook(t):
            print(f"Loading residual: shape={tuple(t.shape)}, dtype={t.dtype}")
            return t

        with torch.autograd.graph.saved_tensors_hooks(pack_hook, unpack_hook):
            if use_checkpoint:
                y = checkpoint(two_blocks, x, use_reentrant=False)
                y = checkpoint(two_blocks, y, use_reentrant=False)
            else:
                y = two_blocks(two_blocks(x))

            # 在 backward 之前统计，观察跨越前后向边界保留的张量。
            print(f"Forward saved tensors: {saved_count}; "
                  f"logical size: {saved_bytes / 1024**2:.4f} MiB")
            phase = "backward"
            y.sum().backward()

        print(f"two_blocks calls: forward={calls['forward']}, "
              f"backward recomputation={calls['backward']}")
        assert x.grad is not None
        parameter_grads = {}
        for name, parameter in block.named_parameters():
            assert parameter.grad is not None, name
            parameter_grads[name] = parameter.grad.detach().clone()
        return y.detach(), x.grad.detach().clone(), parameter_grads

    normal_y, normal_dx, normal_grads = run(use_checkpoint=False)
    checkpoint_y, checkpoint_dx, checkpoint_grads = run(use_checkpoint=True)
    torch.testing.assert_close(checkpoint_y, normal_y)
    torch.testing.assert_close(checkpoint_dx, normal_dx)
    for name in normal_grads:
        torch.testing.assert_close(checkpoint_grads[name], normal_grads[name])
    print("\nPASS: outputs, input gradients and parameter gradients match.")
    print("注意：hook 统计是逻辑张量字节数，不是峰值显存；未去重共享存储，"
          "参数的 view 也可能被计入。checkpoint 的内部重计算保存不一定被外层 hook 看到。")


def mixed_precision_accumulation():
    # fp32 + fp32
    s =  torch.tensor(0, dtype=torch.float32)
    for _ in range(1000):
        s += torch.tensor(0.01, dtype=torch.float32)
    print(f"fp32 + fp32 {s}")

    # fp16 + fp16
    s = torch.tensor(0, dtype=torch.float16)
    for _ in range(1000):
        s += torch.tensor(0.01, dtype=torch.float16)
    print(f"fp16 + fp16 {s}")

    # fp32 + fp16
    s = torch.tensor(0, dtype=torch.float32)
    for _ in range(1000):
        s += torch.tensor(0.01, dtype=torch.float16)
    print(f"fp32 + fp16 {s}")

    # fp32 + fp16 change to fp32
    s = torch.tensor(0, dtype=torch.float32)
    for _ in range(1000):
        x = torch.tensor(0.01, dtype=torch.float16)
        s += x.type(torch.float32)
    print(f"fp32 + fp16 change to fp32 {s}")

def benchmarking_mixed_precision():
    toy = ToyModel(8, 10).to("cuda")

    x = torch.randn(8, device="cuda", dtype=torch.float32)
    # 单个样本的类别标签，对应形状为 [10] 的 logits。
    target = torch.randint(0, 10, (), device="cuda", dtype=torch.long)

    with torch.autocast(device_type="cuda", dtype=torch.float16):
        print("模型参数（autocast 内）:")
        for name, parameter in toy.named_parameters():
            print(f"  {name}: {parameter.dtype}")

        logits = toy(x)
        print(f"logits: {logits.dtype}")
        loss = F.cross_entropy(logits, target)
        print(f"loss: {loss.dtype}")

    # 本例只观察一次反向的 dtype；不执行优化器更新。
    loss.backward()
    print("模型梯度（backward 后）:")
    for name, parameter in toy.named_parameters():
        grad = parameter.grad
        print(f"  {name}.grad: {grad.dtype if grad is not None else None}")


def main() -> None:
    # mixed_precision_accumulation()

    # benchmarking_mixed_precision()

    activation_checkpointing_example()

if __name__ == "__main__":
    main()
