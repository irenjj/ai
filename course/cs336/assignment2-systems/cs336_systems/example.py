import torch
import torch.nn as nn
from torch.nn import functional as F

class ToyModel(nn.Module):
    def __init__(self, in_features: int, out_features: int):
        super().__init__()
        self.fc1 = nn.Linear(in_features, 10, bias=False)
        self.ln = nn.LayerNorm(10)
        self.fc2 = nn.Linear(10, out_features, bias=False)
        self.relu = nn.ReLU()

    def forward(self, x):
        print("\n1. 模型参数(autocast 内)")
        for name, parameter in self.named_parameters():
            print(f"   {name}: {parameter.dtype}")

        x = self.fc1(x)
        print(f"\n2. fc1 输出: {x.dtype}")

        x = self.relu(x)

        x = self.ln(x)
        print(f"\n3. LayerNorm 输出: {x.dtype}")

        x = self.fc2(x)
        print(f"\n3. logits 输出: {x.dtype}")

        return x

def benchmarking_mixed_precision_a():
    if not torch.cuda.is_available():
        raise RuntimeError("需要 cuda gpu")

    torch.manual_seed(42)

    # 参数与输入均以 fp32 初始化, 不调用 model.half()
    model = ToyModel(in_features=8, out_features=4).to(
        device="cuda", dtype=torch.float32
    )
    model.train()

    inputs = torch.randn(
        2, 8, device="cuda", dtype=torch.float32
    )

    # 两个样本, 各自对应一个类别, 类别 ID 保持整数
    targets = torch.tensor([1, 3], device="cuda", dtype=torch.long)

    print(f"输入: {inputs.dtype}")
    print(f"autocast 外 fc1 参数: {model.fc1.weight.dtype}")

    model.zero_grad(set_to_none=True)

    # 前向和 loss 在 autocast 内执行
    with torch.autocast(device_type="cuda", dtype=torch.float16):
        logits = model(inputs)
        loss = F.cross_entropy(logits, targets)
        print(f"5. cross_entropy loss: {loss.dtype}")

    # backward 放在 autocast 外
    # 本实验只观察一次梯度的 dtype, 不进行训练更新
    loss.backward()

    print("\n6. 模型梯度")
    for name, parameter in model.named_parameters():
        gradient = parameter.grad
        dtype = gradient.dtype if gradient is not None else None
        print(f"   {name}: {dtype}")

def mixed_precision_accumulation():
    """Run the handout's four sequential accumulation experiments on CPU."""
    expected = 10.0  # 1000 * 0.01 in exact arithmetic.

    def report(label, result):
        value = result.item()
        print(f"{label:<36} result={value:.9f}  abs_error={abs(value - expected):.9f}")

    s = torch.tensor(0, dtype=torch.float32)
    for _ in range(1000):
        s += torch.tensor(0.01, dtype=torch.float32)
    report("FP32 input, FP32 accumulator", s)

    s = torch.tensor(0, dtype=torch.float16)
    for _ in range(1000):
        s += torch.tensor(0.01, dtype=torch.float16)
    report("FP16 input, FP16 accumulator", s)

    s = torch.tensor(0, dtype=torch.float32)
    for _ in range(1000):
        s += torch.tensor(0.01, dtype=torch.float16)
    report("FP16 input, FP32 accumulator", s)

    s = torch.tensor(0, dtype=torch.float32)
    for _ in range(1000):
        x = torch.tensor(0.01, dtype=torch.float16)
        s += x.type(torch.float32)
    report("FP16 -> FP32, FP32 accumulator", s)


def main() -> None:
    # mixed_precision_accumulation()

    benchmarking_mixed_precision_a()
   

if __name__ == "__main__":
    main()
