# Q & A

模型规模表:

规模     d_model       d_ff         num_layers      num_heads
small     768           3072            12              12
medium    1024          4096            24              16
large     1280          5120            36              20
xl        2560          10240           32              32
10B       4608          12288           50              36

vocab_size: 10,000
batch_size: 4
默认 context_length: 512


## 对上面各个规模模型, 测量 forward, backward, optimizer 步骤指标, 5 个 warmup, 10个 measurement
测量时间平均值 + 方差. 一次前向传播需要多久? 反向传播呢? 各次测量差异很大? 还是标准差很小?

medium:
Stage              Total (ms)   Count      Mean (ms)    StdDev (ms)
-------------------------------------------------------------------
step                 3807.367      10        380.737          1.886
forward              1067.573      10        106.757          0.548
loss                    3.951      10          0.395          0.021
backward             2242.368      10        224.237          0.940
optimizer             474.516      10         47.452          1.602

## 如果没有预热步骤, 结果如何? 

没有预热:
Stage              Total (ms)   Count      Mean (ms)    StdDev (ms)
-------------------------------------------------------------------
step                 4162.603      10        416.260        114.870
forward              1294.167      10        129.417         71.977
loss                   27.912      10          2.791          7.459
backward             2350.641      10        235.064         34.677
optimizer             473.426      10         47.343          0.874
无预热标准差偏大, 可能是首次迭代偏慢, 触发了 cuda 的模块/库的惰性初始化, 工作空间分配, 缓存建立等开销

一次预热:
Stage              Total (ms)   Count      Mean (ms)    StdDev (ms)
-------------------------------------------------------------------
step                 3800.128      10        380.013          1.343
forward              1067.881      10        106.788          0.722
loss                    4.604      10          0.460          0.169
backward             2239.051      10        223.905          0.849
optimizer             471.240      10         47.124          0.348

## nsys profile 性能分析, 选择两种模型规模, 再选择三个大于 128, 为 2 的幂上下文长度, 并且最大长度应该是显存能容纳的最大上下文

```sh
uv run nsys profile --cpuctxsw=none  --trace=cuda,cublas,nvtx,osrt   --output=benchmark_mode2 --force-overwrite true  python -m cs336_systems.benchmark   --vocab-size 10000   --context-length 512   --d-model 1024   --num-layers 24   --num-heads 16   --d-ff 4096   --batch-size 4   --device cuda   --run-mode 3   --warmup-steps 5   --measure-steps 10
```

1. 前向传播和之前 python 标准库测试的一致吗?
> 一致

前向传播期间, 哪个 CUDA 内核累计占用的 GPU 时间最多?
nsight systems: stats system view -> GUDA GPU Kernel Summary -> Total Time(从大到小)

> ampere_sgemm_128x64_tn, 占 53.9%

ampere_sgemm_128x64_tn 是执行单精度矩阵乘法的 CUDA kernel

ampere: 内核实现家族的名称
sgemm: Single-precision General Matrix Multiply, 单精度通用矩阵乘法
128x64: 内部计算分块尺寸
tn: 内部操作数的转置/布局组合, t: transpose, n: non-transpose

2. 在模型一次前向传播中, 这个内核会被调用多少次

> 看 instances, 会被调用 3380 次

3. 同时执行前向和反向传播时, 耗时最多的还是这个内核吗?
查看 Stats System View 下的 GUDA GPU Kernel Summary 并使用 NVTX 范围进行过滤, 以确定模型的哪个部分对应哪些内核 

> 还是 ampere_sgemm_128x64_tn, 但是总耗时占比变成了 17.6%, 第二耗时多的(11.7%) 是 at::native::vectorized_elementwise_kernel<(int)4, at::native::BinaryFunctor<float, float, float, at::native::binary_internal::MulFunctor



4. 尽管大多数 FLOP 都发生在矩阵乘法中, 你会发现一些其他内核占据了不可忽略的总体运行时间.
除了矩阵乘法外, 前向传播还有哪些内核占用了相当比例的 CUDA 运行时间?

forward 中的矩阵乘法有: sgemm_*, cutlass_*
还有很多 elementwise_kernel, vectorized_elementwise_kernel 这种通用逐元素计算框架

ampere_sgemm            矩阵乘法            线性投影, 注意力, FFN
cutlass...sgemm

MulFunctor              逐元素乘法          SwiGLU 门控, RMSNorm 权重乘法, RoPE

DivFunctor              逐元素除法          残差连接, RoPE 等

CUDAFunctor_add         逐元素加法          softmax 归一化, 注意力分数缩放

exp_kernel              指数运算 e^x        softmax

reduce...MaxOps         沿维度求最大值      稳定 softmax 减最大值步骤

reduce...sum_functor    沿维度求和          softmax 分母等

reduce...MeanOps        沿维度求平均值      RMSNorm 的均方计算

reduce...sum_functor    沿维度求和          softmax 分母等

reduce...MeanOps        沿维度求均值        RMSNorm 的均方计算

pow_tensor_scalar       张量元素求幂        RMSNorm 中的平方


5. 使用 AdamW 分析一个完整的训练步骤(前向, 计算 loss, 反向, 优化器更新) 与推理(前向) 相比
矩阵乘法耗时占比如何变化? 其他内核呢



6. 比较模型自注意力层在前向传播中 softmax 运算与矩阵乘法运算的运行时间, 运行时间的差异与
FLOP 数量差异相比如何

softmax 时间占两次矩阵乘法的 2.73 倍, 
softmax 的运行时间和 FLOPs 数量不成比例. 该实现将 softmax 拆成多个内核, 反复读写注意力矩阵, 并执行
规约操作. 矩阵乘法通过数据复用高效执行大量乘法. 所以 softmax 虽然计算量远少于矩阵乘法, 却消耗了更多
GPU 时间

## 混合精度

目前为止, 所有数据类型都是 torch.float32(FP32)

格式  总位数    符号位  指数位  小数位  约有效十进制位数    最大有限值
fp32    32          1       8       23          7                  3.4 x 10^38
fp16    16          1       5       10          3                   65,504
bf16    16          1       8       7          2~3                  3.39 x 10^38

B200 对于 FP32 的吞吐是 80T FLOPs, 对 FP16/BP16 的精度是 2500T FLOPs.

fp16 精度低, 直接使用 fp16 训练时会变成 0. 所以使用 fp16 训练时通常会采用损失缩放: 将损失
乘一个缩放因子, 来增大梯度的幅值

通过 torch.autocast 来实现混合精度

```
model : torch.nn.Module
dtype: torch.dtype = torch.bfloat16
x: torch.Tensor = xxx
with torch.autocast(device_type="cuda", dtype=dtypte):
    y = model(x)
```

### mixed_precision_accumulation

几种精度加法:
fp32 + fp32 10.000133514404297
fp16 + fp16 9.953125
fp32 + fp16 10.00213623046875
fp32 + fp16 change to fp32 10.00213623046875

### benchmarking_mixed_precision

```py
class ToyModel(nn.Module):
    def __init__(self, in_features: int, out_features: int):
        super().__init__()
        self.fc1 = nn.Linear(in_features, 10, bias=False)
        self.ln = nn.LayerNorm(10)
        self.fc2 = nn.Linear(10, out_features, bias=False)
        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.relu(self.fc1(x))
        x = self.ln(x)
        x = self.fc2(x)
        return x
```

1. 假设模型最初参数为 fp32, 希望使用 fp16 自动转换混合精度
- autocast 上下文中的模型参数:
- ToyModel.fc1 的输出: fp16
- ToyModel.ln 的输出: fp32
- 模型预测的 logits: fp16
- 损失: fp32
- 模型梯度: fp32

2. fp16 混合精度自动转换对层归一化和前馈层采取了不同方式.
层归一化中哪些部分对混合精度敏感? x^2 如果用 fp16 会溢出(fp16 最大值 65,504)
如果使用 BF16 还需要对层归一化做不同处理吗? 需要, 虽然不会溢出, 但是会有以下问题:
- 均值, 方差的归约计算: 累加多项会累积舍入误差
- 中心化 x_i - u: 两者接近时, 输入和均值的微小误差可能明显影响差值
- 方差及归一化分母: 方差较小时, 统计误差可能明显影响最终输出.

## 显存分析

```py
# 开始记录显存历史
torch.cuda.memory._record_memory_history(max_entries=1000000)
... # 需要分析显存的部分
torch.cuda.memory._dump_snapshot("memory_snapshot.pickle")
# 停止记录历史
torch.cuda.memory._record_memory_history(enabled=None)
```

对表 1 中 medium 模型, 在上下文长度为 128 和 2048 时, 分析包含前向传播, 反向传播和优化器更新
的完整训练步骤.


1. 对每个上下文长度, 前向传播时的峰值显存是多少? 完整训练步骤的峰值显存又是多少?

medium 模型, mode 1, 上下文 128:  峰值 1.6GB
                     上下文 512:   峰值 1.6~1.8GB
             mode 3,  上下文 128:  峰值 6.5GB
                     上下文 512:  峰值 13.5GB

2. 开启混合精度时, 前向传播和完整训练步骤峰值显存? 混合精度是否显著影响显存?

mode 1
128, 512 峰值变高了, 混合精度会额外存一份 bf16权重

```
前向开始：保留 FP32 权重                  ≈ 1.6 GiB
逐层执行：陆续生成并缓存 BF16 权重副本     显存逐渐上升
前向结束：退出 autocast，释放缓存副本      回落到 ≈ 1.6 GiB
下一轮：重复
```

mode3
128, 512 峰值变低了, 因为 mode 3 要保存大量反向所需要的激活, BF16 节省激活显存

3. memory_viz 中 前向传播显存快照的 active memory timeline, 降低 detail 级别时, 工具会
隐藏相应比例的较小分配, 如果设为 10% 只会显示最大的 10% 分配. 所显示的最大分配大小
是多少? 查看调用栈, 能否判断这些分配来自哪里?
