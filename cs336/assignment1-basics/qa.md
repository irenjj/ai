## 实现大纲

```
assignment1-basics/
├── cs336_basics/
│   ├── __init__.py
│   │
│   ├── bpe.py                   # 已有：训练 BPE，生成词表和合并规则
│   ├── tokenizer.py             # 新增：编码、解码、流式编码、加载词表
│   ├── pretokenization_example.py  # 已有：预分词参考代码
│   ├── bpe_example.py           # 已有：学习示例
│   │
│   ├── transformer/
│   │   ├── __init__.py          # 建议补充
│   │   ├── layers.py           # 已有：基础模块
│   │   ├── attention.py        # 新增：注意力
│   │   ├── block.py            # 新增：一个 Transformer Block
│   │   └── model.py            # 新增：完整 Transformer LM
│   │
│   ├── nn_utils.py             # 新增：SiLU、softmax、交叉熵、梯度裁剪
│   ├── optimizer.py            # 新增：AdamW
│   ├── schedule.py             # 新增：warmup + cosine 学习率调度
│   ├── data.py                 # 新增：采样训练 batch
│   ├── checkpoint.py           # 新增：保存和恢复训练状态
│   │
│   ├── prepare_data.py         # 新增：调用 tokenizer，将语料保存为 token IDs
│   ├── train.py                # 新增：训练入口、验证、日志、保存 checkpoint
│   └── generate.py             # 新增：加载模型、自回归生成、采样
│
├── tests/
│   ├── adapters.py             # 将测试接口连接到你的实现
│   └── test_*.py               # 作业提供的测试
│
├── configs/                    # 建议新增：训练和消融实验配置
└── cs336_basics/ein/            # 已有：einops 学习资料
```


## einops 基础

主要是 4 个函数:
```py
from einops import rearrange, reduce, einsum, repeat
```

### rearrange: 改变形状和轴顺序

场景:
1. RoPE: 把相邻特征拆成对, 旋转后合并回来, 调整位置张量用于广播
2. 多头 Attention: 将特征维拆成多个 head, 交换轴, 合并各 head 输出
3. 交叉熵: 将 batch 和 sequence 合并, 把每个 token 当成一个样本

```py
# 交换轴：(2, 3, 4) → (2, 4, 3)
rearrange(x, "b s d -> b d s")

# 拆分并换轴，多头注意力：(2, 3, 8) → (2, 2, 3, 4)
rearrange(x, "b s (h d) -> b h s d", h=2)

# 合并轴，多头输出：(2, 2, 3, 4) → (2, 3, 8)
rearrange(x, "b h s d -> b s (h d)")

# 增加单元素轴，准备广播：(3,) → (3, 1)
rearrange(x, "s -> s 1")

# 删除单元素轴：(3, 1) → (3,)
rearrange(x, "s 1 -> s")
```

### reduce:

场景:
1. RMSNorm: 沿特征维计算平方的平均值
2. Softmax: 沿指定维度取最大值, 求和
3. 损失统计, 梯度裁剪: 平均损失, 累加梯度平方

```py
# 对最后一维求平均，并保留维度：(2, 3, 8) → (2, 3, 1)
reduce(x, "... d -> ... 1", "mean")

# 所有元素求和，得到零维 Tensor
reduce(x, "... ->", "sum")
```

### einsum

场景:
1. Linear, SwiGLU, Q/K/V 投影, LM Head: 线性变换
2. Attention: Query 和 Key 两两电机, 再对 Value 加权求和
3. RoPE: 通过位置与频率的外积生成角度表

```py
# 线性变换：x (..., 4)，w (5, 4) → (..., 5)
einsum(x, w, "... i, o i -> ... o")

# 两组向量两两点积：q (..., 3, 4)，k (..., 5, 4) → (..., 3, 5)
einsum(q, k, "... q d, ... k d -> ... q k")

# 外积：a (3,)，b (4,) → (3, 4)
einsum(a, b, "i, j -> i j")
```

### repeat:

场景:
1. RoPE, 多头 Attention: 将位置编号或 mask 重复到 batch/head 维度；通常直接广播即可

```py
# 同一组位置编号重复给 2 个 batch：(3,) → (2, 3)
repeat(x, "s -> b s", b=2)
```

## Q & A

1. chr(0) 返回哪个 unicode 字符:

> '\x00'

2. 该字符串表示 (__repr__()) 与打印出来的表示有什么不同

```
>>> '\x00'.__repr__()
"'\\x00'"
>>> print('\x00')

```

3. 当这个字符出现在文本中时, 会发生什么? 

```
>>> chr(0)
'\x00'
>>> print(chr(0))

>>> "this is a test" + chr(0) + "string"
'this is a test\x00string'
>>> print("this is a test" + chr(0) + "string")
this is a teststring
```

4. 与 utf-16 或 utf-32 相比, utf-8 编码的字节上训练分词器有哪些优势?

> utf8 字节表示更少相同 token 可以用更少字节表示, 训练效率更高

5. 考虑下面这个意图将 UTF-8 字节串解码为 Unicode 字符串的（错误）函数。为什么它不正
确？请给出一个会产生错误结果的输入字节串示例。

```
>>> def decode_utf8_bytes_to_str_wrong(bytestring: bytes):
...     return "".join([bytes([b]).decode("utf-8") for b in bytestring])
...
>>> decode_utf8_bytes_to_str_wrong("hello".encode("utf-8"))
'hello'
>>> decode_utf8_bytes_to_str_wrong("hello".encode("は"))
Traceback (most recent call last):
  File "<python-input-31>", line 1, in <module>
    decode_utf8_bytes_to_str_wrong("hello".encode("は"))
                                   ~~~~~~~~~~~~~~^^^^^^
LookupError: unknown encoding: は
```
は 是多字节组合成的

6. 给出一个无法解码为任何 Unicode 字符的双字节序列。

```
>>> wrong_uni = b"\xb3\xa9"
>>> wrong_uni.decode("utf-8")
Traceback (most recent call last):
  File "<python-input-38>", line 1, in <module>
    wrong_uni.decode("utf-8")
    ~~~~~~~~~~~~~~~~^^^^^^^^^
UnicodeDecodeError: 'utf-8' codec can't decode byte 0xb3 in position 0: invalid start byte
```
