# Triton tutorial environment

## 中文练习 Notebook

以下文件对应 [Triton 官方教程目录](https://triton-lang.org/main/getting-started/tutorials/index.html) 的全部 10 篇教程，来源为 2026-10-16 下载的官网 main 版本。

保留并翻译说明、公式、练习题和参考文献。初始练习版将代码单元留空；其中 02 已按官方教程补全为可运行版本，运行输出不写入源 Notebook。许可见 [TRITON-LICENSE.txt](TRITON-LICENSE.txt)。

| 编号 | 中文 Notebook |
| --- | --- |
| 01 | [向量加法](01-vector-add.zh.ipynb) |
| 02 | [融合 Softmax](02-fused-softmax.zh.ipynb) |
| 03 | [矩阵乘法](03-matrix-multiplication.zh.ipynb) |
| 04 | [低内存开销的 Dropout](04-low-memory-dropout.zh.ipynb) |
| 05 | [层归一化](05-layer-norm.zh.ipynb) |
| 06 | [融合注意力](06-fused-attention.zh.ipynb) |
| 07 | [Libdevice 外部函数](07-extern-functions.zh.ipynb) |
| 08 | [分组矩阵乘法](08-grouped-gemm.zh.ipynb) |
| 09 | [持久化矩阵乘法](09-persistent-matmul.zh.ipynb) |
| 10 | [分块缩放矩阵乘法](10-block-scaled-matmul.zh.ipynb) |

可在 VS Code 或 Jupyter 中打开阅读并自行填写尚未完成的代码框。部分原教程几乎只有代码，因此对应中文笔记的说明较短。

02 的运行方式：执行 `uv sync --locked`，在 Notebook 中选择当前目录的 `.venv/bin/python` 内核，然后从头运行全部单元。最后一个单元会测试 98 种列数、比较调优版 Triton、文档原版 Triton、PyTorch 和 naive 四条曲线并绘图，需要等待一段时间。也可以用 `uv run jupyter execute 02-fused-softmax.zh.ipynb --timeout=600 --output=/tmp/02-fused-softmax.zh.executed.ipynb` 独立执行并将输出保存到临时目录。

02 的本机实测参数、原始数据和正确性检查见 [RTX 4080 SUPER 调优记录](benchmarks/softmax-tuning-rtx4080super/README.md)。

本地 uv 环境针对先前的 01 教程配置；这些 main 版本教程并未逐一运行验证。09、10 包含较新硬件或 API 要求，具体限制保留在各篇说明中。

## Python environment

Environment for the official [Vector Addition tutorial](https://triton-lang.org/main/getting-started/tutorials/01-vector-add.html).

Targets Linux x86_64 with an NVIDIA GPU and a CUDA 12.8-compatible driver.
Uses Python 3.12, PyTorch 2.8.0 (CUDA 12.8), and its matching Triton 3.4.0.
NumPy, pandas, and matplotlib support the tutorial's benchmark reporting.
These are pinned release versions, not a checkout of Triton's development branch.

From this directory:

```sh
uv sync --locked
```

Select `.venv/bin/python` as the editor interpreter. Commit `uv.lock` with
`pyproject.toml` to reproduce the environment. Development dependencies include
the Notebook kernel and execution client. No standalone tutorial script is included.
