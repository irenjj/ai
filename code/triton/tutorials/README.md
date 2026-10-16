# Triton tutorial environment

## 中文练习 Notebook

以下文件对应 [Triton 官方教程目录](https://triton-lang.org/main/getting-started/tutorials/index.html) 的全部 10 篇教程，来源为 2026-10-16 下载的官网 main 版本。

保留并翻译说明、公式、练习题和参考文献。原有代码单元以及 Markdown 中的独立代码块均转换为空白代码单元；不包含实现代码、运行输出、测量结果或图表。许可见 [TRITON-LICENSE.txt](TRITON-LICENSE.txt)。

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

可在 VS Code 或 Jupyter 中打开阅读并自行填写代码框。部分原教程几乎只有代码，因此对应中文笔记的说明较短。这里没有补写实现或代码注释中的解题步骤。

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
`pyproject.toml` to reproduce the environment. The notebooks contain empty
code cells for your own implementations; no runnable tutorial script is included.
