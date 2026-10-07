# CS336 Spring 2025 Assignment 1: Basics

For a full description of the assignment, see the assignment handout at
[cs336_assignment1_basics.pdf](./cs336_assignment1_basics.pdf)

If you see any issues with the assignment handout or code, please feel free to
raise a GitHub issue or open a pull request with a fix.

## Setup

### Environment
We manage our environments with `uv` to ensure reproducibility, portability, and ease of use.
Install `uv` [here](https://github.com/astral-sh/uv#installation) (recommended), or run `pip install uv`/`brew install uv`.
We recommend reading a bit about managing projects in `uv` [here](https://docs.astral.sh/uv/guides/projects/#managing-dependencies) (you will not regret it!).

You can now run any code in the repo using
```sh
uv run <python_file_path>
```
and the environment will be automatically solved and activated when necessary.

### Run unit tests


```sh
uv run pytest
```

Initially, all tests should fail with `NotImplementedError`s.
To connect your implementation to the tests, complete the
functions in [./tests/adapters.py](./tests/adapters.py).

### Download data
Download the TinyStories data and a subsample of OpenWebText

``` sh
mkdir -p data
cd data

wget https://huggingface.co/datasets/roneneldan/TinyStories/resolve/main/TinyStoriesV2-GPT4-train.txt
wget https://huggingface.co/datasets/roneneldan/TinyStories/resolve/main/TinyStoriesV2-GPT4-valid.txt

wget https://huggingface.co/datasets/stanford-cs336/owt-sample/resolve/main/owt_train.txt.gz
gunzip owt_train.txt.gz
wget https://huggingface.co/datasets/stanford-cs336/owt-sample/resolve/main/owt_valid.txt.gz
gunzip owt_valid.txt.gz

cd ..
```

## 查看训练曲线

生成离线 HTML，用 Chrome 打开生成的文件：

```sh
uv run python -m cs336_basics.plot_training --logs runs/overfit-one-batch --output runs/overfit-one-batch/curves.html
```

实时查看（每 3 秒重新读取日志）：

```sh
uv run python -m cs336_basics.plot_training --logs runs --serve --port 8765
```

打开 `http://127.0.0.1:8765`。使用 VS Code Remote SSH 时，先在 Ports 面板转发
8765 端口，再在本机 Chrome 打开转发地址。服务默认只监听服务器的回环地址。

页面支持训练/验证 loss、学习率、step/耗时横轴、loss 对数刻度、多个实验及数值悬停。

训练每隔 `--log-every` 步以及最后一步还会输出 `diagnostics` JSONL 事件：
`activations` 记录各模块输出的形状、L2 范数和 RMS；`weights` 记录更新前权重；
`gradients_before_clip` / `gradients_after_clip` 记录裁剪前后的梯度。
权重和梯度同时包含全局及逐参数统计；缺失梯度列在 `missing_gradients`，
非有限范数用 `finite=false` 和空数值表示。这些统计是当前训练 batch 的快照，
不是区间平均；验证阶段不采样。采样会增加统计和设备同步开销。
HTML 页面也展示激活、权重、裁剪前后梯度曲线，可选择模块或参数、
切换 L2 / RMS 及对数刻度。旧日志没有诊断数据时会提示，无法追补历史范数。
更新脚本后，已有页面服务需要重启，再刷新浏览器。

## RMSNorm 消融实验

训练时添加 `--no-rmsnorm`，可将每个 Block 的两处 RMSNorm 和最终输出前的
RMSNorm 替换为无参数的恒等映射。默认仍使用 RMSNorm，残差连接保持不变。
从头训练并使用独立的 `--run-id` 和 `--checkpoint`，不要恢复有 RMSNorm 的
基线 checkpoint。该开关会记录在日志 config 中；生成入口会读取 config 中的
`no_rmsnorm`，旧配置缺少该字段时仍启用 RMSNorm。
每个 session 单独绘制；耗时不拼接不同启动的计时。`--logs` 可以接收多个文件、
目录或带引号的通配符，例如 `--logs 'runs/*/*.jsonl'`。
