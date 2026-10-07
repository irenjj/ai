# TinyStories 数据准备

原始数据使用本仓库 README 指定的 `TinyStoriesV2-GPT4-train.txt` 和
`TinyStoriesV2-GPT4-valid.txt`，放在 `data/official/`。训练集用于训练词表，
验证集只使用这套固定词表编码，不参与词表训练。

准备入口：

```bash
uv run python -m cs336_basics.prepare_data \
  --train-text data/official/TinyStoriesV2-GPT4-train.txt \
  --val-text data/official/TinyStoriesV2-GPT4-valid.txt \
  --output-dir data/tinystories \
  --vocab-size 10000 --workers 8
```

输出文件：

- `data/tinystories/tokenizer/vocab.json`：ID 到字节串的映射，字节串以十六进制保存。
- `data/tinystories/tokenizer/merges.json`：有序合并规则，两个字节串均以十六进制保存。
- `data/tinystories/tokenizer/special_tokens.json`：特殊词元列表。
- `data/tinystories/tokenizer/training.json`：词表训练耗时和训练文本 SHA-256。
- `data/tinystories/train.npy`、`validation.npy`：一维 `uint16` token ID 数组。
- `data/tinystories/manifest.json`：输入/输出校验值、token 数量、ID 范围及 EOS ID。

`Tokenizer.from_files` 读取上述十六进制 JSON 格式；它与测试夹具中的 GPT-2
字符映射格式不同。测试适配器直接传入解码后的词表和合并规则。

编码按文档边界分片并行执行，再按原始顺序拼接；文本中原有的
`<|endoftext|>` 会保留为单个 token，不额外插入分隔符。处理时不把全部 token
放进内存，训练入口通过 `np.load(..., mmap_mode="r")` 读取结果。

训练入口使用 `--train-data data/tinystories/train.npy`、
`--val-data data/tinystories/validation.npy` 和 `--vocab-size 10000`。
完成数据准备不会自动启动模型训练。

输出目录已存在完整 `manifest.json` 时，准备入口会拒绝覆盖。
原始数据和生成的数据都位于 Git 忽略的 `data/` 目录。

本次完整数据编码得到训练集 541,229,347 个 token、验证集 5,465,883 个
token，EOS ID 为 256。两份数组还原字节后的 SHA-256 均与原始文本一致；
GPU 采样得到 `(2, 256)` 的 `int64` 输入和目标，每个目标是输入对应位置的下一个 token。
核验结果保存在 `data/tinystories/verification.json`。
