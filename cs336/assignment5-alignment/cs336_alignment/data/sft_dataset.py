import json
import gzip
import random
import time
import torch
from collections.abc import Callable

from torch.utils.data import Dataset, DataLoader
from transformers import PreTrainedTokenizerBase
from pathlib import Path

class SFTDataset(Dataset):
    def __init__(
        self,
        tokenizer: PreTrainedTokenizerBase,
        dataset_path: str | Path,
        seq_length: int,
        shuffle: bool,
        *,
        max_documents: int | None = None,
        progress_callback: Callable[[str, int, int | None], None] | None = None,
    ):
        super().__init__()

        self.seq_length = seq_length
        dataset_path = Path(dataset_path)
        if max_documents is not None and max_documents <= 0:
            raise ValueError("max_documents must be positive")
        last_report = time.monotonic()

        def report(stage, count, total=None, force=False):
            nonlocal last_report
            now = time.monotonic()
            if progress_callback is not None and (force or now - last_report >= 5):
                progress_callback(stage, count, total)
                last_report = now

        # 1. 读取 jsonl, 兼容普通文件和 gzip 压缩文件
        # 多个问答在同一个文件里, 每行是一组回答:
        # {"prompt": "什么是机器学习？", "response": "机器学习是……"}
        # {"prompt": "翻译 hello", "response": "你好"}
        # {"prompt": "计算 1+1", "response": "2"}
        open_file = gzip.open if dataset_path.suffix == ".gz" else open
        documents = []
        report("read", 0, force=True)
        with open_file(dataset_path, "rt", encoding="utf-8") as f:
            for line in f:
                if not line.strip():
                    continue
                documents.append(json.loads(line))
                report("read", len(documents))
                if max_documents is not None and len(documents) >= max_documents:
                    break
        report("read_done", len(documents), force=True)

        # 2. 在拼接之前, 打乱整篇文档顺序
        if shuffle:
            random.shuffle(documents)

        template = """Below is an instruction that describes a task. Write a response that appropriately completes the request.

### Instruction:
{instruction}

### Response:
{response}"""

        # 3. 格式化, 编码, 并在每篇文档末尾添加 EOS
        token_ids = []
        report("tokenize", 0, len(documents), force=True)
        for index, document in enumerate(documents, 1):
            text = template.format(
                instruction=document["prompt"],
                response=document["response"],
            ).strip()

            token_ids.extend(
                tokenizer.encode(text, add_special_tokens=True)
            )
            token_ids.append(tokenizer.eos_token_id)
            report("tokenize", index, len(documents), force=index == len(documents))

        # 4. 保存连续 token, 取样时再切出出入和标签
        self.token_ids = torch.tensor(token_ids, dtype=torch.long)

        self.num_sequence = max(
            0, (len(self.token_ids) - 1) // self.seq_length
        )


    def __len__(self) -> int:
        return self.num_sequence

    def __getitem__(self, i: int) -> dict[str, torch.Tensor]:
        if i < 0:
            i += len(self)
        if i < 0 or i >= len(self):
            raise IndexError("dataset index out of range")

        start = i * self.seq_length
        end = start + self.seq_length

        return {
            "input_ids": self.token_ids[start:end],
            "labels": self.token_ids[start + 1:end+1]
        }


def iterate_batches(
    dataset: Dataset,
    batch_size: int,
    shuffle: bool,
):
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        drop_last=False,
    )
