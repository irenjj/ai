from collections import Counter
from itertools import pairwise
from concurrent.futures import ProcessPoolExecutor

if __package__:
    from .pretokenization_example import find_chunk_boundaries
else:
    from pretokenization_example import find_chunk_boundaries

import regex as re

PRETOKEN_PATTERN = r"""'(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+"""

NUM_PROCESSES = 4

def init_vocab(special_tokens: list[str]) -> dict[int, bytes]:
    vocab = {i: bytes([i]) for i in range(256)}

    for token_id, special_token in enumerate(special_tokens, start=256):
        vocab[token_id] = special_token.encode("utf-8")

    return vocab


def count_pretokens(text_segments: list[str]) -> Counter[tuple[bytes, ...]]:
    pretoken_counts: Counter[tuple[bytes, ...]] = Counter()

    for segment in text_segments:
        pretoken_texts = re.findall(PRETOKEN_PATTERN, segment)
        for pretoken_text in pretoken_texts:
            token_sequence = tuple(bytes([byte_value]) for byte_value in pretoken_text.encode("utf-8"))
            pretoken_counts[token_sequence] += 1

    return pretoken_counts


def count_token_pairs(
    pretoken_counts: dict[tuple[bytes, ...], int],
) -> Counter[tuple[bytes, bytes]]:
    pair_counts: Counter[tuple[bytes, bytes]] = Counter()

    for token_sequence, count in pretoken_counts.items():
        for pair in pairwise(token_sequence):
            pair_counts[pair] += count

    return pair_counts


def split_on_special_tokens(text: str, special_tokens: list[str]) -> list[str]:
    if not special_tokens:
        return [text]
    pattern = "|".join(
        re.escape(token)
        for token in sorted(special_tokens, key=len, reverse=True)
    )
    return re.split(pattern, text)


def count_pretokens_and_pairs(
    text: str, special_tokens: list[str]
) -> tuple[Counter[tuple[bytes, ...]], Counter[tuple[bytes, bytes]]]:
    pretoken_counts = count_pretokens(split_on_special_tokens(text, special_tokens))
    return pretoken_counts, count_token_pairs(pretoken_counts)


def count_chunk(args):
    input_path, start, end, special_tokens = args
    with open(input_path, "rb") as f:
        f.seek(start)
        chunk = f.read(end - start).decode("utf-8")

    segments = split_on_special_tokens(chunk, special_tokens)
    return count_pretokens(segments)


class BPETrainer:
    """Own the mutable state for BPE training."""

    def __init__(self, vocab_size: int, special_tokens: list[str]) -> None:
        self.vocab_size = vocab_size
        self.special_tokens = list(special_tokens)
        self.vocab: dict[int, bytes] = {}
        self.merges: list[tuple[bytes, bytes]] = []
        self.pretoken_counts: Counter[tuple[bytes, ...]] = Counter()
        self.pair_counts: Counter[tuple[bytes, bytes]] = Counter()

    def merge_best_pair(self) -> None:
        # 优先取最大频次, 并列时取最大的 pair
        best_pair, _ = max(
            self.pair_counts.items(),
            key=lambda kv: (kv[1], kv[0]),
        )
        merged_token = best_pair[0] + best_pair[1]
        self.merges.append(best_pair)

        # 在每个词中合并所有不重叠的匹配
        updated_pretoken_counts: Counter[tuple[bytes, ...]] = Counter()

        for token_sequence, count in self.pretoken_counts.items():
            merged_tokens = []
            i = 0

            while i < len(token_sequence):
                if token_sequence[i: i + 2] == best_pair:
                    merged_tokens.append(merged_token)
                    i += 2
                else:
                    merged_tokens.append(token_sequence[i])
                    i += 1

            merged_sequence = tuple(merged_tokens)
            updated_pretoken_counts[merged_sequence] += count

            if merged_sequence != token_sequence:
                for pair in pairwise(token_sequence):
                    self.pair_counts[pair] -= count

                for pair in pairwise(merged_sequence):
                    self.pair_counts[pair] = self.pair_counts.get(pair, 0) + count


        # 合并不改变片段出现次数的总和
        assert sum(updated_pretoken_counts.values()) == sum(self.pretoken_counts.values())
        self.pretoken_counts.clear()
        self.pretoken_counts.update(updated_pretoken_counts)

        self.vocab[self.vocab.__len__()] = best_pair[0] + best_pair[1]

        for pair, count in list(self.pair_counts.items()):
            assert count >= 0
            if count == 0:
                del self.pair_counts[pair]

        # self.pair_counts.clear()
        # self.pair_counts.update(count_token_pairs(self.pretoken_counts))

    def train(self, input_path: str) -> tuple[dict[int, bytes], list[tuple[bytes, bytes]]]:
        self.vocab = init_vocab(self.special_tokens)
        self.merges: list[tuple[bytes, bytes]] = []

        with open(input_path, "rb") as f:
            if self.special_tokens:
                split_token = max(self.special_tokens, key=len)
                if not split_token:
                    raise ValueError("Special tokens must not be empty")
                boundaries = find_chunk_boundaries(
                    f, NUM_PROCESSES, split_token.encode("utf-8")
                )
            else:
                f.seek(0, 2)
                boundaries = [0, f.tell()]

        tasks = [
            (input_path, start, end, self.special_tokens)
            for start, end in zip(boundaries[:-1], boundaries[1:])
            if start < end
        ]
        self.pretoken_counts: Counter[tuple[bytes, ...]] = Counter()
        if len(tasks) > 1:
            with ProcessPoolExecutor(max_workers=NUM_PROCESSES) as executor:
                for chunk_counts in executor.map(count_chunk, tasks):
                    self.pretoken_counts.update(chunk_counts)
        else:
            for task in tasks:
                self.pretoken_counts.update(count_chunk(task))

        self.pair_counts = count_token_pairs(self.pretoken_counts)

        for _ in range(self.vocab.__len__(), self.vocab_size):
            if not self.pair_counts:
                break

            self.merge_best_pair()

        return self.vocab, self.merges


def train_bpe(
    input_path: str,
    vocab_size: int,
    special_tokens: list[str],
) -> tuple[dict[int, bytes], list[tuple[bytes, bytes]]]:
    return BPETrainer(vocab_size, special_tokens).train(input_path)


if __name__ == "__main__":
    train_bpe("./data/test.txt", 1000, ["<|endoftext|>"])
