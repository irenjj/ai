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
    text_counts: Counter[str] = Counter()
    for segment in text_segments:
        text_counts.update(re.findall(PRETOKEN_PATTERN, segment))
    return Counter({tuple(bytes([value]) for value in text.encode("utf-8")): count
                    for text, count in text_counts.items()})


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


def merge_best_pair(
    merges: list[tuple[bytes, bytes]],
    pair_counts: dict[tuple[bytes, bytes], int],
    pretoken_counts: dict[tuple[bytes, ...], int],
    vocab: dict[int, bytes],
) -> None:
    # 优先取最大频次, 并列时取最大的 pair
    best_pair, pair_frequency = max(
        pair_counts.items(),
        key=lambda kv: (kv[1], kv[0]),
    )
    merged_token = best_pair[0] + best_pair[1]
    merges.append(best_pair)

    # 在每个词中合并所有不重叠的匹配
    updated_pretoken_counts: Counter[tuple[bytes, ...]] = Counter()

    for token_sequence, count in pretoken_counts.items():
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
                pair_counts[pair] -= count

            for pair in pairwise(merged_sequence):
                pair_counts[pair] = pair_counts.get(pair, 0) + count


    # 合并不改变片段出现次数的总和
    assert sum(updated_pretoken_counts.values()) == sum(pretoken_counts.values())
    pretoken_counts.clear()
    pretoken_counts.update(updated_pretoken_counts)

    vocab[vocab.__len__()] = best_pair[0] + best_pair[1]

    for pair, count in list(pair_counts.items()):
        assert count >= 0
        if count == 0:
            del pair_counts[pair]

    # pair_counts.clear()
    # pair_counts.update(count_token_pairs(pretoken_counts))




def count_chunk(args):
    input_path, start, end, special_tokens = args
    with open(input_path, "rb") as f:
        f.seek(start)
        chunk = f.read(end - start).decode("utf-8")

    segments = split_on_special_tokens(chunk, special_tokens)
    return count_pretokens(segments)


def train_bpe(input_path: str, vocab_size: int, special_tokens: list[str], *,
              progress=None) -> tuple[dict[int, bytes], list[tuple[bytes, bytes]]]:
    vocab = init_vocab(special_tokens)
    merges: list[tuple[bytes, bytes]] = []

    with open(input_path, "rb") as f:
        if special_tokens:
            split_token = max(special_tokens, key=len)
            if not split_token:
                raise ValueError("Special tokens must not be empty")
            boundaries = find_chunk_boundaries(
                f, NUM_PROCESSES, split_token.encode("utf-8")
            )
        else:
            f.seek(0, 2)
            boundaries = [0, f.tell()]

    tasks = [
        (input_path, start, end, special_tokens)
        for start, end in zip(boundaries[:-1], boundaries[1:])
        if start < end
    ]
    pretoken_counts: Counter[tuple[bytes, ...]] = Counter()
    if len(tasks) > 1:
        with ProcessPoolExecutor(max_workers=NUM_PROCESSES) as executor:
            for chunk_counts in executor.map(count_chunk, tasks):
                pretoken_counts.update(chunk_counts)
    else:
        for task in tasks:
            pretoken_counts.update(count_chunk(task))

    pair_counts = count_token_pairs(pretoken_counts)
    if progress:
        progress(len(vocab), len(pretoken_counts))

    for _ in range(vocab.__len__(), vocab_size):
        if not pair_counts:
            break

        merge_best_pair(merges, pair_counts, pretoken_counts, vocab)
        if progress and (len(vocab) % 250 == 0 or len(vocab) == vocab_size):
            progress(len(vocab), len(pretoken_counts))

    return vocab, merges


if __name__ == "__main__":
    train_bpe("./data/test.txt", 1000, ["<|endoftext|>"])
