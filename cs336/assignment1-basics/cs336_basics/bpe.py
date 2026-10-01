from collections import Counter
from dataclasses import dataclass
from heapq import heapify, heappop, heappush
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


@dataclass(frozen=True, slots=True)
class _PairCandidate:
    frequency: int
    pair: tuple[bytes, bytes]

    def __lt__(self, other: "_PairCandidate") -> bool:
        # Reverse both priorities so heapq selects the largest frequency and pair.
        if self.frequency != other.frequency:
            return self.frequency > other.frequency
        return self.pair > other.pair


class BPETrainer:
    """Own the mutable state for BPE training."""

    def __init__(self, vocab_size: int, special_tokens: list[str]) -> None:
        self.vocab_size = vocab_size
        self.special_tokens = list(special_tokens)
        self.vocab: dict[int, bytes] = {}
        self.merges: list[tuple[bytes, bytes]] = []
        self.pretoken_counts: Counter[tuple[bytes, ...]] = Counter()
        # 这个pair 在语料中出现了几次
        self.pair_counts: Counter[tuple[bytes, bytes]] = Counter()
        self.pair_to_pretokens: dict[tuple[bytes, bytes], set[tuple[bytes, ...]]] = {}
        self.pair_heap: list[_PairCandidate] = []

    def build_pair_index(self) -> None:
        """Index distinct pairs by the current pretoken sequences containing them."""
        self.pair_to_pretokens.clear()
        for token_sequence in self.pretoken_counts:
            for pair in set(pairwise(token_sequence)):
                self.pair_to_pretokens.setdefault(pair, set()).add(token_sequence)

    def build_pair_heap(self) -> None:
        self.pair_heap = [
            _PairCandidate(count, pair) for pair, count in self.pair_counts.items()
        ]
        heapify(self.pair_heap)

    def pop_best_pair(self) -> tuple[bytes, bytes]:
        while self.pair_heap:
            candidate = heappop(self.pair_heap)
            if self.pair_counts.get(candidate.pair, 0) == candidate.frequency:
                return candidate.pair
        raise RuntimeError("No valid pair candidate remains in the heap")

    def merge_best_pair(self) -> None:
        best_pair = self.pop_best_pair()
        merged_token = best_pair[0] + best_pair[1]
        # Snapshot before changing the index; remove all old sequences before adding new ones.
        affected_sequences = tuple(self.pair_to_pretokens[best_pair])
        updated_pretoken_counts: Counter[tuple[bytes, ...]] = Counter()
        touched_pairs: set[tuple[bytes, bytes]] = set()

        for token_sequence in affected_sequences:
            count = self.pretoken_counts.pop(token_sequence)
            old_pairs = Counter(pairwise(token_sequence))
            for pair, occurrences in old_pairs.items():
                self.pair_counts[pair] -= occurrences * count
                touched_pairs.add(pair)
                members = self.pair_to_pretokens[pair]
                members.remove(token_sequence)
                if not members:
                    del self.pair_to_pretokens[pair]

            merged_tokens = []
            i = 0
            while i < len(token_sequence):
                if token_sequence[i:i + 2] == best_pair:
                    merged_tokens.append(merged_token)
                    i += 2
                else:
                    merged_tokens.append(token_sequence[i])
                    i += 1
            updated_pretoken_counts[tuple(merged_tokens)] += count

        for token_sequence, count in updated_pretoken_counts.items():
            self.pretoken_counts[token_sequence] += count
            for pair, occurrences in Counter(pairwise(token_sequence)).items():
                self.pair_counts[pair] += occurrences * count
                touched_pairs.add(pair)
                self.pair_to_pretokens.setdefault(pair, set()).add(token_sequence)

        for pair in touched_pairs:
            assert self.pair_counts[pair] >= 0
            if self.pair_counts[pair] == 0:
                del self.pair_counts[pair]
            else:
                heappush(self.pair_heap, _PairCandidate(self.pair_counts[pair], pair))

        # Bound stale-entry growth; rebuild from the current authoritative counts.
        if len(self.pair_heap) > max(64, 4 * len(self.pair_counts)):
            self.build_pair_heap()

        self.merges.append(best_pair)
        self.vocab[len(self.vocab)] = merged_token

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
        self.build_pair_index()
        self.build_pair_heap()

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
