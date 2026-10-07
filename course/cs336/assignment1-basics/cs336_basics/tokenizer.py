"""Byte-level BPE encoding using a fixed vocabulary and ranked merge rules."""

import json
from collections.abc import Iterable, Iterator
from functools import lru_cache
from pathlib import Path

import regex as re

from .bpe import PRETOKEN_PATTERN


class Tokenizer:
    def __init__(self, vocab: dict[int, bytes], merges: list[tuple[bytes, bytes]],
                 special_tokens: list[str] | None = None):
        self.vocab = dict(vocab)
        self.special_tokens = sorted(set(special_tokens or []), key=len, reverse=True)
        if any(not token for token in self.special_tokens):
            raise ValueError("Special tokens must not be empty")
        existing = set(self.vocab.values())
        for token in self.special_tokens:
            value = token.encode("utf-8")
            if value not in existing:
                self.vocab[max(self.vocab, default=-1) + 1] = value
                existing.add(value)
        self.token_ids = {value: key for key, value in self.vocab.items()}
        self.merge_ranks = {pair: rank for rank, pair in enumerate(merges)}
        self.merges = list(merges)
        self.pretoken_pattern = re.compile(PRETOKEN_PATTERN)
        self.special_pattern = (re.compile("|".join(map(re.escape, self.special_tokens)))
                                if self.special_tokens else None)
        self._encode_pretoken = lru_cache(maxsize=4096)(self._merge_pretoken)

    def _merge_pretoken(self, text: str) -> tuple[int, ...]:
        tokens = [bytes([value]) for value in text.encode("utf-8")]
        while len(tokens) > 1:
            ranks = [self.merge_ranks.get(pair, float("inf"))
                     for pair in zip(tokens, tokens[1:])]
            index = min(range(len(ranks)), key=ranks.__getitem__)
            if ranks[index] == float("inf"):
                break
            pair = (tokens[index], tokens[index + 1])
            merged = []
            i = 0
            while i < len(tokens):
                if i + 1 < len(tokens) and (tokens[i], tokens[i + 1]) == pair:
                    merged.append(tokens[i] + tokens[i + 1])
                    i += 2
                else:
                    merged.append(tokens[i])
                    i += 1
            tokens = merged
        return tuple(self.token_ids[token] for token in tokens)

    def _pieces(self, text: str) -> Iterator[tuple[int, int, str, bool]]:
        start = 0
        specials = self.special_pattern.finditer(text) if self.special_pattern else ()
        for match in specials:
            for pretoken in self.pretoken_pattern.finditer(text, start, match.start()):
                yield pretoken.start(), pretoken.end(), pretoken.group(), False
            yield match.start(), match.end(), match.group(), True
            start = match.end()
        for pretoken in self.pretoken_pattern.finditer(text, start):
            yield pretoken.start(), pretoken.end(), pretoken.group(), False

    def encode(self, text: str) -> list[int]:
        result = []
        for _, _, piece, special in self._pieces(text):
            if special:
                result.append(self.token_ids[piece.encode("utf-8")])
            else:
                result.extend(self._encode_pretoken(piece))
        return result

    def encode_iterable(self, iterable: Iterable[str]) -> Iterator[int]:
        """Keep an unfinished suffix so chunk boundaries do not alter tokenization."""
        buffer = ""
        special_length = max(map(len, self.special_tokens), default=0)
        for chunk in iterable:
            buffer += chunk
            pieces = list(self._pieces(buffer))
            # A later chunk may extend a word, whitespace, contraction or special token.
            safe_end = 0
            for _, end, piece, special in pieces[:-2]:
                if end > len(buffer) - special_length:
                    break
                if special:
                    yield self.token_ids[piece.encode("utf-8")]
                else:
                    yield from self._encode_pretoken(piece)
                safe_end = end
            buffer = buffer[safe_end:]
        yield from self.encode(buffer)

    def decode(self, ids: Iterable[int]) -> str:
        return b"".join(self.vocab[token_id] for token_id in ids).decode("utf-8", errors="replace")

    def save(self, directory: str | Path) -> None:
        """JSON stores bytes as hex, avoiding lossy Unicode conversions."""
        directory = Path(directory)
        directory.mkdir(parents=True, exist_ok=True)
        (directory / "vocab.json").write_text(json.dumps(
            {key: value.hex() for key, value in self.vocab.items()}), encoding="utf-8")
        (directory / "merges.json").write_text(json.dumps(
            [[left.hex(), right.hex()] for left, right in self.merges]), encoding="utf-8")
        (directory / "special_tokens.json").write_text(
            json.dumps(self.special_tokens), encoding="utf-8")

    @classmethod
    def from_files(cls, vocab_filepath: str | Path, merges_filepath: str | Path,
                   special_tokens: list[str] | None = None) -> "Tokenizer":
        vocab = {int(key): bytes.fromhex(value) for key, value in
                 json.loads(Path(vocab_filepath).read_text(encoding="utf-8")).items()}
        merges = [(bytes.fromhex(left), bytes.fromhex(right)) for left, right in
                  json.loads(Path(merges_filepath).read_text(encoding="utf-8"))]
        return cls(vocab, merges, special_tokens)
