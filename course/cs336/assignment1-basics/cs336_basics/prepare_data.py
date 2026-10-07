"""Train one TinyStories tokenizer and encode both splits into mapped .npy files."""

import argparse
import hashlib
import itertools
import json
import multiprocessing
import os
import shutil
import tempfile
import time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np

from .bpe import train_bpe
from .pretokenization_example import find_chunk_boundaries
from .tokenizer import Tokenizer

SPECIAL_TOKENS = ["<|endoftext|>"]
_tokenizer = None


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def log(event: str, **values) -> None:
    print(json.dumps({"event": event, **values}), flush=True)


def init_worker(directory: str) -> None:
    global _tokenizer
    directory = Path(directory)
    _tokenizer = Tokenizer.from_files(directory / "vocab.json", directory / "merges.json",
                                     SPECIAL_TOKENS)


def encode_shard(task):
    source_path, start, end, output = task
    started = time.perf_counter()
    count = 0
    with open(source_path, "rb") as source, open(output, "wb") as target:
        source.seek(start)

        def lines():
            while source.tell() < end:
                # Byte ranges end at document markers, never inside UTF-8 characters.
                yield source.readline(end - source.tell()).decode("utf-8")

        ids = _tokenizer.encode_iterable(lines())
        while True:
            batch = list(itertools.islice(ids, 131072))
            if not batch:
                break
            np.asarray(batch, dtype="<u2").tofile(target)
            count += len(batch)
    return {"path": output, "tokens": count, "seconds": time.perf_counter() - started}


def encode_dataset(source: Path, output: Path, directory: Path, workers: int):
    started = time.perf_counter()
    temporary = Path(tempfile.mkdtemp(prefix=output.stem + ".shards-", dir=output.parent))
    try:
        with source.open("rb") as file:
            boundaries = find_chunk_boundaries(file, workers * 2, b"<|endoftext|>")
        tasks = [(str(source), start, end, str(temporary / f"{index:04d}.bin"))
                 for index, (start, end) in enumerate(zip(boundaries[:-1], boundaries[1:]))
                 if start < end]
        with ProcessPoolExecutor(max_workers=workers,
                                 mp_context=multiprocessing.get_context("spawn"),
                                 initializer=init_worker, initargs=(str(directory),)) as pool:
            results = []
            for result in pool.map(encode_shard, tasks):
                results.append(result)
                log("encoded_shard", source=source.name, **result)
        total = sum(result["tokens"] for result in results)
        staged = output.with_suffix(".partial.npy")
        data = np.lib.format.open_memmap(staged, mode="w+", dtype="<u2", shape=(total,))
        offset = 0
        token_min, token_max = 65535, 0
        for result in results:
            shard = np.memmap(result["path"], mode="r", dtype="<u2")
            for start in range(0, len(shard), 1_000_000):
                chunk = shard[start:start + 1_000_000]
                data[offset:offset + len(chunk)] = chunk
                offset += len(chunk)
                token_min = min(token_min, int(chunk.min()))
                token_max = max(token_max, int(chunk.max()))
            del shard
        data.flush()
        del data
        os.replace(staged, output)
        result = {"source": str(source), "output": str(output), "tokens": total,
                  "dtype": "uint16", "min_id": token_min, "max_id": token_max,
                  "seconds": time.perf_counter() - started, "sha256": sha256(output)}
        log("encoded_dataset", **result)
        return result
    finally:
        shutil.rmtree(temporary)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--train-text", type=Path, required=True)
    parser.add_argument("--val-text", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--vocab-size", type=int, default=10000)
    parser.add_argument("--workers", type=int, default=8)
    args = parser.parse_args()
    if not 257 <= args.vocab_size <= 65536 or args.workers < 1:
        parser.error("require 257 <= vocab-size <= 65536 and workers >= 1")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = args.output_dir / "manifest.json"
    if manifest_path.exists():
        parser.error("Output manifest already exists; choose another directory")
    inputs = {split: {"path": str(path.resolve()), "bytes": path.stat().st_size,
                       "sha256": sha256(path)}
              for split, path in (("train", args.train_text), ("validation", args.val_text))}
    log("input_verified", inputs=inputs)
    tokenizer_dir = args.output_dir / "tokenizer"
    started = time.perf_counter()
    if tokenizer_dir.exists():
        tokenizer = Tokenizer.from_files(tokenizer_dir / "vocab.json",
                                         tokenizer_dir / "merges.json", SPECIAL_TOKENS)
        metadata = json.loads((tokenizer_dir / "training.json").read_text())
        if metadata["source_sha256"] != inputs["train"]["sha256"]:
            raise ValueError("Cached tokenizer was trained on different data")
    else:
        def progress(vocab_size, pretokens):
            log("bpe_progress", vocab_size=vocab_size, unique_pretokens=pretokens,
                seconds=time.perf_counter() - started)

        vocab, merges = train_bpe(str(args.train_text), args.vocab_size, SPECIAL_TOKENS,
                                  progress=progress)
        tokenizer = Tokenizer(vocab, merges, SPECIAL_TOKENS)
        longest = max(vocab.values(), key=len)
        metadata = {"source_sha256": inputs["train"]["sha256"],
                    "seconds": time.perf_counter() - started,
                    "longest_token_hex": longest.hex(), "longest_token_bytes": len(longest)}
        staged_dir = args.output_dir / "tokenizer.partial"
        tokenizer.save(staged_dir)
        (staged_dir / "training.json").write_text(json.dumps(metadata, indent=2))
        staged_dir.rename(tokenizer_dir)
    if len(tokenizer.vocab) != args.vocab_size:
        raise ValueError("Tokenizer vocabulary size differs from requested size")
    log("tokenizer_ready", vocab_size=len(tokenizer.vocab), merges=len(tokenizer.merges),
        **metadata)
    splits = {}
    for split, source in (("train", args.train_text), ("validation", args.val_text)):
        splits[split] = encode_dataset(source, args.output_dir / f"{split}.npy",
                                       tokenizer_dir, args.workers)
        if sha256(source) != inputs[split]["sha256"]:
            raise ValueError("Source changed while encoding")
        if splits[split]["max_id"] >= len(tokenizer.vocab):
            raise ValueError("Encoded token ID exceeds vocabulary")
    manifest = {"inputs": inputs, "vocab_size": len(tokenizer.vocab),
                "special_tokens": SPECIAL_TOKENS,
                "eos_token_id": tokenizer.token_ids[b"<|endoftext|>"],
                "tokenizer_training": metadata, "splits": splits}
    manifest_path.write_text(json.dumps(manifest, indent=2))
    log("finished", manifest=str(manifest_path))


if __name__ == "__main__":
    main()
