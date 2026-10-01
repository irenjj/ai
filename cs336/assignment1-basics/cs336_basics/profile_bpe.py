"""Profile the main process of BPE training without changing its implementation."""

import argparse
import cProfile
from pathlib import Path
import pstats
from time import perf_counter

from cs336_basics import bpe


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("input_path", type=Path)
    parser.add_argument("--vocab-size", type=int, default=10_000)
    parser.add_argument("--processes", type=int, default=bpe.NUM_PROCESSES)
    parser.add_argument("--special-token", action="append", dest="special_tokens")
    parser.add_argument("--output", type=Path, default=Path("profiles/bpe.prof"))
    parser.add_argument("--top", type=int, default=20)
    args = parser.parse_args()
    if args.processes < 1 or args.top < 1:
        parser.error("--processes and --top must be positive")
    if not args.input_path.is_file():
        parser.error(f"Input file does not exist: {args.input_path}")

    special_tokens = args.special_tokens or ["<|endoftext|>"]
    bpe.NUM_PROCESSES = args.processes
    args.output.parent.mkdir(parents=True, exist_ok=True)
    profiler = cProfile.Profile()
    started = perf_counter()
    try:
        vocab, merges = profiler.runcall(
            bpe.train_bpe, args.input_path, args.vocab_size, special_tokens
        )
    finally:
        elapsed = perf_counter() - started
        profiler.dump_stats(str(args.output))

    print(f"Profiled training wall time: {elapsed:.3f} s (includes profiler overhead)")
    print(f"Vocabulary: {len(vocab)}; merges: {len(merges)}; processes: {args.processes}")
    print(f"Profile saved to: {args.output.resolve()}")
    print("Main process only; worker internals are not profiled.")
    stats = pstats.Stats(profiler).strip_dirs()
    print("\nCumulative time (includes called functions):")
    stats.sort_stats(pstats.SortKey.CUMULATIVE).print_stats(args.top)
    print("\nInternal time (excludes called functions):")
    stats.sort_stats(pstats.SortKey.TIME).print_stats(args.top)


if __name__ == "__main__":
    main()
