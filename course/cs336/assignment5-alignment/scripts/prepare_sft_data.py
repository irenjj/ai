"""Prepare a verified, reusable dataset cache on the training PVC (stdlib only)."""

import argparse
import fcntl
import hashlib
import json
import os
import shutil
import time
import urllib.request
from pathlib import Path

SOURCE = "https://downloads.cs.stanford.edu/nlp/data/nfliu/cs336-spring-2024/assignment5/safety_augmented_ultrachat_200k_single_turn/"
DEFAULT_CACHE = "/models/cs336-datasets/safety_augmented_ultrachat_200k_single_turn/stanford-2024-v1"
FILES = {
    "train.jsonl.gz": {"bytes": 210634586, "sha256": "98cdcc6f5e73c1ea08b8d003eb71194274c479dea7a7db74f1610a475fa86445"},
    "test.jsonl.gz": {"bytes": 23318675, "sha256": "457ee5b893f7f24c0ab268043a6e8e57b51f65a9af447fac944f3febef8148b7"},
}


def valid(path, expected):
    if not path.is_file() or path.stat().st_size != expected["bytes"]:
        return False
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest() == expected["sha256"]


def download(url, partial, expected):
    for attempt in range(3):
        if valid(partial, expected):
            return
        offset = partial.stat().st_size if partial.exists() else 0
        if offset >= expected["bytes"]:
            partial.replace(partial.with_name(partial.name + f".invalid-{time.time_ns()}"))
            offset = 0
        headers = {"Range": f"bytes={offset}-"} if offset else {}
        print(f"Downloading on remote node: {url} (resume={offset} bytes)", flush=True)
        try:
            request = urllib.request.Request(url, headers=headers)
            with urllib.request.urlopen(request, timeout=60) as response:
                if response.status == 206:
                    if not response.headers.get("Content-Range", "").startswith(f"bytes {offset}-"):
                        raise RuntimeError("Unexpected download range")
                else:
                    offset = 0
                last_report = time.monotonic()
                with partial.open("ab" if offset else "wb") as output:
                    while block := response.read(1024 * 1024):
                        output.write(block)
                        offset += len(block)
                        if offset > expected["bytes"]:
                            raise RuntimeError("Download exceeded the expected size")
                        if time.monotonic() - last_report >= 5:
                            print(f"{partial.name}: {offset}/{expected['bytes']} bytes", flush=True)
                            last_report = time.monotonic()
            if not valid(partial, expected):
                raise RuntimeError("Downloaded file failed size/SHA256 verification")
            return
        except Exception as error:
            print(f"Download attempt {attempt + 1} failed: {error}", flush=True)
            if attempt == 2:
                raise
            time.sleep(2)


def prepare(cache_dir, reuse_root):
    cache_dir.mkdir(parents=True, exist_ok=True)
    with (cache_dir / ".prepare.lock").open("a") as lock:
        print(f"Preparing shared dataset cache: {cache_dir}", flush=True)
        fcntl.flock(lock, fcntl.LOCK_EX)
        for name, expected in FILES.items():
            target = cache_dir / name
            if valid(target, expected):
                print(f"Cache hit, SHA256 verified: {name}", flush=True)
                continue
            partial = cache_dir / (name + ".part")
            candidates = reuse_root.glob(f"*/workspace/data/safety_augmented_ultrachat_200k_single_turn/{name}")
            source = next((candidate for candidate in candidates if valid(candidate, expected)), None)
            if source is not None:
                print(f"Reusing an existing remote copy: {source}", flush=True)
                shutil.copyfile(source, partial)
                if not valid(partial, expected):
                    raise RuntimeError("Existing source changed during copying")
            else:
                download(SOURCE + name, partial, expected)
            os.replace(partial, target)
            print(f"Ready, SHA256 verified: {name}", flush=True)
        manifest = {"release": "Stanford CS336 Spring 2024", "source": SOURCE, "files": FILES,
                    "note": "Byte identity with the 2026 Modal version has not been established."}
        temporary = cache_dir / "manifest.json.part"
        temporary.write_text(json.dumps(manifest, indent=2) + "\n")
        temporary.replace(cache_dir / "manifest.json")
        return manifest


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cache-dir", type=Path, default=Path(DEFAULT_CACHE))
    parser.add_argument("--reuse-root", type=Path, default=Path("/models/cs336-sft-runs"))
    args = parser.parse_args()
    prepare(args.cache_dir, args.reuse_root)
