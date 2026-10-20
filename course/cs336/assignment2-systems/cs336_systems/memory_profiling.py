"""Reproducible medium-model memory measurements, one process per case."""

import argparse
import json
from pathlib import Path
import subprocess
import sys


def run_study(output_dir, lengths=(128, 512), warmup_steps=5, measurement_steps=10):
    output_dir = Path(output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    results = []
    for length in lengths:
        for precision in ("fp32", "bf16"):
            for mode in (1, 3):
                label = f"medium-L{length}-{precision}-mode{mode}"
                snapshot = output_dir / f"{label}.pickle"
                if snapshot.exists() or (output_dir / f"{label}.log").exists():
                    raise FileExistsError(f"Use a fresh output directory: {label}")
                config = dict(d_model=1024, d_ff=4096, num_layers=24, num_heads=16,
                              vocab_size=10000, context_length=length, batch_size=4,
                              mode=mode, precision=precision, warmup_steps=warmup_steps,
                              measurement_steps=measurement_steps, memory_profile=True,
                              memory_snapshot_path=str(snapshot))
                print(f"Running {label}", flush=True)
                completed = subprocess.run(
                    [sys.executable, "-m", "cs336_systems.memory_profiling", "--worker", json.dumps(config)],
                    capture_output=True, text=True,
                )
                (output_dir / f"{label}.log").write_text(completed.stdout + completed.stderr)
                if completed.returncode:
                    raise RuntimeError(f"{label} failed; inspect its log in {output_dir}")
                payload = next(line[len("RESULT_JSON:"):] for line in completed.stdout.splitlines()
                               if line.startswith("RESULT_JSON:"))
                result = dict(config=config, **json.loads(payload))
                results.append(result)
                (output_dir / "results.json").write_text(json.dumps(results, indent=2))
                print(result["status"], result.get("summary", {}).get("memory", {}), flush=True)
    return results


def _worker(config):
    import torch
    from cs336_systems.benchmark import benchmark
    torch.manual_seed(42)
    # Verify the actual model entry sees no_grad only for mode 1.
    from cs336_basics.model import BasicsTransformerLM
    original = BasicsTransformerLM.forward
    observed = []

    def checked_forward(self, *args, **kwargs):
        enabled = torch.is_grad_enabled()
        assert enabled == (config["mode"] != 1), "Unexpected grad mode"
        observed.append(enabled)
        return original(self, *args, **kwargs)

    BasicsTransformerLM.forward = checked_forward
    result = dict(device=torch.cuda.get_device_name(), torch_version=torch.__version__, seed=42)
    try:
        result.update(status="ok", summary=benchmark(**config))
    except torch.cuda.OutOfMemoryError as exc:
        result.update(status="OOM", error=str(exc))
    result["forward_grad_enabled"] = sorted(set(observed))
    print("RESULT_JSON:" + json.dumps(result))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir")
    parser.add_argument("--worker")
    args = parser.parse_args()
    if args.worker:
        _worker(json.loads(args.worker))
    elif args.output_dir:
        run_study(args.output_dir)
    else:
        parser.error("--output-dir is required")
