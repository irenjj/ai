# Local submission and browser monitoring

Run these commands from `assignment5-alignment` on cloud-computer-1250.
Training executes on the dedicated remote A30 node, not on the local GPU.
The defaults target the existing CS336 PVC and upload pod in mining-infra-vlm.
The launcher never suspends, deletes, or scales other jobs/services; occupied
GPUs cause the new Job to queue. An output snapshot is isolated per run name.

## Submit a short eight-GPU run

```bash
uv run python scripts/remote_sft.py submit \
  --run-name cs336-sft-smoke-01 --watch -- \
  --max-steps 50 \
  --max-train-documents 256 \
  --max-val-documents 32 \
  --eval-steps 10
```

The launcher uploads only the two source files, verifies their SHA256 checksums,
creates an eight-GPU Kubernetes Job, and mirrors
logs/events locally every 10 seconds. Each run installs TensorBoard/matplotlib
in its own venv using the pinned image's torch/transformers. The default training
parameters are sequence length 512, BF16, microbatch 1, accumulation 4, effective
batch 32. The pinned image lacks the `flash_attn` package, so this launcher uses
SDPA explicitly. This differs from the handout's FlashAttention-2 configuration.

For manifest validation without uploads or Job creation, add `--dry-run` before
`--`. The optional arguments after `--` go to the training script. Model and
dataset/output paths are controlled by the launcher.

## Shared dataset cache

Train/test data lives once on the PVC under
`/models/cs336-datasets/safety_augmented_ultrachat_200k_single_turn/stanford-2024-v1/`.
Every submission verifies the cached size and SHA256 and reuses it. It does not
require local copies of the dataset. If absent, a small helper reuses a verified
copy from an older run, or downloads directly from Stanford on the remote node.
Downloads support partial-file resume, retries, locking, and atomic publication.
The cache is the public Stanford 2024 release; byte identity with the 2026 Modal
release is not established. Existing submitted jobs keep their original paths.

Prepare the cache independently (also done automatically during submission):

```bash
uv run python scripts/remote_sft.py prepare-data
```

## Open TensorBoard

In another terminal:

```bash
uv run --with tensorboard python scripts/remote_sft.py dashboard --port 6006
```

If `cs336-sft-tensorboard.service` is already running, use that instance rather
than launching a duplicate. Its status is available with
`systemctl --user status cs336-sft-tensorboard.service`.

In VS Code Remote SSH, use **Ports -> Forward a Port -> 6006**, then open the
forwarded localhost URL in the laptop browser. The server binds to 127.0.0.1;
it is not published on the internet. TensorBoard displays `loss/train`,
`loss/validation`, `optimizer/learning_rate`, and `optimizer/grad_norm` against
optimizer update steps. `local-toy-demo` is a random tiny-model verification
run, not a trained Llama 8B result.

## Detach, reconnect, inspect

Ctrl-C on the watcher detaches locally; the remote training Job continues.
TensorBoard remains available, but new data requires an active watcher or fetch.

```bash
uv run python scripts/remote_sft.py status --run-name cs336-sft-smoke-01
uv run python scripts/remote_sft.py watch --run-name cs336-sft-smoke-01
uv run python scripts/remote_sft.py fetch --run-name cs336-sft-smoke-01
```

Local diagnostics: `outputs/safety/remote_sft/<run-name>/`, including
`status.json`, `environment.log`, `train.log`, `run.log`, `metrics.jsonl`, and
`tensorboard/`. The launcher mirrors diagnostics only; full model weights stay
on the PVC at `/models/cs336-sft-runs/<run-name>/output/final/`.

For a full epoch after validating the short run, submit a new run name without
the max-step or document limits. Every run starts from the supplied base model.
Exports do not include optimizer state and do not support exact training resume.

## Validation boundary

The manifest has been accepted by server-side dry-run. TensorBoard scalars and
browser HTTP endpoints have been checked using a local tiny model. Eight-GPU
FSDP training has not yet been executed; the dedicated node currently has a
separate judge Job reserving its GPUs.
