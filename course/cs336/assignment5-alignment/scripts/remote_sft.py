"""Submit isolated SFT runs to the CS336 A30 node and mirror TensorBoard events.

No shared inference or judge jobs are stopped. Ctrl-C stops local watching only.
Run from any directory; the repository root is inferred from this script.
"""

import argparse
import hashlib
import io
import json
import os
import re
import shutil
import subprocess
import sys
import tarfile
import tempfile
import time
from datetime import datetime
from pathlib import Path, PurePosixPath
from zoneinfo import ZoneInfo

from prepare_sft_data import DEFAULT_CACHE, FILES, SOURCE

ROOT = Path(__file__).resolve().parents[1]
LOCAL_RUNS = ROOT / "outputs/safety/remote_sft"
IMAGE = "reg.deeproute.ai/deeproute-public/vllm-openai@sha256:c2f3b1b964e47809b722b5e75b61b1e7b39a50f70388cf2bf2418f16a9f31da2"
DATA = "data/safety_augmented_ultrachat_200k_single_turn"
REMOTE_BASE = "/models/cs336-sft-runs"


class Kube:
    def __init__(self, args):
        binary = shutil.which("kubectl") or str(Path.home() / ".local/bin/kubectl")
        self.command = [binary, "--kubeconfig", str(args.kubeconfig), "--server", args.server,
                        "--namespace", args.namespace]
        self.env = {key: value for key, value in os.environ.items()
                    if key.lower() not in ("http_proxy", "https_proxy", "all_proxy")}

    def run(self, *args, input=None, stdin=None, timeout=120, check=True):
        result = subprocess.run(self.command + list(args), input=input, stdin=stdin,
                                capture_output=True, env=self.env, timeout=timeout)
        if check and result.returncode:
            raise RuntimeError(result.stderr.decode(errors="replace"))
        return result

    def json(self, *args):
        return json.loads(self.run(*args, "-o", "json").stdout)

    def stream(self, *args, input):
        with subprocess.Popen(self.command + list(args), stdin=subprocess.PIPE,
                              stdout=subprocess.PIPE, stderr=subprocess.STDOUT, env=self.env) as process:
            process.stdin.write(input)
            process.stdin.close()
            for line in process.stdout:
                print(line.decode(errors="replace").rstrip(), flush=True)
            if process.wait():
                raise RuntimeError("Remote dataset preparation failed; no training Job was submitted")


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--kubeconfig", type=Path, default=Path.home() / ".kube/config")
    parser.add_argument("--server", default="https://10.3.7.173:6443")
    parser.add_argument("--namespace", default="mining-infra-vlm")
    sub = parser.add_subparsers(dest="action", required=True)
    submit = sub.add_parser("submit", help="Upload a snapshot and create a training Job")
    submit.add_argument("--run-name", default="cs336-sft-" + datetime.now(ZoneInfo("Asia/Shanghai")).strftime("%Y%m%d-%H%M%S"))
    submit.add_argument("--node", default="10-3-7-195.nvidia.gpu.a30")
    submit.add_argument("--pvc", default="cs336-llama31-8b-models")
    submit.add_argument("--upload-pod", default="cs336-llama31-model-upload")
    submit.add_argument("--image", default=IMAGE)
    submit.add_argument("--model-path", default="/models/Meta-Llama-3.1-8B")
    submit.add_argument("--gpus", type=int, choices=[1, 2, 4, 8], default=8)
    submit.add_argument("--dry-run", action="store_true", help="Save/validate manifest without upload or Job creation")
    submit.add_argument("--watch", action="store_true", help="Mirror logs and events until the Job finishes")
    prepare = sub.add_parser("prepare-data", help="Download or reuse verified data on the remote PVC once")
    prepare.add_argument("--upload-pod", default="cs336-llama31-model-upload")
    prepare.add_argument("--pvc", default="cs336-llama31-8b-models")
    prepare.add_argument("--node", default="10-3-7-195.nvidia.gpu.a30")
    for name in ("watch", "status", "fetch"):
        command = sub.add_parser(name)
        command.add_argument("--run-name", required=True)
    dashboard = sub.add_parser("dashboard", help="Serve mirrored TensorBoard events on localhost")
    dashboard.add_argument("--port", type=int, default=6006)
    # Everything following -- is passed directly to the trainer as an argv list.
    argv = sys.argv[1:]
    trainer_args = []
    if "--" in argv:
        boundary = argv.index("--")
        argv, trainer_args = argv[:boundary], argv[boundary + 1:]
    args = parser.parse_args(argv)
    if trainer_args and args.action != "submit":
        parser.error("Trainer arguments are only accepted by submit")
    if getattr(args, "run_name", None) and not re.fullmatch(r"[a-z0-9](?:[a-z0-9-]{0,48}[a-z0-9])?", args.run_name):
        parser.error("run-name must contain 1-50 lowercase letters, digits, or hyphens")
    protected = {"--model-path", "--train-data-path", "--val-data-path", "--output-dir", "--tensorboard"}
    if any(value.split("=", 1)[0] in protected for value in trainer_args):
        parser.error("The launcher manages model/data/output paths and TensorBoard")
    args.trainer_args = trainer_args
    return args


def build_job(args):
    run_root = f"{REMOTE_BASE}/{args.run_name}"
    workspace = f"{run_root}/workspace"
    # Installation is isolated to this run. torch/transformers come from the pinned image.
    setup = (
        "import subprocess; "
        "subprocess.run(['uv','venv','--system-site-packages','--python','/usr/bin/python3','.venv'],check=True); "
        "subprocess.run(['uv','pip','install','--python','.venv/bin/python','tensorboard','matplotlib'],check=True)"
    )
    trainer = ["--model-path", args.model_path, "--train-data-path", f"{DEFAULT_CACHE}/train.jsonl.gz",
               "--val-data-path", f"{DEFAULT_CACHE}/test.jsonl.gz", "--output-dir", f"{run_root}/output",
               "--micro-batch-size", "1", "--gradient-accumulation-steps", str(32 // args.gpus),
               "--logging-steps", "1", "--attn-implementation", "sdpa", "--tensorboard", *args.trainer_args]
    mount = {"name": "models", "mountPath": "/models"}
    return {"apiVersion": "batch/v1", "kind": "Job", "metadata": {"name": args.run_name},
        "spec": {"backoffLimit": 0, "template": {"metadata": {"labels": {"app": "cs336-sft", "run": args.run_name}},
        "spec": {"restartPolicy": "Never", "terminationGracePeriodSeconds": 120,
            "nodeSelector": {"kubernetes.io/hostname": args.node},
            "tolerations": [{"key": "nvidia.com/gpu", "operator": "Equal", "value": "a30", "effect": "NoSchedule"}],
            "imagePullSecrets": [{"name": "image-pull-secret"}],
            "initContainers": [{"name": "environment", "image": args.image, "workingDir": workspace,
                "command": ["python3", "-u", "-c", setup], "volumeMounts": [mount],
                "resources": {"requests": {"cpu": "1", "memory": "2Gi"}, "limits": {"cpu": "4", "memory": "4Gi"}}}],
            "containers": [{"name": "train", "image": args.image, "workingDir": workspace,
                "command": [f"{workspace}/.venv/bin/python", "-u", "-m", "torch.distributed.run", "--standalone",
                            f"--nproc_per_node={args.gpus}", "--module", "cs336_alignment.training.sft"], "args": trainer,
                "env": [{"name": name, "value": value} for name, value in {
                    "PYTHONUNBUFFERED": "1", "OMP_NUM_THREADS": "4", "TOKENIZERS_PARALLELISM": "false",
                    "HF_HUB_OFFLINE": "1", "CUDA_DEVICE_ORDER": "PCI_BUS_ID"}.items()],
                "resources": {"requests": {"cpu": "32", "memory": "128Gi", "nvidia.com/gpu": str(args.gpus)},
                              "limits": {"cpu": "64", "memory": "256Gi", "nvidia.com/gpu": str(args.gpus)}},
                "volumeMounts": [mount, {"name": "shm", "mountPath": "/dev/shm"}]}],
            "volumes": [{"name": "models", "persistentVolumeClaim": {"claimName": args.pvc}},
                        {"name": "shm", "emptyDir": {"medium": "Memory", "sizeLimit": "8Gi"}}]}}}}


def run_dir(name):
    return LOCAL_RUNS / name


def verify_upload_pod(args, kube):
    pod = kube.json("get", "pod", args.upload_pod)
    if pod["status"].get("phase") != "Running" or pod["spec"].get("nodeName") != args.node:
        raise RuntimeError("Upload pod must be running on the selected training node")
    volumes = {v["name"]: v.get("persistentVolumeClaim", {}).get("claimName") for v in pod["spec"]["volumes"]}
    mounts = pod["spec"]["containers"][0].get("volumeMounts", [])
    if not any(m["mountPath"] == "/models" and volumes.get(m["name"]) == args.pvc for m in mounts):
        raise RuntimeError("Upload pod /models does not mount the requested PVC")


def prepare_data(args, kube):
    verify_upload_pod(args, kube)
    program = (ROOT / "scripts/prepare_sft_data.py").read_bytes()
    # Only this small helper crosses the network. The data remains on the PVC.
    kube.stream("exec", "-i", args.upload_pod, "--", "python3", "-u", "-",
                "--cache-dir", DEFAULT_CACHE, "--reuse-root", REMOTE_BASE, input=program)


def submit(args, kube):
    directory = run_dir(args.run_name)
    directory.mkdir(parents=True, exist_ok=True)
    job = build_job(args)
    encoded = json.dumps(job, indent=2).encode()
    (directory / "job.json").write_bytes(encoded)
    if args.dry_run:
        kube.run("create", "--dry-run=server", "-f", "-", input=encoded)
        print(f"Manifest validated, no upload or Job created: {directory / 'job.json'}", flush=True)
        return
    existing = kube.json("get", "jobs", "--field-selector", f"metadata.name={args.run_name}")
    if existing["items"] or (directory / "submission.json").exists():
        raise RuntimeError("Run already exists; use watch/status or choose a new run-name")
    verify_upload_pod(args, kube)
    kube.run("exec", args.upload_pod, "--", "test", "-f", args.model_path + "/config.json")
    free_bytes = int(kube.run("exec", args.upload_pod, "--", "python3", "-c",
                     "import shutil; print(shutil.disk_usage('/models').free)").stdout)
    if free_bytes < 25 * 2**30:
        raise RuntimeError("At least 25 GiB free space is required for the snapshot, data cache, and final 8B model")
    pods = kube.json("get", "pods", "--field-selector", f"spec.nodeName={args.node}")["items"]
    blockers = [p["metadata"]["name"] for p in pods if p["status"].get("phase") not in ("Succeeded", "Failed")
                and any(int(c.get("resources", {}).get("requests", {}).get("nvidia.com/gpu", "0"))
                        for c in p["spec"]["containers"])]
    if blockers:
        print(f"GPU reservations already exist: {blockers}; this Job may queue. No jobs will be stopped.", flush=True)
    prepare_data(args, kube)
    files = [ROOT / "cs336_alignment/training/sft.py", ROOT / "cs336_alignment/data/sft_dataset.py"]
    manifest = {str(path.relative_to(ROOT)): sha256(path) for path in files}
    workspace = f"{REMOTE_BASE}/{args.run_name}/workspace"
    kube.run("exec", args.upload_pod, "--", "mkdir", "-p", REMOTE_BASE)
    # Atomic mkdir prevents overwriting an earlier or concurrently submitted snapshot.
    kube.run("exec", args.upload_pod, "--", "mkdir", f"{REMOTE_BASE}/{args.run_name}")
    kube.run("exec", args.upload_pod, "--", "mkdir", workspace)
    print(f"Uploading code only ({sum(path.stat().st_size for path in files):,} bytes); using shared remote data...", flush=True)
    with tempfile.TemporaryFile() as handle:
        with tarfile.open(fileobj=handle, mode="w") as archive:
            for path in files:
                archive.add(path, arcname=str(path.relative_to(ROOT)))
        handle.seek(0)
        kube.run("exec", "-i", args.upload_pod, "--", "tar", "-xf", "-", "-C", workspace,
                 stdin=handle, timeout=1800)
    check = kube.run("exec", args.upload_pod, "--", "sha256sum", *[workspace + "/" + p for p in manifest], timeout=300)
    observed = {line.split(None, 1)[1].removeprefix(workspace + "/"): line.split()[0]
                for line in check.stdout.decode().splitlines()}
    if observed != manifest:
        raise RuntimeError("Uploaded snapshot checksum mismatch")
    metadata = {"run_name": args.run_name, "remote_output": f"{REMOTE_BASE}/{args.run_name}/output",
                "upload_pod": args.upload_pod, "source_sha256": manifest,
                "dataset": {"remote_path": DEFAULT_CACHE, "source": SOURCE, "files": FILES},
                "server": args.server, "namespace": args.namespace, "kubeconfig": str(args.kubeconfig)}
    (directory / "submission.json").write_text(json.dumps(metadata, indent=2) + "\n")
    result = kube.run("create", "-f", "-", input=encoded)
    print(result.stdout.decode().strip(), flush=True)
    print(f"Run: {args.run_name}; local logs: {directory}", flush=True)
    if args.watch:
        watch(args, kube)


def sha256(path):
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def status(args, kube):
    job = kube.json("get", "job", args.run_name)
    pods = kube.json("get", "pods", "-l", f"job-name={args.run_name}")["items"]
    conditions = job.get("status", {}).get("conditions", [])
    state = next((c["type"] for c in conditions if c["type"] in ("Complete", "Failed") and c["status"] == "True"), "Pending")
    if state == "Pending":
        if any(p["status"].get("phase") == "Running" for p in pods):
            state = "Running"
        elif any(any("running" in c.get("state", {}) for c in p["status"].get("initContainerStatuses", [])) for p in pods):
            state = "Initializing"
    details = {"state": state, "job": args.run_name, "pods": [{"name": p["metadata"]["name"],
        "phase": p["status"].get("phase"), "conditions": p["status"].get("conditions", []),
        "init": p["status"].get("initContainerStatuses", [])} for p in pods]}
    run_dir(args.run_name).mkdir(parents=True, exist_ok=True)
    (run_dir(args.run_name) / "status.json").write_text(json.dumps(details, indent=2) + "\n")
    return state


def fetch(args, kube):
    directory = run_dir(args.run_name)
    metadata = json.loads((directory / "submission.json").read_text())
    if (metadata["server"], metadata["namespace"]) != (args.server, args.namespace):
        raise ValueError("Selected cluster differs from the submitted run")
    # Copy only diagnostics/events, never multi-GB weights or dataset caches.
    program = """import io, pathlib, sys, tarfile
root = pathlib.Path(sys.argv[1])
if not root.exists():
    sys.exit(0)
with tarfile.open(fileobj=sys.stdout.buffer, mode='w|') as archive:
    for name in ['run.log', 'metrics.jsonl', 'config.json', 'summary.json', 'learning_curves.png', 'tensorboard']:
        path = root / name
        if path.exists():
            archive.add(path, arcname=name)
"""
    result = kube.run("exec", metadata["upload_pod"], "--", "python3", "-c", program, metadata["remote_output"])
    if result.stdout:
        with tarfile.open(fileobj=io.BytesIO(result.stdout)) as archive:
            for member in archive.getmembers():
                path = PurePosixPath(member.name)
                if path.is_absolute() or ".." in path.parts or not (member.isfile() or member.isdir()):
                    raise ValueError("Unexpected archive entry")
            archive.extractall(directory, filter="data")
    for container in ("environment", "train"):
        logs = kube.run("logs", f"job/{args.run_name}", "-c", container, "--tail=500", check=False, timeout=30)
        if logs.returncode == 0:
            (directory / f"{container}.log").write_bytes(logs.stdout)


def watch(args, kube):
    print("Mirroring events every 10 seconds. Ctrl-C detaches; remote training continues.", flush=True)
    previous = None
    while True:
        state = status(args, kube)
        fetch(args, kube)
        if state != previous:
            print(f"{args.run_name}: {state}", flush=True)
            previous = state
        metrics = run_dir(args.run_name) / "metrics.jsonl"
        if metrics.exists():
            lines = metrics.read_text().splitlines()
            if lines:
                print(lines[-1], flush=True)
        if state in ("Complete", "Failed"):
            if state == "Failed":
                raise RuntimeError(f"Remote Job failed. See {run_dir(args.run_name)}")
            return
        time.sleep(10)


def main():
    args = parse_args()
    if args.action == "dashboard":
        LOCAL_RUNS.mkdir(parents=True, exist_ok=True)
        print(f"TensorBoard: http://127.0.0.1:{args.port} (forward this port in VS Code when using SSH)", flush=True)
        raise SystemExit(subprocess.call([sys.executable, "-m", "tensorboard.main", "--logdir", str(LOCAL_RUNS),
                                          "--host", "127.0.0.1", "--port", str(args.port), "--reload_interval", "5"]))
    kube = Kube(args)
    if args.action == "submit":
        submit(args, kube)
    elif args.action == "prepare-data":
        prepare_data(args, kube)
    elif args.action == "status":
        print(status(args, kube))
    elif args.action == "fetch":
        fetch(args, kube)
    else:
        watch(args, kube)


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("Detached. Remote Job was not cancelled.", flush=True)
