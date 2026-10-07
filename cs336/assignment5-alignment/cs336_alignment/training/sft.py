"""Full-parameter SFT without Trainer; torchrun enables CUDA FSDP.

Dataset labels already contain the next token: never shift them again.
Checkpoints are HF model/tokenizer exports, not optimizer-resume checkpoints.
Distributed runs require a shared output directory across all ranks.
"""

import argparse
import json
import logging
import math
import os
import random
import time
from datetime import timedelta
from functools import partial
from itertools import islice
from pathlib import Path

import torch
import torch.distributed as dist
import torch.nn.functional as F
from torch.distributed.fsdp import (
    FullStateDictConfig,
    FullyShardedDataParallel as FSDP,
    ShardingStrategy,
    StateDictType,
)
from torch.distributed.fsdp.wrap import transformer_auto_wrap_policy
from torch.utils.data import DataLoader, Dataset, Sampler
from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer
from transformers import get_cosine_schedule_with_warmup
from transformers.models.llama.modeling_llama import LlamaDecoderLayer

from cs336_alignment.data.sft_dataset import SFTDataset

LOGGER = logging.getLogger("cs336.sft")
TB_WRITER = None


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-path", type=str, required=True)
    parser.add_argument("--train-data-path", type=Path, required=True)
    parser.add_argument("--val-data-path", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--seq-length", type=int, default=512)
    parser.add_argument("--num-epochs", "--num-epoches", dest="num_epochs", type=int, default=1)
    parser.add_argument("--micro-batch-size", type=int, default=1)
    parser.add_argument("--eval-batch-size", type=int, default=1)
    parser.add_argument("--gradient-accumulation-steps", type=int, default=32)
    parser.add_argument("--learning-rate", type=float, default=2e-5)
    parser.add_argument("--weight-decay", type=float, default=0.1)
    parser.add_argument("--warmup-ratio", type=float, default=0.03)
    parser.add_argument("--max-grad-norm", type=float, default=1.0)
    parser.add_argument("--logging-steps", type=int, default=10)
    parser.add_argument("--eval-steps", type=int, default=200)
    parser.add_argument("--save-steps", type=int, default=0, help="0: save only the final model")
    parser.add_argument("--max-steps", type=int, default=0, help="0: train for all epochs")
    parser.add_argument("--max-train-documents", type=int, default=None,
                        help="Debug only: read at most this many training JSONL records")
    parser.add_argument("--max-val-documents", type=int, default=None,
                        help="Debug only: read at most this many validation JSONL records")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--device", choices=["cuda", "cpu"], default="cuda")
    parser.add_argument("--dtype", choices=["bfloat16", "float32"], default="bfloat16")
    parser.add_argument("--attn-implementation", choices=["flash_attention_2", "sdpa", "eager"],
                        default="flash_attention_2")
    parser.add_argument("--gradient-checkpointing", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--tensorboard", action="store_true", help="Write live TensorBoard scalars")
    args = parser.parse_args()
    for name in ("seq_length", "num_epochs", "micro_batch_size", "eval_batch_size",
                 "gradient_accumulation_steps", "logging_steps", "eval_steps"):
        if getattr(args, name) <= 0:
            parser.error(f"{name} must be positive")
    if min(args.save_steps, args.max_steps, args.num_workers) < 0:
        parser.error("save_steps, max_steps, and num_workers must be nonnegative")
    for name in ("max_train_documents", "max_val_documents"):
        if getattr(args, name) is not None and getattr(args, name) <= 0:
            parser.error(f"{name} must be positive")
    if not 0 <= args.warmup_ratio <= 1:
        parser.error("warmup_ratio must be between 0 and 1")
    if args.learning_rate <= 0 or args.max_grad_norm <= 0 or args.weight_decay < 0:
        parser.error("invalid optimizer hyperparameters")
    if args.device == "cpu" and (args.dtype != "float32" or args.attn_implementation == "flash_attention_2"):
        parser.error("CPU smoke tests require --dtype float32 --attn-implementation sdpa (or eager)")
    return args


class MaskedDataset(Dataset):
    """Mark distributed padding rows so they never contribute to loss."""

    def __init__(self, dataset):
        self.dataset = dataset

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, index):
        row, valid = index
        return {**self.dataset[row], "valid": valid}


class EpochSampler(Sampler):
    """Equal work per rank, with masked repeats only for collective synchronization."""

    def __init__(self, size, rank, world_size, shuffle, seed):
        self.size, self.rank, self.world_size = size, rank, world_size
        self.shuffle, self.seed, self.epoch = shuffle, seed, 0

    def __len__(self):
        return math.ceil(self.size / self.world_size)

    def __iter__(self):
        generator = torch.Generator().manual_seed(self.seed + self.epoch)
        indices = torch.randperm(self.size, generator=generator).tolist() if self.shuffle else range(self.size)
        for position in range(self.rank, len(self) * self.world_size, self.world_size):
            yield indices[position % self.size], position < self.size


def barrier(world_size):
    if world_size > 1:
        dist.barrier()


def sum_across_ranks(value, world_size):
    if world_size > 1:
        dist.all_reduce(value, op=dist.ReduceOp.SUM)
    return value


def make_loader(dataset, batch_size, rank, world_size, shuffle, args):
    sampler = EpochSampler(len(dataset), rank, world_size, shuffle, args.seed)
    return DataLoader(MaskedDataset(dataset), batch_size=batch_size, sampler=sampler,
                      num_workers=args.num_workers, pin_memory=args.device == "cuda", drop_last=False)


def load_datasets(tokenizer, args, rank, world_size):
    # Tokenize once, then mmap the tensor storage. This avoids eight simultaneous
    # copies of the raw JSON documents and Python token lists on an eight-GPU node.
    cache_dir = args.output_dir / "data_cache"
    if rank == 0:
        cache_dir.mkdir()
        for name, path in (("train", args.train_data_path), ("val", args.val_data_path)):
            limit = args.max_train_documents if name == "train" else args.max_val_documents
            LOGGER.info("Preparing %s data: %s (max_documents=%s)", name, path, limit)
            started = time.perf_counter()

            def progress(stage, count, total):
                LOGGER.info("%s %s: %d%s documents", name, stage, count,
                            f"/{total}" if total is not None else "")

            dataset = SFTDataset(tokenizer, path, args.seq_length, shuffle=name == "train",
                                 max_documents=limit, progress_callback=progress)
            if not len(dataset):
                raise ValueError(f"{name} data contains no complete input/label sequence")
            torch.save(dataset, cache_dir / f"{name}.pt")
            LOGGER.info("%s ready: %d tokens, %d sequences, %.1fs", name,
                        len(dataset.token_ids), len(dataset), time.perf_counter() - started)
            del dataset
    barrier(world_size)
    # These pickles were created immediately above in a fresh output directory.
    return tuple(torch.load(cache_dir / f"{name}.pt", mmap=True, weights_only=False)
                 for name in ("train", "val"))


def load_model(args, rank, world_size, device):
    dtype = getattr(torch, args.dtype)
    config = AutoConfig.from_pretrained(args.model_path)
    if args.seq_length > config.max_position_embeddings:
        raise ValueError("seq_length exceeds the model context length")
    kwargs = {"dtype": dtype, "attn_implementation": args.attn_implementation}
    if world_size == 1 or rank == 0:
        model = AutoModelForCausalLM.from_pretrained(args.model_path, **kwargs)
    else:
        # Rank 0 supplies the checkpoint weights through sync_module_states.
        with torch.device("meta"):
            model = AutoModelForCausalLM.from_config(config, **kwargs)
    model.config.use_cache = False
    if args.gradient_checkpointing:
        model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
    if world_size == 1:
        return model.to(device)
    if config.model_type != "llama":
        raise ValueError("This FSDP wrapping policy supports Llama models")
    return FSDP(
        model,
        auto_wrap_policy=partial(transformer_auto_wrap_policy, transformer_layer_cls={LlamaDecoderLayer}),
        sharding_strategy=ShardingStrategy.FULL_SHARD,
        device_id=device,
        sync_module_states=True,
        param_init_fn=lambda module: module.to_empty(device=device, recurse=False),
        limit_all_gathers=True,
        use_orig_params=True,
    )


def batch_loss_sum(model, batch, device):
    input_ids = batch["input_ids"].to(device, non_blocking=True)
    labels = batch["labels"].to(device, non_blocking=True)
    valid = batch["valid"].to(device)
    logits = model(input_ids=input_ids, use_cache=False).logits
    # Labels were already shifted by SFTDataset. FP32 CE improves stability.
    losses = F.cross_entropy(logits.float().reshape(-1, logits.size(-1)),
                             labels.reshape(-1), reduction="none").reshape_as(labels)
    loss_sum = (losses * valid[:, None]).sum()
    token_count = valid.sum() * labels.size(1)
    return loss_sum, token_count


@torch.no_grad()
def evaluate(model, loader, device, world_size):
    LOGGER.info("Validation started: %d batches per rank", len(loader))
    model.eval()
    totals = torch.zeros(2, device=device, dtype=torch.float64)
    for batch in loader:
        loss_sum, count = batch_loss_sum(model, batch, device)
        totals[0] += loss_sum.double()
        totals[1] += count
    sum_across_ranks(totals, world_size)
    model.train()
    loss = (totals[0] / totals[1]).item()
    if not math.isfinite(loss):
        raise FloatingPointError("non-finite validation loss")
    return {"val_loss": loss, "val_tokens": int(totals[1].item())}


def save_model(model, tokenizer, output_dir, rank, world_size):
    LOGGER.info("Saving model: %s", output_dir)
    if world_size > 1:
        with FSDP.state_dict_type(model, StateDictType.FULL_STATE_DICT,
                                 FullStateDictConfig(offload_to_cpu=True, rank0_only=True)):
            state = model.state_dict()
        raw_model = model.module
    else:
        state, raw_model = model.state_dict(), model
    if rank == 0:
        raw_model.save_pretrained(output_dir, state_dict=state, safe_serialization=True,
                                  max_shard_size="5GB")
        tokenizer.save_pretrained(output_dir)
    del state
    barrier(world_size)
    LOGGER.info("Model saved: %s", output_dir)


def log_record(args, rank, record):
    if rank == 0:
        line = json.dumps(record, allow_nan=False)
        LOGGER.info(line)
        with (args.output_dir / "metrics.jsonl").open("a", encoding="utf-8") as handle:
            handle.write(line + "\n")
        if TB_WRITER is not None:
            tags = {"train_loss": "loss/train", "val_loss": "loss/validation",
                    "learning_rate": "optimizer/learning_rate", "grad_norm": "optimizer/grad_norm"}
            for key, tag in tags.items():
                if key in record:
                    TB_WRITER.add_scalar(tag, record[key], record["step"])
            TB_WRITER.flush()


def plot_curves(args):
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        print("matplotlib unavailable; learning-curve data is saved in metrics.jsonl", flush=True)
        return
    rows = [json.loads(line) for line in (args.output_dir / "metrics.jsonl").read_text().splitlines()]
    figure, axis = plt.subplots()
    for key, label in (("train_loss", "Training"), ("val_loss", "Validation")):
        values = [row for row in rows if key in row]
        if values:
            axis.plot([row["step"] for row in values], [row[key] for row in values], label=label)
    axis.set(xlabel="Optimizer step", ylabel="Mean token cross-entropy")
    axis.legend()
    figure.tight_layout()
    figure.savefig(args.output_dir / "learning_curves.png", dpi=160)
    plt.close(figure)


def train(args, rank, world_size, device):
    global TB_WRITER
    if rank == 0:
        if args.output_dir.exists() and any(args.output_dir.iterdir()):
            raise FileExistsError(f"Use a new, empty output directory: {args.output_dir}")
        args.output_dir.mkdir(parents=True, exist_ok=True)
        LOGGER.setLevel(logging.INFO)
        LOGGER.propagate = False
        for handler in (logging.StreamHandler(), logging.FileHandler(args.output_dir / "run.log")):
            handler.setFormatter(logging.Formatter("%(asctime)s %(levelname)s %(message)s"))
            LOGGER.addHandler(handler)
        LOGGER.info("SFT starting: model=%s device=%s world_size=%d", args.model_path, device, world_size)
        if args.tensorboard:
            from torch.utils.tensorboard import SummaryWriter
            TB_WRITER = SummaryWriter(str(args.output_dir / "tensorboard"), flush_secs=5)
            LOGGER.info("TensorBoard events: %s", args.output_dir / "tensorboard")
    barrier(world_size)
    random.seed(args.seed)
    torch.manual_seed(args.seed)
    # A conservative lower bound, checked before expensive full-data tokenization.
    # Optimizer states and activations need additional memory beyond this bound.
    if world_size == 1 and device.type == "cuda":
        config = AutoConfig.from_pretrained(args.model_path)
        with torch.device("meta"):
            probe = AutoModelForCausalLM.from_config(config, dtype=getattr(torch, args.dtype),
                                                     attn_implementation="eager")
        minimum_bytes = sum(p.numel() * p.element_size() * 2 for p in probe.parameters())
        del probe
        free_bytes, total_bytes = torch.cuda.mem_get_info(device)
        LOGGER.info("GPU %s: %.2f GiB free / %.2f GiB total; weights + gradients alone need %.2f GiB",
                    torch.cuda.get_device_name(device), free_bytes / 2**30, total_bytes / 2**30,
                    minimum_bytes / 2**30)
        if minimum_bytes > free_bytes:
            raise RuntimeError("Single-GPU full-parameter training cannot fit weights plus gradients; "
                               "use a smaller model for local smoke tests or multi-GPU FSDP for 8B")
    LOGGER.info("Loading tokenizer")
    tokenizer = AutoTokenizer.from_pretrained(args.model_path)
    train_data, val_data = load_datasets(tokenizer, args, rank, world_size)
    train_loader = make_loader(train_data, args.micro_batch_size, rank, world_size, True, args)
    val_loader = make_loader(val_data, args.eval_batch_size, rank, world_size, False, args)
    updates_per_epoch = math.ceil(len(train_loader) / args.gradient_accumulation_steps)
    total_updates = updates_per_epoch * args.num_epochs
    if args.max_steps:
        total_updates = min(total_updates, args.max_steps)
    settings = {**vars(args), "world_size": world_size, "total_optimizer_steps": total_updates,
                "effective_batch_size": args.micro_batch_size * args.gradient_accumulation_steps * world_size,
                "train_sequences": len(train_data), "val_sequences": len(val_data),
                "optimizer": "AdamW", "betas": [0.9, 0.999], "eps": 1e-8,
                "scheduler": "cosine", "checkpoints_include_optimizer_state": False}
    if rank == 0:
        (args.output_dir / "config.json").write_text(json.dumps(settings, default=str, indent=2) + "\n")
        LOGGER.info("Training configuration: %s", json.dumps(settings, default=str))
    LOGGER.info("Loading model weights")
    model = load_model(args, rank, world_size, device)
    LOGGER.info("Model ready; creating optimizer")
    model.train()
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay)
    scheduler = get_cosine_schedule_with_warmup(
        optimizer, num_warmup_steps=math.ceil(args.warmup_ratio * total_updates),
        num_training_steps=total_updates,
    )
    step, last_eval_step = 0, -1
    window_loss, window_tokens = 0.0, 0
    started = time.perf_counter()
    LOGGER.info("Training started: %d optimizer steps planned", total_updates)
    for epoch in range(args.num_epochs):
        train_loader.sampler.epoch = epoch
        batches = iter(train_loader)
        while group := list(islice(batches, args.gradient_accumulation_steps)):
            # Count actual valid tokens, including the partial final accumulation group.
            local_tokens = sum(int(batch["valid"].sum()) * args.seq_length for batch in group)
            global_tokens = sum_across_ranks(torch.tensor(local_tokens, device=device), world_size)
            optimizer.zero_grad(set_to_none=True)
            loss_total = torch.zeros((), device=device, dtype=torch.float64)
            for batch in group:
                loss_sum, _ = batch_loss_sum(model, batch, device)
                # FSDP averages gradients across ranks. Undo that factor for a
                # global token-weighted average; communicate each microbatch to
                # keep gradients sharded rather than using memory-heavy no_sync.
                (loss_sum * world_size / global_tokens).backward()
                loss_total += loss_sum.detach().double()
            loss_total = sum_across_ranks(loss_total, world_size)
            if not torch.isfinite(loss_total):
                raise FloatingPointError("non-finite training loss")
            grad_norm = (model.clip_grad_norm_(args.max_grad_norm) if world_size > 1
                         else torch.nn.utils.clip_grad_norm_(model.parameters(), args.max_grad_norm))
            if not torch.isfinite(grad_norm):
                raise FloatingPointError("non-finite gradient norm")
            used_lr = optimizer.param_groups[0]["lr"]
            optimizer.step()
            scheduler.step()
            step += 1
            window_loss += loss_total.item()
            window_tokens += global_tokens.item()
            if step == 1 or step % args.logging_steps == 0 or step == total_updates:
                log_record(args, rank, {"step": step, "epoch": epoch + 1,
                    "train_loss": window_loss / window_tokens, "learning_rate": used_lr,
                    "grad_norm": grad_norm.item(), "elapsed_seconds": time.perf_counter() - started})
                window_loss, window_tokens = 0.0, 0
            if step % args.eval_steps == 0 or step == total_updates:
                final_eval = evaluate(model, val_loader, device, world_size)
                log_record(args, rank, {"step": step, **final_eval})
                if rank == 0:
                    plot_curves(args)
                last_eval_step = step
            if args.save_steps and step % args.save_steps == 0 and step < total_updates:
                save_model(model, tokenizer, args.output_dir / f"checkpoint-{step}", rank, world_size)
            if step >= total_updates:
                break
        if step >= total_updates:
            break
    if last_eval_step != step:
        final_eval = evaluate(model, val_loader, device, world_size)
        log_record(args, rank, {"step": step, **final_eval})
    save_model(model, tokenizer, args.output_dir / "final", rank, world_size)
    if rank == 0:
        summary = {"optimizer_steps": step, **final_eval,
                   "elapsed_seconds": time.perf_counter() - started, "model_dir": str(args.output_dir / "final")}
        (args.output_dir / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
        plot_curves(args)


def main() -> None:
    args = parse_args()
    world_size, rank = int(os.environ.get("WORLD_SIZE", "1")), int(os.environ.get("RANK", "0"))
    if args.device == "cuda":
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA unavailable; CPU is only supported for small smoke tests")
        device = torch.device("cuda", int(os.environ.get("LOCAL_RANK", "0")))
        torch.cuda.set_device(device)
        if args.dtype == "bfloat16" and not torch.cuda.is_bf16_supported():
            raise RuntimeError("This device does not support bfloat16")
    else:
        device = torch.device("cpu")
    if world_size > 1:
        if device.type != "cuda":
            raise ValueError("FSDP multi-process training requires CUDA")
        dist.init_process_group("nccl", timeout=timedelta(hours=2))
    try:
        train(args, rank, world_size, device)
    except Exception:
        LOGGER.exception("SFT failed")
        raise
    finally:
        if TB_WRITER is not None:
            TB_WRITER.close()
        if dist.is_initialized():
            dist.destroy_process_group()


if __name__ == "__main__":
    main()
