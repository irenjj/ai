import argparse
import csv
import json
import time
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[3]


TASKS = {
    "alpaca_eval": {
        "data": "data/alpaca_eval/alpaca_eval_gpt4_turbo.json",
        "template": "alpaca_eval_zero_shot.prompt",
    },
    "simple_safety_tests": {
        "data": "data/simple_safety_tests/simple_safety_tests.csv",
        "template": "simple_safety_tests_zero_shot.prompt",
    },
}


def load_examples(task: str, data_path: Path) -> list[dict]:
    if task == "alpaca_eval":
        raw = json.loads(data_path.read_text(encoding="utf-8"))
        examples = [
            {"id": f"alpaca_eval:{i}", "instruction": item["instruction"],
             "dataset": item["dataset"]}
            for i, item in enumerate(raw, start=1)
        ]
    elif task == "simple_safety_tests":
        with data_path.open(encoding="utf-8", newline="") as stream:
            examples = [dict(row) for row in csv.DictReader(stream)]
        for item in examples:
            item["instruction"] = item["prompts_final"]
    else:
        raise ValueError(f"Unknown task: {task}")
    if not examples or any(not isinstance(e["instruction"], str) or not e["instruction"].strip() for e in examples):
        raise ValueError(f"Missing instructions in {data_path}")
    if len({e["id"] for e in examples}) != len(examples):
        raise ValueError(f"Duplicate sample IDs in {data_path}")
    return examples


def format_prompt(example: dict, task_template: str, system_template: str) -> str:
    instruction = task_template.format(instruction=example["instruction"])
    return system_template.format(instruction=instruction)


def main(task: str):
    import torch
    from transformers import (
        AutoModelForCausalLM,
        AutoTokenizer,
        GenerationConfig,
    )

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model",
        default="/home/renjj/models/Meta-Llama-3.1-8B",
    )
    parser.add_argument(
        "--data-path",
        type=Path,
        default=REPO_ROOT / TASKS[task]["data"],
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=REPO_ROOT / f"outputs/safety/baseline/{task}",
    )
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--max-new-tokens", type=int, default=1024)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--generator", default="llama-3.1-8b-base")
    parser.add_argument("--num-shards", type=int, default=1)
    parser.add_argument("--shard-id", type=int, default=0)
    args = parser.parse_args()

    if args.batch_size <= 0 or args.max_new_tokens <= 0:
        parser.error("batch-size and max-new-tokens must be positive")
    if args.limit is not None and args.limit <= 0:
        parser.error("limit must be positive")
    if args.num_shards <= 0 or not 0 <= args.shard_id < args.num_shards:
        parser.error("shard-id must be in [0, num-shards)")
    if not torch.cuda.is_available():
        raise RuntimeError("This script requires a CUDA GPU.")

    examples = load_examples(task, args.data_path)
    if args.limit is not None:
        examples = examples[:args.limit]

    examples = examples[args.shard_id::args.num_shards]
    if not examples:
        parser.error("This shard contains no examples")

    prompt_dir = REPO_ROOT / "cs336_alignment/prompts_safety"
    task_template = (prompt_dir / TASKS[task]["template"]).read_text(
        encoding="utf-8"
    )
    system_template = (
        prompt_dir / "zero_shot_system_prompt.prompt"
    ).read_text(encoding="utf-8")

    tokenizer = AutoTokenizer.from_pretrained(
        args.model,
        local_files_only=True,
        padding_side="left",
    )
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        args.model,
        local_files_only=True,
        dtype=torch.bfloat16,
        device_map="auto",
    )
    model.eval()
    input_device = model.get_input_embeddings().weight.device

    generation_config = GenerationConfig(
        do_sample=False,
        num_beams=1,
        max_new_tokens=args.max_new_tokens,
        pad_token_id=tokenizer.pad_token_id,
        eos_token_id=tokenizer.eos_token_id,
        stop_strings=["# Query:"],
    )

    def synchronize():
        for device_index in range(torch.cuda.device_count()):
            torch.cuda.synchronize(device_index)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    predictions_path = args.output_dir / "predictions.jsonl"
    metrics_path = args.output_dir / "metrics.json"

    # Avoid accidentally overwriting an earlier experiment.
    if predictions_path.exists() or metrics_path.exists():
        raise FileExistsError(
            f"Results already exist in {args.output_dir}; "
            "choose a new --output-dir."
        )

    records = []
    generation_seconds = 0.0
    evaluation_start = time.perf_counter()

    with predictions_path.open("x", encoding="utf-8") as f:
        for start in range(0, len(examples), args.batch_size):
            batch = examples[start:start + args.batch_size]
            prompts = [
                format_prompt(example, task_template, system_template)
                for example in batch
            ]

            inputs = tokenizer(
                prompts,
                padding=True,
                return_tensors="pt",
            ).to(input_device)

            synchronize()
            generation_start = time.perf_counter()

            with torch.inference_mode():
                generated_ids = model.generate(
                    **inputs,
                    generation_config=generation_config,
                    tokenizer=tokenizer,
                )

            synchronize()
            generation_seconds += (
                time.perf_counter() - generation_start
            )

            # generate() returns the prompt followed by the new tokens.
            prompt_width = inputs["input_ids"].shape[1]
            outputs = tokenizer.batch_decode(
                generated_ids[:, prompt_width:],
                skip_special_tokens=True,
            )

            for example, prompt, raw_output in zip(batch, prompts, outputs):
                output = raw_output.split("# Query:", 1)[0].strip()
                record = {
                    **example,
                    "prompt": prompt,
                    "raw_output": raw_output,
                    "output": output,
                    "generator": args.generator,
                }
                records.append(record)
                f.write(json.dumps(record, ensure_ascii=False) + "\n")

            f.flush()
            processed = start + len(batch)
            print(
                f"{processed}/{len(examples)}",
                flush=True,
            )

    metrics = {
        "model": args.model,
        "data_path": str(args.data_path),
        "num_shards": args.num_shards,
        "shard_id": args.shard_id,
        "num_examples": len(examples),
        "task": task,
        "generator": args.generator,
        "generation_seconds": generation_seconds,
        "generation_examples_per_second": (
            len(examples) / generation_seconds
        ),
        "evaluation_seconds": time.perf_counter() - evaluation_start,
        "batch_size": args.batch_size,
        "max_new_tokens": args.max_new_tokens,
        "decoding": "greedy",
        "stop_strings": ["# Query:"],
    }
    if task == "alpaca_eval":
        (args.output_dir / "model_outputs.json").write_text(
            json.dumps([{k: r[k] for k in ("instruction", "output", "generator", "dataset")}
                        for r in records], ensure_ascii=False, indent=2),
            encoding="utf-8",
        )

    metrics_path.write_text(
        json.dumps(metrics, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    print(json.dumps(metrics, ensure_ascii=False, indent=2))

