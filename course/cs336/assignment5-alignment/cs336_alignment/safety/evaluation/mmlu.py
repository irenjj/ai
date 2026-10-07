import argparse
import csv
import json
import re
import time
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[3]


def parse_mmlu_response(text: str) -> str | None:
    match = re.match(r"The correct answer is\s+([ABCD])\b", text.lstrip())
    return match.group(1) if match else None


def load_mmlu(data_dir: Path) -> list[dict]:
    examples = []

    for path in sorted(data_dir.glob("*_test.csv")):
        subject = path.stem.removesuffix("_test")

        with path.open(encoding="utf-8", newline="") as f:
            for row_number, row in enumerate(csv.reader(f), start=1):
                if len(row) != 6 or row[5] not in "ABCD" or len(row[5]) != 1:
                    raise ValueError(
                        f"Invalid MMLU row: {path}:{row_number}"
                    )

                examples.append({
                    "id": f"{subject}:{row_number}",
                    "subject": subject,
                    "question": row[0],
                    "options": row[1:5],
                    "answer": row[5],
                })

    if not examples:
        raise ValueError(f"No MMLU test examples found in {data_dir}")

    return examples


def format_prompt(
    example: dict,
    task_template: str,
    system_template: str,
) -> str:
    instruction = task_template.format(
        subject=example["subject"].replace("_", " "),
        question=example["question"],
        options=example["options"],
    )
    return system_template.format(instruction=instruction)


def main():
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
        "--data-dir",
        type=Path,
        default=REPO_ROOT / "data/mmlu/test",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=REPO_ROOT / "outputs/safety/baseline/mmlu",
    )
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--max-new-tokens", type=int, default=128)
    parser.add_argument("--limit", type=int, default=None)
    args = parser.parse_args()

    if args.batch_size <= 0 or args.max_new_tokens <= 0:
        parser.error("batch-size and max-new-tokens must be positive")
    if args.limit is not None and args.limit <= 0:
        parser.error("limit must be positive")
    if not torch.cuda.is_available():
        raise RuntimeError("This script requires a CUDA GPU.")

    examples = load_mmlu(args.data_dir)
    if args.limit is not None:
        examples = examples[:args.limit]

    prompt_dir = REPO_ROOT / "cs336_alignment/prompts_safety"
    task_template = (prompt_dir / "mmlu_zero_shot.prompt").read_text(
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

    correct_count = 0
    parse_failures = 0
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
                prediction = parse_mmlu_response(output)
                correct = prediction == example["answer"]

                correct_count += int(correct)
                parse_failures += int(prediction is None)

                record = {
                    **example,
                    "prompt": prompt,
                    "raw_output": raw_output,
                    "output": output,
                    "prediction": prediction,
                    "score": int(correct),
                }
                f.write(json.dumps(record, ensure_ascii=False) + "\n")

            f.flush()
            processed = start + len(batch)
            print(
                f"{processed}/{len(examples)} "
                f"accuracy={correct_count / processed:.4f} "
                f"parse_failures={parse_failures}",
                flush=True,
            )

    metrics = {
        "model": args.model,
        "data_dir": str(args.data_dir),
        "num_examples": len(examples),
        "correct": correct_count,
        "accuracy": correct_count / len(examples),
        "parse_failures": parse_failures,
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
    metrics_path.write_text(
        json.dumps(metrics, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    print(json.dumps(metrics, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
