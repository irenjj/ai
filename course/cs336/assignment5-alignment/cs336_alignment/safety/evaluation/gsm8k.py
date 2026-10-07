import argparse
import json
import re
import time
from decimal import Decimal, InvalidOperation
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[3]


_NUMBER_PATTERN = re.compile(r"[+-]?(?:\d+(?:,\d{3})*(?:\.\d+)?|\.\d+)")


def parse_gsm8k_response(model_output: str) -> str | None:
    """Return the last numeric answer, removing thousands separators."""
    matches = list(_NUMBER_PATTERN.finditer(model_output))
    if not matches:
        return None
    return matches[-1].group().replace(",", "")


def answers_equal(prediction: str | None, answer: str) -> bool:
    if prediction is None:
        return False
    try:
        return Decimal(prediction) == Decimal(answer)
    except InvalidOperation:
        return False


def load_gsm8k(data_path: Path) -> list[dict]:
    examples = []
    with data_path.open(encoding="utf-8") as stream:
        for row_number, line in enumerate(stream, start=1):
            raw = json.loads(line)
            question, reference = raw["question"], raw["answer"]
            if not isinstance(question, str) or not isinstance(reference, str):
                raise ValueError(f"Invalid GSM8K row: {data_path}:{row_number}")
            _, separator, gold_text = reference.rpartition("####")
            answer = parse_gsm8k_response(gold_text)
            if not separator or answer is None:
                raise ValueError(f"Missing gold answer: {data_path}:{row_number}")
            examples.append({
                "id": f"gsm8k:{row_number}",
                "question": question,
                "reference_answer": reference,
                "answer": answer,
            })
    if not examples:
        raise ValueError(f"No GSM8K examples found in {data_path}")
    return examples


def format_prompt(example: dict, task_template: str, system_template: str) -> str:
    instruction = task_template.format(question=example["question"])
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
        "--data-path",
        type=Path,
        default=REPO_ROOT / "data/gsm8k/test.jsonl",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=REPO_ROOT / "outputs/safety/baseline/gsm8k",
    )
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--max-new-tokens", type=int, default=512)
    parser.add_argument("--limit", type=int, default=None)
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

    examples = load_gsm8k(args.data_path)
    if args.limit is not None:
        examples = examples[:args.limit]

    examples = examples[args.shard_id::args.num_shards]
    if not examples:
        parser.error("This shard contains no examples")

    prompt_dir = REPO_ROOT / "cs336_alignment/prompts_safety"
    task_template = (prompt_dir / "gsm8k_zero_shot.prompt").read_text(
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
                prediction = parse_gsm8k_response(output)
                correct = answers_equal(prediction, example["answer"])

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
        "data_path": str(args.data_path),
        "num_shards": args.num_shards,
        "shard_id": args.shard_id,
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
