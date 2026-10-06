import re

def parse_mmlu_response(text: str) -> str | None:
    match  = re.match(r"The correct answer is\s+([ABCD])\b", text.lstrip())
    return match.group(1) if match else None
