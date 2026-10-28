from __future__ import annotations

from collections.abc import Callable
from pathlib import Path

from .io import read_jsonl, write_jsonl


def _blocks(messages: list[dict]) -> tuple[list[dict], list[list[dict]]]:
    system = [message for message in messages if message.get("role") == "system"]
    rest = [message for message in messages if message.get("role") != "system"]
    blocks: list[list[dict]] = []
    for message in rest:
        if message.get("role") == "user" or not blocks:
            blocks.append([message])
        else:
            blocks[-1].append(message)
    return system, blocks


def fit_row_to_budget(
    original: dict,
    encode: Callable[[dict, int], tuple[list[int], list[int], str]],
    max_tokens: int,
) -> dict:
    row = {**original, "state": {**original["state"]}}
    messages = list(row["state"]["messages"])
    system, blocks = _blocks(messages)
    omitted = 0
    while True:
        summary = []
        if omitted:
            summary = [
                {
                    "role": "system",
                    "content": f"History window: {omitted} earlier conversation block(s) omitted.",
                }
            ]
        row["state"]["messages"] = system + summary + [item for block in blocks for item in block]
        try:
            ids, _, _ = encode(row, max_tokens)
            row["window"] = {
                "max_tokens": max_tokens,
                "input_tokens": len(ids),
                "omitted_blocks": omitted,
                "strategy": "keep-system-and-most-recent-user-blocks-v1",
            }
            return row
        except ValueError as error:
            if "input tokens exceed" not in str(error):
                raise
            if not blocks:
                raise
            blocks.pop(0)
            omitted += 1


def window_dataset(
    input_path: Path,
    output_path: Path,
    rejected_path: Path,
    semif_repo: Path,
    model_name: str,
    revision: str,
    max_tokens: int,
) -> tuple[int, int]:
    import sys
    import transformers

    source = str((semif_repo / "src").resolve())
    if source not in sys.path:
        sys.path.insert(0, source)
    from semif_phase1.direct import encode_prompt

    tokenizer = transformers.AutoTokenizer.from_pretrained(
        model_name,
        revision=revision,
        trust_remote_code=False,
    )
    encode = lambda row, budget: encode_prompt(tokenizer, row, budget)
    accepted, rejected = [], []
    for original in read_jsonl(input_path):
        try:
            accepted.append(fit_row_to_budget(original, encode, max_tokens))
        except ValueError as error:
            rejected.append({"id": original["id"], "reason": str(error), "max_tokens": max_tokens})
    write_jsonl(output_path, accepted)
    write_jsonl(rejected_path, rejected)
    return len(accepted), len(rejected)

