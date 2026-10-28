from __future__ import annotations

import hashlib
import json
import random
import re
import time
from pathlib import Path

from .io import read_jsonl, write_jsonl


def _load_semif(semif_repo: Path):
    import sys

    source = str((semif_repo / "src").resolve())
    if source not in sys.path:
        sys.path.insert(0, source)
    from semif_phase1.core import direct_messages, load_causal_model, synchronize
    from semif_phase1.direct import score

    return direct_messages, load_causal_model, synchronize, score


def run_semif_direct(
    input_path: Path,
    output_path: Path,
    semif_repo: Path,
    model_name: str,
    revision: str,
    device: str,
    dtype: str,
    max_tokens: int,
    seed: int,
) -> int:
    _, load_model, _, score = _load_semif(semif_repo)
    rows = list(read_jsonl(input_path))
    if not rows:
        raise ValueError("input is empty")
    model, tokenizer, metadata = load_model(model_name, revision, device, dtype)
    score(model, tokenizer, rows[0], metadata, max_tokens)
    order = list(range(len(rows)))
    random.Random(seed).shuffle(order)
    outputs = [None] * len(rows)
    for index in order:
        try:
            result = score(model, tokenizer, rows[index], metadata, max_tokens)
            best = max(range(len(result["probabilities"])), key=result["probabilities"].__getitem__)
            result.update(
                {
                    "prediction": result["option_ids"][best],
                    "status": "ok",
                    "method": "semif_direct",
                    "run_seed": seed,
                }
            )
        except (RuntimeError, ValueError) as error:
            result = {
                "id": rows[index]["id"],
                "option_ids": [option["id"] for option in rows[index]["options"]],
                "probabilities": [1.0 / len(rows[index]["options"])] * len(rows[index]["options"]),
                "prediction": None,
                "status": "failed",
                "error_type": type(error).__name__,
                "error": str(error),
                "method": "semif_direct",
                "run_seed": seed,
            }
        outputs[index] = result
    return write_jsonl(output_path, outputs)


def _parse_letter(text: str, count: int) -> int | None:
    match = re.fullmatch(r"\s*([A-P])\s*", text)
    if not match:
        return None
    index = ord(match.group(1)) - ord("A")
    return index if index < count else None


def run_same_weight_generation(
    input_path: Path,
    output_path: Path,
    semif_repo: Path,
    model_name: str,
    revision: str,
    device: str,
    dtype: str,
    max_tokens: int,
    seed: int,
) -> int:
    direct_messages, load_model, synchronize, _ = _load_semif(semif_repo)
    import torch

    rows = list(read_jsonl(input_path))
    if not rows:
        raise ValueError("input is empty")
    model, tokenizer, metadata = load_model(model_name, revision, device, dtype)

    def predict(row: dict, record: bool) -> dict:
        started = time.perf_counter()
        prompt = tokenizer.apply_chat_template(
            direct_messages(row), tokenize=False, add_generation_prompt=True, enable_thinking=False
        )
        encode_started = time.perf_counter()
        inputs = tokenizer(prompt, return_tensors="pt", add_special_tokens=False)
        input_tokens = int(inputs["input_ids"].shape[-1])
        if input_tokens > max_tokens:
            raise ValueError(f"{input_tokens} input tokens exceed {max_tokens}")
        inputs = {key: value.to(next(model.parameters()).device) for key, value in inputs.items()}
        encode_seconds = time.perf_counter() - encode_started
        synchronize(next(model.parameters()).device)
        forward_started = time.perf_counter()
        with torch.inference_mode():
            generated = model.generate(
                **inputs,
                do_sample=False,
                max_new_tokens=1,
                use_cache=True,
                pad_token_id=tokenizer.eos_token_id,
            )
        synchronize(next(model.parameters()).device)
        forward_seconds = time.perf_counter() - forward_started
        token_ids = generated[0, input_tokens:].detach().cpu().tolist()
        text = tokenizer.decode(token_ids, skip_special_tokens=True)
        index = _parse_letter(text, len(row["options"]))
        option_ids = [option["id"] for option in row["options"]]
        probabilities = [0.0] * len(option_ids)
        if index is not None:
            probabilities[index] = 1.0
        else:
            probabilities = [1.0 / len(option_ids)] * len(option_ids)
        return {
            "id": row["id"],
            "option_ids": option_ids,
            "probabilities": probabilities,
            "prediction": option_ids[index] if index is not None else None,
            "status": "ok" if index is not None else "parse_failure",
            "method": "same_weight_generation",
            "output_text": text,
            "output_token_ids": token_ids,
            "input_tokens": input_tokens,
            "output_tokens": len(token_ids),
            "encode_seconds": encode_seconds,
            "forward_seconds": forward_seconds,
            "total_seconds": time.perf_counter() - started,
            "prompt_sha256": hashlib.sha256(prompt.encode()).hexdigest(),
            "model": metadata,
            "run_seed": seed,
        }

    predict(rows[0], False)
    order = list(range(len(rows)))
    random.Random(seed).shuffle(order)
    outputs = [None] * len(rows)
    for index in order:
        try:
            outputs[index] = predict(rows[index], True)
        except (RuntimeError, ValueError) as error:
            option_ids = [option["id"] for option in rows[index]["options"]]
            outputs[index] = {
                "id": rows[index]["id"],
                "option_ids": option_ids,
                "probabilities": [1.0 / len(option_ids)] * len(option_ids),
                "prediction": None,
                "status": "failed",
                "error_type": type(error).__name__,
                "error": str(error),
                "method": "same_weight_generation",
                "run_seed": seed,
            }
    return write_jsonl(output_path, outputs)

