from __future__ import annotations

import time
from collections import Counter, defaultdict
from pathlib import Path

from .constants import CANONICAL_TOOL_IDS
from .io import read_jsonl, write_jsonl


def _last_tool(state: dict) -> str:
    messages = state.get("messages", []) if isinstance(state, dict) else []
    for message in reversed(messages):
        if message.get("role") == "assistant":
            calls = message.get("tool_calls") or []
            if len(calls) == 1:
                function = calls[0].get("function", {})
                name = function.get("name")
                if isinstance(name, str):
                    return name
    return "<START>"


def _distribution(counts: Counter, options: list[str], alpha: float) -> list[float]:
    total = sum(counts[option] for option in options) + alpha * len(options)
    if total == 0:
        return [1.0 / len(options)] * len(options)
    return [(counts[option] + alpha) / total for option in options]


def _gold_maps(data_dir: Path) -> tuple[dict[str, str], dict[str, str]]:
    gate, tool = {}, {}
    for row in read_jsonl(data_dir / "gold.jsonl"):
        if row["split"] != "train":
            continue
        gate[row["gate_id"]] = row["label_action_type"]
        if row["tool_id"]:
            tool[row["tool_id"]] = row["label_tool"]
    return gate, tool


def run_count_baseline(
    data_dir: Path,
    split: str,
    output: Path,
    method: str,
    alpha: float = 1.0,
) -> int:
    if method not in {"frequency", "markov"}:
        raise ValueError("method must be frequency or markov")
    gate_gold, tool_gold = _gold_maps(data_dir)
    train_rows = list(read_jsonl(data_dir / "train_gate.jsonl")) + list(read_jsonl(data_dir / "train_tool.jsonl"))
    train_by_id = {row["id"]: row for row in train_rows}
    global_counts = {"gate": Counter(gate_gold.values()), "tool": Counter(tool_gold.values())}
    transitions: dict[str, dict[str, Counter]] = {
        "gate": defaultdict(Counter),
        "tool": defaultdict(Counter),
    }
    for identifier, label in gate_gold.items():
        transitions["gate"][_last_tool(train_by_id[identifier]["state"])][label] += 1
    for identifier, label in tool_gold.items():
        transitions["tool"][_last_tool(train_by_id[identifier]["state"])][label] += 1

    inputs = list(read_jsonl(data_dir / f"{split}_gate.jsonl")) + list(read_jsonl(data_dir / f"{split}_tool_all.jsonl"))
    outputs = []
    for row in inputs:
        started = time.perf_counter()
        kind = "gate" if row["id"].endswith("-gate") else "tool"
        option_ids = [option["id"] for option in row["options"]]
        counts = global_counts[kind]
        context = None
        if method == "markov":
            context = _last_tool(row["state"])
            if transitions[kind][context]:
                counts = transitions[kind][context]
        probabilities = _distribution(counts, option_ids, alpha)
        best = max(range(len(option_ids)), key=lambda index: (probabilities[index], -index))
        outputs.append(
            {
                "id": row["id"],
                "option_ids": option_ids,
                "probabilities": probabilities,
                "prediction": option_ids[best],
                "status": "ok",
                "method": method,
                "context": context,
                "total_seconds": time.perf_counter() - started,
                "forward_seconds": 0.0,
                "input_tokens": None,
            }
        )
    return write_jsonl(output, outputs)


def validate_canonical_tool_options(rows: list[dict]) -> None:
    for row in rows:
        if row["id"].endswith("-tool"):
            ids = [option["id"] for option in row["options"]]
            if ids != CANONICAL_TOOL_IDS:
                raise ValueError(f"{row['id']}: noncanonical tool option order")

