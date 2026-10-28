from __future__ import annotations

import hashlib
import importlib.metadata
import json
import platform
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

from .constants import (
    CANONICAL_TOOL_IDS,
    GATE_OPTIONS,
    GATE_QUESTION,
    PREDICTION_SCHEMA_VERSION,
    READ_ONLY_SPECULATION_TOOLS,
    TOOL_QUESTION,
)
from .io import read_json, write_json, write_jsonl
from .upstream import git_sha, load_retail_tool_schemas, options_from_schemas


@dataclass(frozen=True)
class ParsedAction:
    action_type: str
    tool_name: str | None
    arguments: dict[str, Any] | None
    exclusion: str | None = None


def parse_assistant_action(message: dict) -> ParsedAction:
    if message.get("role") != "assistant":
        raise ValueError("parse_assistant_action requires an assistant message")
    calls = message.get("tool_calls") or []
    content = message.get("content")
    has_response = isinstance(content, str) and bool(content.strip())
    if calls:
        if not isinstance(calls, list):
            return ParsedAction("invalid", None, None, "tool_calls_not_list")
        if len(calls) != 1:
            return ParsedAction("invalid", None, None, "multiple_tool_calls")
        if has_response:
            return ParsedAction("invalid", None, None, "mixed_response_and_tool_call")
        function = calls[0].get("function") if isinstance(calls[0], dict) else None
        if not isinstance(function, dict) or not isinstance(function.get("name"), str):
            return ParsedAction("invalid", None, None, "malformed_tool_call")
        raw_arguments = function.get("arguments", {})
        if isinstance(raw_arguments, str):
            try:
                arguments = json.loads(raw_arguments)
            except json.JSONDecodeError:
                return ParsedAction("invalid", function["name"], None, "invalid_arguments_json")
        else:
            arguments = raw_arguments
        if not isinstance(arguments, dict):
            return ParsedAction("invalid", function["name"], None, "arguments_not_object")
        return ParsedAction("call_tool", function["name"], arguments)
    if has_response:
        return ParsedAction("respond", None, None)
    return ParsedAction("invalid", None, None, "empty_assistant_action")


def assign_split(task_id: str | int, seed: int, ratios=(0.70, 0.15, 0.15)) -> str:
    if len(ratios) != 3 or any(value < 0 for value in ratios) or abs(sum(ratios) - 1.0) > 1e-9:
        raise ValueError("Split ratios must be three nonnegative values summing to 1")
    digest = hashlib.sha256(f"{seed}:{task_id}".encode()).digest()
    value = int.from_bytes(digest[:8], "big") / 2**64
    if value < ratios[0]:
        return "train"
    if value < ratios[0] + ratios[1]:
        return "val"
    return "test"


def _actor_id(path: Path) -> str:
    name = path.stem
    return name[:-7] if name.endswith("-retail") else name


def _clean_prior_message(message: dict) -> dict:
    allowed = {"role", "content", "name", "tool_call_id", "tool_calls"}
    result = {key: value for key, value in message.items() if key in allowed}
    if result.get("role") == "assistant" and result.get("tool_calls"):
        cleaned_calls = []
        for call in result["tool_calls"]:
            cleaned_calls.append(
                {
                    key: value
                    for key, value in call.items()
                    if key in {"id", "type", "function"}
                }
            )
        result["tool_calls"] = cleaned_calls
    return result


def _sample_base(actor_id: str, trial: Any, task_id: Any, step: int) -> str:
    safe_actor = actor_id.replace(" ", "-").replace("/", "-")
    return f"retail-{safe_actor}-t{trial}-task{task_id}-s{step}"


def build_rows(
    sources: Iterable[Path],
    tool_options: list[dict],
    seed: int,
) -> tuple[dict[str, list[dict]], list[dict], dict]:
    datasets = {split: [] for split in ("train", "val", "test")}
    all_tool_inputs = {split: [] for split in datasets}
    gold_rows: list[dict] = []
    exclusions = Counter()
    action_counts = Counter()
    tool_counts = Counter()
    tasks_by_split: dict[str, set[str]] = defaultdict(set)
    runs = 0
    assistant_messages = 0

    for source in sources:
        actor_id = _actor_id(source)
        raw = read_json(source)
        if not isinstance(raw, list):
            raise ValueError(f"{source}: top level must be a list")
        for run in raw:
            runs += 1
            task_id = run.get("task_id")
            trial = run.get("trial", 0)
            trajectory = run.get("traj")
            if task_id is None or not isinstance(trajectory, list):
                exclusions["malformed_run"] += 1
                continue
            split = assign_split(task_id, seed)
            tasks_by_split[split].add(str(task_id))
            success = run.get("reward")
            for index, message in enumerate(trajectory):
                if not isinstance(message, dict) or message.get("role") != "assistant":
                    continue
                assistant_messages += 1
                action = parse_assistant_action(message)
                if action.exclusion:
                    exclusions[action.exclusion] += 1
                    continue
                if action.tool_name is not None and action.tool_name not in CANONICAL_TOOL_IDS:
                    exclusions["unknown_tool"] += 1
                    continue

                base = _sample_base(actor_id, trial, task_id, index)
                state = {
                    "messages": [_clean_prior_message(item) for item in trajectory[:index]],
                    "available_tools": tool_options,
                }
                gate_id = f"{base}-gate"
                gate_row = {
                    "id": gate_id,
                    "state": state,
                    "question": GATE_QUESTION,
                    "options": GATE_OPTIONS,
                }
                datasets[split].append(gate_row)
                action_counts[action.action_type] += 1

                tool_id = f"{base}-tool"
                universal_tool_row = {
                    "id": tool_id,
                    "state": state,
                    "question": TOOL_QUESTION,
                    "options": tool_options,
                }
                all_tool_inputs[split].append(universal_tool_row)
                if action.action_type == "call_tool":
                    datasets[split].append(universal_tool_row)
                    tool_counts[action.tool_name] += 1

                gold_rows.append(
                    {
                        "sample_id": base,
                        "gate_id": gate_id,
                        "tool_input_id": tool_id,
                        "tool_id": tool_id if action.action_type == "call_tool" else None,
                        "task_id": str(task_id),
                        "trial": trial,
                        "actor_id": actor_id,
                        "step": index,
                        "split": split,
                        "label_action_type": action.action_type,
                        "label_tool": action.tool_name,
                        "label_args": action.arguments,
                        "is_task_success": None if success is None else bool(success),
                        "eligible_for_speculation": action.tool_name in READ_ONLY_SPECULATION_TOOLS,
                    }
                )

    audit = {
        "schema_version": PREDICTION_SCHEMA_VERSION,
        "runs": runs,
        "assistant_messages": assistant_messages,
        "included_decisions": len(gold_rows),
        "excluded": dict(sorted(exclusions.items())),
        "action_distribution": dict(sorted(action_counts.items())),
        "tool_distribution": {tool: tool_counts[tool] for tool in CANONICAL_TOOL_IDS},
        "tasks_by_split": {split: len(tasks_by_split[split]) for split in datasets},
        "decisions_by_split": {
            split: sum(row["split"] == split for row in gold_rows) for split in datasets
        },
        "eligible_for_speculation": sum(row["eligible_for_speculation"] for row in gold_rows),
    }
    for split in datasets:
        datasets[split].sort(key=lambda row: row["id"])
        all_tool_inputs[split].sort(key=lambda row: row["id"])
    gold_rows.sort(key=lambda row: row["sample_id"])
    return {"scored": datasets, "all_tool": all_tool_inputs}, gold_rows, audit


def prepare_dataset(
    sources: list[Path],
    upstream_repo: Path,
    out_dir: Path,
    seed: int = 20261026,
) -> dict:
    schemas = load_retail_tool_schemas(upstream_repo)
    tool_options = options_from_schemas(schemas)
    rows, gold, audit = build_rows(sources, tool_options, seed)
    provenance = {
        "schema_version": PREDICTION_SCHEMA_VERSION,
        "seed": seed,
        "tau_bench_git_sha": git_sha(upstream_repo),
        "sources": [path.name for path in sources],
        "source_sha256": {
            path.name: hashlib.sha256(path.read_bytes()).hexdigest() for path in sources
        },
        "runtime": {
            "python": platform.python_version(),
            "packages": {
                name: importlib.metadata.version(name)
                for name in ("torch", "transformers", "numpy")
                if _package_available(name)
            },
        },
        "tool_ids": CANONICAL_TOOL_IDS,
        "tool_schemas": schemas,
    }
    out_dir.mkdir(parents=True, exist_ok=True)
    write_jsonl(out_dir / "gold.jsonl", gold)
    for split in ("train", "val", "test"):
        split_rows = rows["scored"][split]
        write_jsonl(out_dir / f"{split}_gate.jsonl", (row for row in split_rows if row["id"].endswith("-gate")))
        write_jsonl(out_dir / f"{split}_tool.jsonl", (row for row in split_rows if row["id"].endswith("-tool")))
        write_jsonl(out_dir / f"{split}_tool_all.jsonl", rows["all_tool"][split])
    manifest = {
        "schema_version": PREDICTION_SCHEMA_VERSION,
        "seed": seed,
        "task_split": {row["task_id"]: row["split"] for row in gold},
        "counts": audit["decisions_by_split"],
    }
    write_json(out_dir / "split_manifest.json", manifest)
    write_json(out_dir / "tool_schemas.json", schemas)
    write_json(out_dir / "provenance.json", provenance)
    write_json(out_dir / "data_audit.json", audit)
    return audit


def _package_available(name: str) -> bool:
    try:
        importlib.metadata.version(name)
    except importlib.metadata.PackageNotFoundError:
        return False
    return True

