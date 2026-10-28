from __future__ import annotations

from collections import Counter
from pathlib import Path

from .constants import CANONICAL_TOOL_IDS, GATE_OPTIONS
from .io import index_unique, read_json, read_jsonl


def validate_data_dir(data_dir: Path) -> dict:
    gold = list(read_jsonl(data_dir / "gold.jsonl"))
    gold_by_sample = index_unique(gold, "sample_id")
    manifest = read_json(data_dir / "split_manifest.json")
    errors: list[str] = []
    counts = Counter()
    task_splits: dict[str, set[str]] = {}

    for row in gold:
        task_splits.setdefault(row["task_id"], set()).add(row["split"])
        if row["label_action_type"] not in {"call_tool", "respond"}:
            errors.append(f"{row['sample_id']}: invalid action label")
        if row["label_action_type"] == "call_tool" and row["label_tool"] not in CANONICAL_TOOL_IDS:
            errors.append(f"{row['sample_id']}: invalid tool label")
        if row["label_action_type"] == "respond" and (row["label_tool"] is not None or row["tool_id"] is not None):
            errors.append(f"{row['sample_id']}: respond row carries a tool label")
        if not isinstance(row.get("tool_input_id"), str):
            errors.append(f"{row['sample_id']}: missing universal tool input id")
        counts[f"gold_{row['split']}"] += 1
    for task_id, splits in task_splits.items():
        if len(splits) != 1:
            errors.append(f"task {task_id} crosses splits: {sorted(splits)}")

    expected_order = CANONICAL_TOOL_IDS
    for split in ("train", "val", "test"):
        gate = list(read_jsonl(data_dir / f"{split}_gate.jsonl"))
        tool = list(read_jsonl(data_dir / f"{split}_tool.jsonl"))
        gate_by_id = index_unique(gate, "id")
        tool_by_id = index_unique(tool, "id")
        counts[f"gate_{split}"] = len(gate)
        counts[f"tool_{split}"] = len(tool)
        for row in gate:
            if row["options"] != GATE_OPTIONS:
                errors.append(f"{row['id']}: gate options differ from canonical order")
            if not isinstance(row.get("state", {}).get("messages"), list) or not row["state"]["messages"]:
                errors.append(f"{row['id']}: empty message history")
        for row in tool:
            ids = [option.get("id") for option in row.get("options", [])]
            if ids != expected_order or len(ids) != len(set(ids)):
                errors.append(f"{row['id']}: tool options differ from canonical 16-way order")
        for sample in (item for item in gold if item["split"] == split):
            if sample["gate_id"] not in gate_by_id:
                errors.append(f"{sample['sample_id']}: missing gate input")
            if sample["label_action_type"] == "call_tool" and sample["tool_id"] not in tool_by_id:
                errors.append(f"{sample['sample_id']}: missing tool input")

    if manifest.get("task_split") != {row["task_id"]: row["split"] for row in gold}:
        errors.append("split_manifest task mapping differs from gold")
    if errors:
        preview = "\n".join(errors[:30])
        suffix = f"\n... {len(errors) - 30} more" if len(errors) > 30 else ""
        raise ValueError(f"Dataset validation failed:\n{preview}{suffix}")
    return {"valid": True, "samples": len(gold_by_sample), "counts": dict(sorted(counts.items()))}
