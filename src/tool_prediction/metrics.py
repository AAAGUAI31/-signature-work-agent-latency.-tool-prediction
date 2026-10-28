from __future__ import annotations

import json
import math
import random
import statistics
from collections import Counter, defaultdict
from pathlib import Path
from typing import Callable

from .calibration import reliability
from .constants import CANONICAL_TOOL_IDS
from .io import read_jsonl, write_json
from .predictions import rows_by_id


def canonical_arguments(value):
    if isinstance(value, dict):
        return {key: canonical_arguments(value[key]) for key in sorted(value)}
    if isinstance(value, list):
        return [canonical_arguments(item) for item in value]
    return value


def exact_call_match(predicted_tool, predicted_args, gold_tool, gold_args) -> bool:
    return predicted_tool == gold_tool and canonical_arguments(predicted_args) == canonical_arguments(gold_args)


def macro_f1(labels: list[str], predictions: list[str], classes: list[str]) -> float:
    scores = []
    for label in classes:
        tp = sum(gold == label and pred == label for gold, pred in zip(labels, predictions))
        fp = sum(gold != label and pred == label for gold, pred in zip(labels, predictions))
        fn = sum(gold == label and pred != label for gold, pred in zip(labels, predictions))
        denominator = 2 * tp + fp + fn
        scores.append(0.0 if denominator == 0 else 2 * tp / denominator)
    return sum(scores) / len(scores) if scores else 0.0


def percentile(values: list[float], probability: float) -> float | None:
    if not values:
        return None
    ordered = sorted(values)
    position = (len(ordered) - 1) * probability
    lower = math.floor(position)
    upper = math.ceil(position)
    if lower == upper:
        return ordered[lower]
    return ordered[lower] + (ordered[upper] - ordered[lower]) * (position - lower)


def clustered_ci(
    rows: list[dict],
    value: Callable[[list[dict]], float],
    samples: int,
    seed: int,
) -> list[float] | None:
    by_task: dict[str, list[dict]] = defaultdict(list)
    for row in rows:
        by_task[row["task_id"]].append(row)
    groups = list(by_task.values())
    if not groups:
        return None
    rng = random.Random(seed)
    values = []
    for _ in range(samples):
        draw = []
        for _ in groups:
            draw.extend(rng.choice(groups))
        values.append(value(draw))
    values.sort()
    return [
        values[min(len(values) - 1, int(0.025 * len(values)))],
        values[min(len(values) - 1, int(0.975 * len(values)))],
    ]


def _prediction(predictions: dict[str, dict], identifier: str | None) -> dict | None:
    if identifier is None:
        return None
    row = predictions.get(identifier)
    if row is None or row.get("status") != "ok":
        return None
    return row


def _top(row: dict | None, k: int = 1) -> list[str]:
    if row is None:
        return []
    ranked = row.get("ranked_option_ids")
    if ranked:
        return ranked[:k]
    return [row["prediction"]] if row.get("prediction") else []


def _confusion(labels: list[str], predictions: list[str], classes: list[str]) -> dict:
    matrix = []
    for gold in classes:
        matrix.append([sum(a == gold and b == predicted for a, b in zip(labels, predictions)) for predicted in classes])
    missing = [sum(a == gold and b not in classes for a, b in zip(labels, predictions)) for gold in classes]
    return {"labels": classes, "matrix": matrix, "missing_by_gold": missing}


def evaluate_method(
    gold_rows: list[dict],
    predictions: dict[str, dict],
    bootstrap_samples: int,
    seed: int,
) -> dict:
    gate_labels, gate_predictions = [], []
    tool_labels, tool_predictions = [], []
    gate_reliability, tool_reliability = [], []
    latency_total, latency_forward, input_tokens = [], [], []
    gate_latency, tool_latency, full_action_latency = [], [], []
    evaluated = []

    for gold in gold_rows:
        gate = _prediction(predictions, gold["gate_id"])
        tool = _prediction(predictions, gold["tool_id"])
        tool_for_latency = _prediction(predictions, gold.get("tool_input_id") or gold["tool_id"])
        gate_pred = _top(gate, 1)
        tool_pred = _top(tool, 1)
        gate_value = gate_pred[0] if gate_pred else "<missing>"
        tool_value = tool_pred[0] if tool_pred else "<missing>"
        gate_correct = gate_value == gold["label_action_type"]
        tool_correct = gold["label_action_type"] == "call_tool" and tool_value == gold["label_tool"]
        tool_hit3 = gold["label_action_type"] == "call_tool" and gold["label_tool"] in _top(tool, 3)
        full_correct = gate_correct and (
            gold["label_action_type"] == "respond" or tool_correct
        )
        evaluated.append(
            {
                **gold,
                "gate_correct": gate_correct,
                "tool_correct": tool_correct,
                "tool_hit3": tool_hit3,
                "full_correct": full_correct,
            }
        )
        gate_labels.append(gold["label_action_type"])
        gate_predictions.append(gate_value)
        if gate is not None:
            confidence = max(gate["probabilities"])
            true_index = gate["option_ids"].index(gold["label_action_type"])
            gate_reliability.append(
                {
                    "confidence": confidence,
                    "correct": gate_correct,
                    "true_probability": gate["probabilities"][true_index],
                    "brier": sum(
                        (probability - int(index == true_index)) ** 2
                        for index, probability in enumerate(gate["probabilities"])
                    ),
                }
            )
            if isinstance(gate.get("total_seconds"), (int, float)):
                gate_ms = float(gate["total_seconds"]) * 1000
                gate_latency.append(gate_ms)
                if gate_value == "respond":
                    full_action_latency.append(gate_ms)
                elif tool_for_latency is not None and isinstance(tool_for_latency.get("total_seconds"), (int, float)):
                    full_action_latency.append(gate_ms + float(tool_for_latency["total_seconds"]) * 1000)
        if gold["label_action_type"] == "call_tool":
            tool_labels.append(gold["label_tool"])
            tool_predictions.append(tool_value)
            if tool is not None:
                true_index = tool["option_ids"].index(gold["label_tool"])
                tool_reliability.append(
                    {
                        "confidence": max(tool["probabilities"]),
                        "correct": tool_correct,
                        "true_probability": tool["probabilities"][true_index],
                        "brier": sum(
                            (probability - int(index == true_index)) ** 2
                            for index, probability in enumerate(tool["probabilities"])
                        ),
                    }
                )
                if isinstance(tool.get("total_seconds"), (int, float)):
                    tool_latency.append(float(tool["total_seconds"]) * 1000)

    for row in predictions.values():
        if row.get("status") != "ok":
            continue
        if isinstance(row.get("total_seconds"), (int, float)):
            latency_total.append(float(row["total_seconds"]) * 1000)
        if isinstance(row.get("forward_seconds"), (int, float)):
            latency_forward.append(float(row["forward_seconds"]) * 1000)
        if isinstance(row.get("input_tokens"), int):
            input_tokens.append(row["input_tokens"])

    gate_accuracy = sum(row["gate_correct"] for row in evaluated) / len(evaluated)
    tool_rows = [row for row in evaluated if row["label_action_type"] == "call_tool"]
    tool_hit1 = sum(row["tool_correct"] for row in tool_rows) / len(tool_rows) if tool_rows else 0.0
    tool_hit3 = sum(row["tool_hit3"] for row in tool_rows) / len(tool_rows) if tool_rows else 0.0
    full_accuracy = sum(row["full_correct"] for row in evaluated) / len(evaluated)
    expected_ids = {row["gate_id"] for row in gold_rows} | {
        row["tool_id"] for row in gold_rows if row["tool_id"]
    }
    successful = sum(identifier in predictions and predictions[identifier].get("status") == "ok" for identifier in expected_ids)
    status_counts = Counter(
        predictions[identifier].get("status", "unknown") if identifier in predictions else "missing"
        for identifier in expected_ids
    )

    return {
        "denominators": {
            "all_decisions": len(evaluated),
            "tool_decisions": len(tool_rows),
            "eligible_for_speculation": sum(row["eligible_for_speculation"] for row in evaluated),
            "expected_prediction_rows": len(expected_ids),
        },
        "coverage": successful / len(expected_ids) if expected_ids else 0.0,
        "prediction_status": dict(sorted(status_counts.items())),
        "gate": {
            "accuracy": gate_accuracy,
            "macro_f1": macro_f1(gate_labels, gate_predictions, ["call_tool", "respond"]),
            "ci95_task_cluster": clustered_ci(
                evaluated,
                lambda rows: sum(row["gate_correct"] for row in rows) / len(rows),
                bootstrap_samples,
                seed,
            ),
            "calibration": reliability(gate_reliability),
            "confusion": _confusion(gate_labels, gate_predictions, ["call_tool", "respond"]),
        },
        "tool": {
            "hit_at_1": tool_hit1,
            "hit_at_3": tool_hit3,
            "macro_f1": macro_f1(tool_labels, tool_predictions, CANONICAL_TOOL_IDS),
            "ci95_task_cluster_hit_at_1": clustered_ci(
                tool_rows,
                lambda rows: sum(row["tool_correct"] for row in rows) / len(rows),
                bootstrap_samples,
                seed + 1,
            ),
            "calibration": reliability(tool_reliability),
            "confusion": _confusion(tool_labels, tool_predictions, CANONICAL_TOOL_IDS),
        },
        "full_action": {
            "accuracy": full_accuracy,
            "ci95_task_cluster": clustered_ci(
                evaluated,
                lambda rows: sum(row["full_correct"] for row in rows) / len(rows),
                bootstrap_samples,
                seed + 2,
            ),
        },
        "latency_ms": {
            "n": len(latency_total),
            "p50_total": percentile(latency_total, 0.50),
            "p90_total": percentile(latency_total, 0.90),
            "p95_total": percentile(latency_total, 0.95),
            "p50_forward": percentile(latency_forward, 0.50),
            "mean_input_tokens": statistics.fmean(input_tokens) if input_tokens else None,
            "gate": {
                "n": len(gate_latency),
                "p50": percentile(gate_latency, 0.50),
                "p90": percentile(gate_latency, 0.90),
                "p95": percentile(gate_latency, 0.95),
            },
            "tool_only": {
                "n": len(tool_latency),
                "p50": percentile(tool_latency, 0.50),
                "p90": percentile(tool_latency, 0.90),
                "p95": percentile(tool_latency, 0.95),
            },
            "two_stage_observed_gate": {
                "n": len(full_action_latency),
                "coverage": len(full_action_latency) / len(evaluated),
                "p50": percentile(full_action_latency, 0.50),
                "p90": percentile(full_action_latency, 0.90),
                "p95": percentile(full_action_latency, 0.95),
            },
        },
    }


def evaluate_files(
    gold_path: Path,
    prediction_paths: dict[str, Path],
    output_path: Path,
    split: str = "test",
    bootstrap_samples: int = 2000,
    seed: int = 20261026,
) -> dict:
    gold = [row for row in read_jsonl(gold_path) if row["split"] == split]
    if not gold:
        raise ValueError(f"No gold rows for split {split}")
    methods = {}
    for method, path in prediction_paths.items():
        methods[method] = evaluate_method(gold, rows_by_id(read_jsonl(path)), bootstrap_samples, seed)
    report = {
        "split": split,
        "bootstrap": {"unit": "task_id", "samples": bootstrap_samples, "seed": seed},
        "methods": methods,
    }
    write_json(output_path, report)
    return report


def simulate_speculation(
    accuracy: float,
    predictor_seconds: float,
    actor_seconds: float,
    tool_seconds: float,
) -> dict:
    baseline = actor_seconds + tool_seconds
    hit = max(actor_seconds, predictor_seconds + tool_seconds)
    miss = baseline
    expected = accuracy * hit + (1.0 - accuracy) * miss
    return {
        "baseline_seconds": baseline,
        "expected_speculative_seconds": expected,
        "expected_seconds_saved": baseline - expected,
        "expected_fraction_saved": (baseline - expected) / baseline if baseline else 0.0,
        "expected_wasted_tool_seconds": (1.0 - accuracy) * tool_seconds,
    }

