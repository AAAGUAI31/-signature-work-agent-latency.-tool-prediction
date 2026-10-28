from __future__ import annotations

import math
from typing import Iterable


def softmax(values: list[float], temperature: float = 1.0) -> list[float]:
    if not math.isfinite(temperature) or temperature <= 0:
        raise ValueError("temperature must be finite and positive")
    if len(values) < 2 or any(not math.isfinite(value) for value in values):
        raise ValueError("scores must contain at least two finite values")
    scaled = [value / temperature for value in values]
    maximum = max(scaled)
    weights = [math.exp(value - maximum) for value in scaled]
    total = sum(weights)
    return [value / total for value in weights]


def normalize_prediction(row: dict) -> dict:
    identifier = row.get("id") or row.get("sample_id")
    option_ids = row.get("option_ids")
    probabilities = row.get("probabilities")
    if not isinstance(identifier, str) or not identifier:
        raise ValueError("prediction id must be a nonempty string")
    if not isinstance(option_ids, list) or len(option_ids) < 2 or len(option_ids) != len(set(option_ids)):
        raise ValueError(f"{identifier}: option_ids must be unique")
    if not isinstance(probabilities, list) or len(probabilities) != len(option_ids):
        raise ValueError(f"{identifier}: probabilities length mismatch")
    if any(not isinstance(value, (int, float)) or not math.isfinite(value) or value < 0 for value in probabilities):
        raise ValueError(f"{identifier}: invalid probability")
    total = sum(probabilities)
    if total <= 0:
        raise ValueError(f"{identifier}: zero probability mass")
    probabilities = [float(value / total) for value in probabilities]
    ranked = sorted(range(len(option_ids)), key=lambda index: (-probabilities[index], index))
    return {
        **row,
        "id": identifier,
        "option_ids": option_ids,
        "probabilities": probabilities,
        "prediction": option_ids[ranked[0]],
        "ranked_option_ids": [option_ids[index] for index in ranked],
        "status": row.get("status", "ok"),
    }


def rows_by_id(rows: Iterable[dict]) -> dict[str, dict]:
    result = {}
    for raw in rows:
        row = normalize_prediction(raw)
        if row["id"] in result:
            raise ValueError(f"duplicate prediction id: {row['id']}")
        result[row["id"]] = row
    return result

