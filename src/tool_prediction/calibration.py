from __future__ import annotations

import math

from .predictions import softmax


def mean_nll(rows: list[dict], temperature: float) -> float:
    total = 0.0
    for row in rows:
        probabilities = softmax(row["logits"], temperature)
        total -= math.log(max(probabilities[row["true_index"]], 1e-12))
    return total / len(rows)


def fit_temperature(rows: list[dict], bounds=(0.05, 20.0), iterations: int = 80) -> float:
    if not rows:
        raise ValueError("calibration set is empty")
    ratio = (math.sqrt(5.0) - 1.0) / 2.0
    low, high = bounds
    left = high - ratio * (high - low)
    right = low + ratio * (high - low)
    left_loss, right_loss = mean_nll(rows, left), mean_nll(rows, right)
    for _ in range(iterations):
        if left_loss < right_loss:
            high, right, right_loss = right, left, left_loss
            left = high - ratio * (high - low)
            left_loss = mean_nll(rows, left)
        else:
            low, left, left_loss = left, right, right_loss
            right = low + ratio * (high - low)
            right_loss = mean_nll(rows, right)
    return (low + high) / 2.0


def reliability(rows: list[dict], bins: int = 10) -> dict:
    if not rows:
        return {"ece": None, "brier": None, "nll": None, "bins": []}
    grouped = [[] for _ in range(bins)]
    brier = nll = 0.0
    for row in rows:
        confidence = min(1.0, max(0.0, row["confidence"]))
        correct = int(row["correct"])
        grouped[min(bins - 1, int(confidence * bins))].append((confidence, correct))
        brier += row.get("brier", (confidence - correct) ** 2)
        true_probability = row.get("true_probability")
        nll -= math.log(max(true_probability if true_probability is not None else (confidence if correct else 1.0 - confidence), 1e-12))
    result, ece = [], 0.0
    for index, values in enumerate(grouped):
        if not values:
            continue
        mean_confidence = sum(value[0] for value in values) / len(values)
        accuracy = sum(value[1] for value in values) / len(values)
        ece += len(values) / len(rows) * abs(mean_confidence - accuracy)
        result.append(
            {
                "lower": index / bins,
                "upper": (index + 1) / bins,
                "n": len(values),
                "mean_confidence": mean_confidence,
                "accuracy": accuracy,
            }
        )
    return {"ece": ece, "brier": brier / len(rows), "nll": nll / len(rows), "bins": result}
