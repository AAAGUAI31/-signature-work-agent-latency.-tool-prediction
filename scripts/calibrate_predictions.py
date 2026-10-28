from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from tool_prediction.calibration import fit_temperature, reliability
from tool_prediction.io import read_jsonl, write_json, write_jsonl
from tool_prediction.predictions import softmax


def labels(gold_path: Path, split: str) -> dict[str, str]:
    result = {}
    for row in read_jsonl(gold_path):
        if row["split"] != split:
            continue
        result[row["gate_id"]] = row["label_action_type"]
        if row["tool_id"]:
            result[row["tool_id"]] = row["label_tool"]
    return result


def calibration_rows(predictions: list[dict], expected: dict[str, str]) -> list[dict]:
    result = []
    for row in predictions:
        if row.get("status", "ok") != "ok" or row["id"] not in expected:
            continue
        option_ids = row.get("option_ids")
        logits = row.get("option_logits")
        if not isinstance(logits, list) or len(logits) != len(option_ids):
            raise ValueError(f"{row['id']}: calibration requires option_logits")
        true_id = expected[row["id"]]
        if true_id not in option_ids:
            raise ValueError(f"{row['id']}: gold label is absent from option_ids")
        result.append({"id": row["id"], "logits": [float(value) for value in logits], "true_index": option_ids.index(true_id)})
    return result


def scored(rows: list[dict], temperature: float) -> list[dict]:
    result = []
    for row in rows:
        probabilities = softmax(row["logits"], temperature)
        prediction = max(range(len(probabilities)), key=probabilities.__getitem__)
        result.append(
            {
                "confidence": probabilities[prediction],
                "correct": prediction == row["true_index"],
                "true_probability": probabilities[row["true_index"]],
                "brier": sum(
                    (probability - int(index == row["true_index"])) ** 2
                    for index, probability in enumerate(probabilities)
                ),
            }
        )
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description="Fit temperature on val and apply the frozen value to test predictions.")
    parser.add_argument("--gold", type=Path, required=True)
    parser.add_argument("--val-predictions", type=Path, required=True)
    parser.add_argument("--test-predictions", type=Path, required=True)
    parser.add_argument("--calibrated-out", type=Path, required=True)
    parser.add_argument("--report", type=Path, required=True)
    args = parser.parse_args()
    val_predictions = list(read_jsonl(args.val_predictions))
    test_predictions = list(read_jsonl(args.test_predictions))
    val = calibration_rows(val_predictions, labels(args.gold, "val"))
    test = calibration_rows(test_predictions, labels(args.gold, "test"))
    temperature = fit_temperature(val)
    calibrated = []
    for row in test_predictions:
        result = dict(row)
        if row.get("status", "ok") == "ok" and isinstance(row.get("option_logits"), list):
            result["probabilities"] = softmax([float(value) for value in row["option_logits"]], temperature)
            result["calibration"] = {"method": "temperature_scaling", "temperature": temperature, "fit_split": "val"}
        calibrated.append(result)
    write_jsonl(args.calibrated_out, calibrated)
    report = {
        "temperature": temperature,
        "fit_split": "val",
        "val_rows": len(val),
        "test_rows": len(test),
        "val_uncalibrated": reliability(scored(val, 1.0)),
        "val_calibrated": reliability(scored(val, temperature)),
        "test_uncalibrated": reliability(scored(test, 1.0)),
        "test_calibrated_frozen": reliability(scored(test, temperature)),
    }
    write_json(args.report, report)
    print(json.dumps({"temperature": temperature, "val_rows": len(val), "test_rows": len(test)}))


if __name__ == "__main__":
    main()

