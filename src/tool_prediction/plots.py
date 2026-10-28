from __future__ import annotations

import json
from pathlib import Path


def make_plots(report_path: Path, output_dir: Path) -> list[Path]:
    import matplotlib.pyplot as plt
    import numpy as np

    report = json.loads(report_path.read_text(encoding="utf-8"))
    methods = report["methods"]
    output_dir.mkdir(parents=True, exist_ok=True)
    created = []

    figure, axis = plt.subplots(figsize=(7, 5))
    for name, result in methods.items():
        latency = result["latency_ms"]["p50_total"]
        if latency is None:
            continue
        axis.scatter(latency, result["full_action"]["accuracy"], s=70, label=name)
        axis.annotate(name, (latency, result["full_action"]["accuracy"]), xytext=(4, 4), textcoords="offset points")
    axis.set(xlabel="P50 prediction latency (ms)", ylabel="Full-action accuracy", title="Accuracy-latency tradeoff")
    axis.grid(alpha=0.25)
    path = output_dir / "pareto_accuracy_latency.png"
    figure.tight_layout()
    figure.savefig(path, dpi=180)
    plt.close(figure)
    created.append(path)

    for name, result in methods.items():
        confusion = result["tool"]["confusion"]
        matrix = np.asarray(confusion["matrix"], dtype=float)
        figure, axis = plt.subplots(figsize=(10, 9))
        image = axis.imshow(matrix, cmap="Blues")
        labels = confusion["labels"]
        axis.set_xticks(range(len(labels)), labels, rotation=90, fontsize=7)
        axis.set_yticks(range(len(labels)), labels, fontsize=7)
        axis.set(xlabel="Predicted tool", ylabel="Recorded tool", title=f"Tool confusion - {name}")
        figure.colorbar(image, ax=axis, fraction=0.046)
        path = output_dir / f"tool_confusion_{name}.png"
        figure.tight_layout()
        figure.savefig(path, dpi=180)
        plt.close(figure)
        created.append(path)

    figure, axis = plt.subplots(figsize=(6, 6))
    axis.plot([0, 1], [0, 1], "--", color="0.5", label="perfect")
    for name, result in methods.items():
        bins = result["gate"]["calibration"]["bins"]
        if bins:
            axis.plot(
                [entry["mean_confidence"] for entry in bins],
                [entry["accuracy"] for entry in bins],
                marker="o",
                label=name,
            )
    axis.set(xlim=(0, 1), ylim=(0, 1), xlabel="Confidence", ylabel="Accuracy", title="Gate reliability")
    axis.legend()
    axis.grid(alpha=0.25)
    path = output_dir / "gate_reliability.png"
    figure.tight_layout()
    figure.savefig(path, dpi=180)
    plt.close(figure)
    created.append(path)
    return created
