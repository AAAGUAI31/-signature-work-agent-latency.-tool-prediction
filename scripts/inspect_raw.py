from __future__ import annotations

import argparse
import hashlib
import json
import sys
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from tool_prediction.io import read_json, write_json
from tool_prediction.prepare import parse_assistant_action


def main() -> None:
    parser = argparse.ArgumentParser(description="Audit a raw tau-bench historical trajectory file.")
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    runs = read_json(args.input)
    role_counts = Counter()
    action_counts = Counter()
    exclusions = Counter()
    tools = Counter()
    lengths = []
    fields = Counter()
    for run in runs:
        fields.update(run.keys())
        trajectory = run.get("traj", [])
        lengths.append(len(trajectory))
        for message in trajectory:
            role_counts[message.get("role", "<missing>")] += 1
            if message.get("role") == "assistant":
                action = parse_assistant_action(message)
                action_counts[action.action_type] += 1
                if action.exclusion:
                    exclusions[action.exclusion] += 1
                if action.tool_name:
                    tools[action.tool_name] += 1
    report = {
        "source": args.input.name,
        "sha256": hashlib.sha256(args.input.read_bytes()).hexdigest(),
        "runs": len(runs),
        "top_level_fields": dict(sorted(fields.items())),
        "roles": dict(sorted(role_counts.items())),
        "assistant_actions": dict(sorted(action_counts.items())),
        "exclusions": dict(sorted(exclusions.items())),
        "tools": dict(sorted(tools.items())),
        "trajectory_messages": {
            "min": min(lengths) if lengths else None,
            "max": max(lengths) if lengths else None,
            "mean": sum(lengths) / len(lengths) if lengths else None,
        },
    }
    write_json(args.output, report)
    print(json.dumps({"runs": report["runs"], "assistant_actions": report["assistant_actions"]}))


if __name__ == "__main__":
    main()

