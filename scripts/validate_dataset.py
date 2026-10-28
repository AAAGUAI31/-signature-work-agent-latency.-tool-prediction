from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from tool_prediction.validation import validate_data_dir


def main() -> None:
    parser = argparse.ArgumentParser(description="Validate joins, labels, splits, and canonical options.")
    parser.add_argument("--data-dir", type=Path, required=True)
    args = parser.parse_args()
    print(json.dumps(validate_data_dir(args.data_dir), indent=2))


if __name__ == "__main__":
    main()

