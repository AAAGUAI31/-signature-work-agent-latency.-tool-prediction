from __future__ import annotations

import ast
import re
import subprocess
from pathlib import Path

from .constants import CANONICAL_TOOL_IDS


def git_sha(repo: Path) -> str:
    result = subprocess.run(
        ["git", "-C", str(repo), "rev-parse", "HEAD"],
        check=True,
        capture_output=True,
        text=True,
    )
    sha = result.stdout.strip()
    if not re.fullmatch(r"[0-9a-f]{40}", sha):
        raise ValueError(f"Unexpected git revision: {sha!r}")
    return sha


def _literal_return(path: Path) -> dict:
    tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    for node in ast.walk(tree):
        if isinstance(node, ast.FunctionDef) and node.name == "get_info":
            for statement in node.body:
                if isinstance(statement, ast.Return):
                    value = ast.literal_eval(statement.value)
                    if isinstance(value, dict):
                        return value
    raise ValueError(f"No literal get_info() return found in {path}")


def load_retail_tool_schemas(repo: Path) -> list[dict]:
    tools_dir = repo / "tau_bench" / "envs" / "retail" / "tools"
    schemas = []
    for tool_id in CANONICAL_TOOL_IDS:
        info = _literal_return(tools_dir / f"{tool_id}.py")
        function = info.get("function", {})
        if function.get("name") != tool_id:
            raise ValueError(f"Schema name mismatch for {tool_id}")
        schemas.append(info)
    return schemas


def options_from_schemas(schemas: list[dict]) -> list[dict]:
    options = []
    for schema in schemas:
        function = schema["function"]
        options.append({"id": function["name"], "description": function["description"]})
    if [option["id"] for option in options] != CANONICAL_TOOL_IDS:
        raise ValueError("Tool option order differs from the registered canonical order")
    return options

