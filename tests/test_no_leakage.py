from tool_prediction.constants import CANONICAL_TOOL_IDS
from tool_prediction.prepare import build_rows


def test_current_and_future_actions_are_absent_from_state(tmp_path):
    path = tmp_path / "gpt-4o-retail.json"
    prefix = [
        {"role": "system", "content": "policy"},
        {"role": "user", "content": "find my order"},
    ]
    current = {
        "role": "assistant",
        "content": None,
        "tool_calls": [{"function": {"name": "get_order_details", "arguments": '{"order_id":"#W1"}'}}],
    }
    future = {"role": "tool", "name": "get_order_details", "content": "SECRET_FUTURE_RESULT"}
    path.write_text(
        __import__("json").dumps([{"task_id": 7, "trial": 0, "reward": 1, "traj": prefix + [current, future]}]),
        encoding="utf-8",
    )
    options = [{"id": name, "description": name} for name in CANONICAL_TOOL_IDS]
    rows_a, _, _ = build_rows([path], options, 1)
    state_a = rows_a["scored"]["train"] + rows_a["scored"]["val"] + rows_a["scored"]["test"]
    gate_a = next(row for row in state_a if row["id"].endswith("-gate"))["state"]
    current["tool_calls"][0]["function"]["arguments"] = '{"order_id":"#CHANGED"}'
    future["content"] = "CHANGED_FUTURE"
    path.write_text(
        __import__("json").dumps([{"task_id": 7, "trial": 0, "reward": 0, "traj": prefix + [current, future]}]),
        encoding="utf-8",
    )
    rows_b, _, _ = build_rows([path], options, 1)
    state_b = rows_b["scored"]["train"] + rows_b["scored"]["val"] + rows_b["scored"]["test"]
    gate_b = next(row for row in state_b if row["id"].endswith("-gate"))["state"]
    assert gate_a == gate_b
    assert gate_a["messages"] == prefix
    assert gate_a["available_tool_ids"] == CANONICAL_TOOL_IDS
    assert "reward" not in str(gate_a).lower()
    assert "SECRET" not in str(gate_a)

