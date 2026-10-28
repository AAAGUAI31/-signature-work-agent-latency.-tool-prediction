import pytest

from tool_prediction.windowing import fit_row_to_budget


def _encode(row, max_tokens):
    length = 20 * len(row["state"]["messages"])
    if length > max_tokens:
        raise ValueError(f"Row {row['id']}: {length} input tokens exceed limit {max_tokens}; no truncation allowed")
    return list(range(length)), [1, 2], "digest"


def test_window_keeps_system_and_most_recent_complete_user_block():
    row = {
        "id": "sample-gate",
        "state": {
            "messages": [
                {"role": "system", "content": "policy"},
                {"role": "user", "content": "old"},
                {"role": "assistant", "content": "old answer"},
                {"role": "user", "content": "recent"},
                {"role": "assistant", "content": "recent answer"},
            ]
        },
        "question": "next action?",
        "options": [
            {"id": "call_tool", "description": "tool"},
            {"id": "respond", "description": "message"},
        ],
    }
    fitted = fit_row_to_budget(row, _encode, 80)
    contents = [message["content"] for message in fitted["state"]["messages"]]
    assert "policy" in contents
    assert "old" not in contents
    assert "recent" in contents
    assert fitted["window"]["omitted_blocks"] == 1
    assert row["state"]["messages"][1]["content"] == "old"


def test_window_does_not_hide_non_length_validation_errors():
    def invalid(row, max_tokens):
        raise ValueError("answer slots collide")

    row = {
        "id": "bad",
        "state": {"messages": [{"role": "system", "content": "policy"}]},
        "question": "next action?",
        "options": [
            {"id": "call_tool", "description": "tool"},
            {"id": "respond", "description": "message"},
        ],
    }
    with pytest.raises(ValueError, match="answer slots collide"):
        fit_row_to_budget(row, invalid, 80)
