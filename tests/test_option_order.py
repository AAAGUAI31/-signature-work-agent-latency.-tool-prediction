from tool_prediction.constants import CANONICAL_TOOL_IDS, GATE_OPTIONS


def test_gate_is_two_way_and_tool_order_is_unique_16_way():
    assert [option["id"] for option in GATE_OPTIONS] == ["call_tool", "respond"]
    assert len(CANONICAL_TOOL_IDS) == 16
    assert len(set(CANONICAL_TOOL_IDS)) == 16
    assert CANONICAL_TOOL_IDS[0] == "calculate"
    assert CANONICAL_TOOL_IDS[-1] == "transfer_to_human_agents"

