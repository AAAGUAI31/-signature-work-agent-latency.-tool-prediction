from tool_prediction.constants import CANONICAL_TOOL_IDS, GATE_OPTIONS
from tool_prediction.prepare import gate_options_for_id


def test_gate_is_two_way_and_tool_order_is_unique_16_way():
    assert [option["id"] for option in GATE_OPTIONS] == ["call_tool", "respond"]
    assert len(CANONICAL_TOOL_IDS) == 16
    assert len(set(CANONICAL_TOOL_IDS)) == 16
    assert CANONICAL_TOOL_IDS[0] == "calculate"
    assert CANONICAL_TOOL_IDS[-1] == "transfer_to_human_agents"


def test_gate_order_is_deterministic_and_counterbalanced():
    orders = [tuple(option["id"] for option in gate_options_for_id(f"sample-{index}")) for index in range(100)]
    assert orders == [
        tuple(option["id"] for option in gate_options_for_id(f"sample-{index}")) for index in range(100)
    ]
    assert set(orders) == {("call_tool", "respond"), ("respond", "call_tool")}

