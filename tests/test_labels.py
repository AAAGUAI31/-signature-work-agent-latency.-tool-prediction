from tool_prediction.prepare import parse_assistant_action


def test_response_and_tool_labels():
    response = parse_assistant_action({"role": "assistant", "content": "Please confirm."})
    assert response.action_type == "respond"
    tool = parse_assistant_action(
        {
            "role": "assistant",
            "content": None,
            "tool_calls": [
                {"function": {"name": "get_order_details", "arguments": '{"order_id":"#W1234567"}'}}
            ],
        }
    )
    assert tool.action_type == "call_tool"
    assert tool.tool_name == "get_order_details"
    assert tool.arguments == {"order_id": "#W1234567"}


def test_multiple_calls_and_mixed_actions_are_explicitly_excluded():
    call = {"function": {"name": "think", "arguments": "{}"}}
    multiple = parse_assistant_action({"role": "assistant", "content": None, "tool_calls": [call, call]})
    mixed = parse_assistant_action({"role": "assistant", "content": "also text", "tool_calls": [call]})
    assert multiple.exclusion == "multiple_tool_calls"
    assert mixed.exclusion == "mixed_response_and_tool_call"

