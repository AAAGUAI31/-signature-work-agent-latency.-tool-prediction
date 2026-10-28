from __future__ import annotations

GATE_OPTIONS = [
    {"id": "call_tool", "description": "The actor invokes one of the available tools next."},
    {"id": "respond", "description": "The actor sends a message to the user without a tool invocation."},
]

GATE_QUESTION = (
    "Predict whether the main actor's very next action will be a tool invocation or a direct "
    "message to the user. Predict the recorded actor, not the ideal action. Do not perform the task."
)
TOOL_QUESTION = (
    "Predict the exact tool NAME the main actor will invoke next. Predict the recorded actor, "
    "not the ideal tool. Do not perform the task."
)

CANONICAL_TOOL_IDS = [
    "calculate",
    "cancel_pending_order",
    "exchange_delivered_order_items",
    "find_user_id_by_email",
    "find_user_id_by_name_zip",
    "get_order_details",
    "get_product_details",
    "get_user_details",
    "list_all_product_types",
    "modify_pending_order_address",
    "modify_pending_order_items",
    "modify_pending_order_payment",
    "modify_user_address",
    "return_delivered_order_items",
    "think",
    "transfer_to_human_agents",
]

READ_ONLY_SPECULATION_TOOLS = {
    "calculate",
    "find_user_id_by_email",
    "find_user_id_by_name_zip",
    "get_order_details",
    "get_product_details",
    "get_user_details",
    "list_all_product_types",
}

PREDICTION_SCHEMA_VERSION = "tau-retail-next-action-v1"

