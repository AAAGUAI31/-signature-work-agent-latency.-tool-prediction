from tool_prediction.metrics import evaluate_method, exact_call_match, simulate_speculation


def _gold(sample, task, action, tool=None):
    return {
        "sample_id": sample,
        "gate_id": f"{sample}-gate",
        "tool_id": f"{sample}-tool" if tool else None,
        "tool_input_id": f"{sample}-tool",
        "task_id": task,
        "split": "test",
        "label_action_type": action,
        "label_tool": tool,
        "label_args": None,
        "eligible_for_speculation": False,
    }


def _prediction(identifier, options, probabilities, status="ok"):
    ranked = [item for _, item in sorted(zip(probabilities, options), reverse=True)]
    return {
        "id": identifier,
        "option_ids": options,
        "probabilities": probabilities,
        "prediction": ranked[0],
        "ranked_option_ids": ranked,
        "status": status,
    }


def test_join_is_by_id_and_missing_rows_reduce_coverage_and_accuracy():
    gold = [
        _gold("a", "task-1", "respond"),
        _gold("b", "task-2", "call_tool", "get_order_details"),
    ]
    predictions = {
        "b-gate": _prediction("b-gate", ["call_tool", "respond"], [0.9, 0.1]),
        "a-gate": _prediction("a-gate", ["call_tool", "respond"], [0.1, 0.9]),
    }
    report = evaluate_method(gold, predictions, bootstrap_samples=30, seed=1)
    assert report["gate"]["accuracy"] == 1.0
    assert report["tool"]["hit_at_1"] == 0.0
    assert report["full_action"]["accuracy"] == 0.5
    assert report["coverage"] == 2 / 3
    assert report["prediction_status"]["missing"] == 1


def test_exact_call_keeps_identifier_punctuation_and_values():
    assert exact_call_match("get_order_details", {"order_id": "#W123"}, "get_order_details", {"order_id": "#W123"})
    assert not exact_call_match("get_order_details", {"order_id": "W123"}, "get_order_details", {"order_id": "#W123"})
    assert not exact_call_match("get_order_details", {"order_id": "#W124"}, "get_order_details", {"order_id": "#W123"})


def test_speculation_equation_matches_client_side_model():
    result = simulate_speculation(accuracy=1.0, predictor_seconds=0.1, actor_seconds=2.0, tool_seconds=1.0)
    assert result["baseline_seconds"] == 3.0
    assert result["expected_speculative_seconds"] == 2.0
    miss = simulate_speculation(accuracy=0.0, predictor_seconds=0.1, actor_seconds=2.0, tool_seconds=1.0)
    assert miss["expected_speculative_seconds"] == 3.0

