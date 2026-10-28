from tool_prediction.metrics import clustered_ci


def test_cluster_bootstrap_resamples_whole_tasks():
    rows = [
        {"task_id": "a", "value": 1},
        {"task_id": "a", "value": 1},
        {"task_id": "b", "value": 0},
    ]
    interval = clustered_ci(rows, lambda sample: sum(row["value"] for row in sample) / len(sample), 200, 3)
    assert interval[0] == 0.0
    assert interval[1] == 1.0

