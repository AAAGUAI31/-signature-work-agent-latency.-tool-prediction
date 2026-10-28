from tool_prediction.prepare import assign_split


def test_same_task_is_always_in_same_split_across_actors_and_trials():
    for task_id in range(115):
        expected = assign_split(task_id, 20261026)
        assert expected == assign_split(str(task_id), 20261026)
        assert expected == assign_split(task_id, 20261026)


def test_split_is_deterministic_and_complete():
    values = [assign_split(task_id, 20261026) for task_id in range(1000)]
    assert set(values) == {"train", "val", "test"}
    assert values == [assign_split(task_id, 20261026) for task_id in range(1000)]

