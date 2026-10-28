from tool_prediction.prepare import assign_split


def test_seed_changes_are_explicit_and_reproducible():
    first = [assign_split(value, 1) for value in range(200)]
    assert first == [assign_split(value, 1) for value in range(200)]
    assert first != [assign_split(value, 2) for value in range(200)]
