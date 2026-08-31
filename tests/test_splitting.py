import numpy as np

from gradcell.data import group_split


def test_group_split_is_disjoint_deterministic_and_nonempty():
    groups = np.repeat(np.arange(10), np.arange(1, 11))
    first = group_split(groups, seed=7)
    second = group_split(groups, seed=7)

    assert all(np.array_equal(left, right) for left, right in zip(first, second))
    assert all(len(partition) > 0 for partition in first)
    split_groups = [set(groups[index]) for index in first]
    assert split_groups[0].isdisjoint(split_groups[1])
    assert split_groups[0].isdisjoint(split_groups[2])
    assert split_groups[1].isdisjoint(split_groups[2])


def test_group_split_rejects_too_few_groups():
    try:
        group_split(np.array([0, 0, 1, 1]), seed=7)
    except ValueError as error:
        assert "three groups" in str(error)
    else:
        raise AssertionError("Expected a ValueError for fewer than three groups")
