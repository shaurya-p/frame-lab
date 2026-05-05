import pytest
from starter import filter_detections_by_score


@pytest.fixture
def sample():
    return [
        {"score": 0.9, "class_id": 0, "box": [0, 0, 10, 10]},
        {"score": 0.5, "class_id": 1, "box": [5, 5, 15, 15]},
        {"score": 0.3, "class_id": 0, "box": [2, 2, 8, 8]},
        {"score": 0.7, "class_id": 2, "box": [1, 1, 6, 6]},
    ]


def test_basic_filter(sample):
    result = filter_detections_by_score(sample, min_score=0.6)
    assert [d["score"] for d in result] == [0.9, 0.7]


def test_order_preserved(sample):
    result = filter_detections_by_score(sample, min_score=0.4)
    assert [d["score"] for d in result] == [0.9, 0.5, 0.7]


def test_threshold_is_inclusive(sample):
    result = filter_detections_by_score(sample, min_score=0.5)
    assert any(d["score"] == 0.5 for d in result)


def test_all_pass(sample):
    result = filter_detections_by_score(sample, min_score=0.0)
    assert len(result) == 4


def test_none_pass(sample):
    result = filter_detections_by_score(sample, min_score=1.0)
    assert result == []


def test_empty_input():
    assert filter_detections_by_score([], min_score=0.5) == []


def test_single_passing():
    dets = [{"score": 0.8, "class_id": 0, "box": [0, 0, 1, 1]}]
    assert len(filter_detections_by_score(dets, min_score=0.5)) == 1


def test_single_failing():
    dets = [{"score": 0.2, "class_id": 0, "box": [0, 0, 1, 1]}]
    assert filter_detections_by_score(dets, min_score=0.5) == []


def test_all_same_score():
    dets = [{"score": 0.5, "class_id": i, "box": [0, 0, 1, 1]} for i in range(3)]
    assert len(filter_detections_by_score(dets, min_score=0.5)) == 3


def test_input_list_not_mutated(sample):
    original_len = len(sample)
    filter_detections_by_score(sample, min_score=0.6)
    assert len(sample) == original_len
