import pytest
from starter import group_detections_by_class


def det(score: float, class_id: int) -> dict:
    return {"bbox": [0, 0, 10, 10], "score": score, "class_id": class_id}


def test_normal_grouping():
    detections = [det(0.9, 0), det(0.8, 1), det(0.7, 0), det(0.6, 1), det(0.5, 2)]
    result = group_detections_by_class(detections)
    assert set(result.keys()) == {0, 1, 2}
    assert [d["score"] for d in result[0]] == [0.9, 0.7], (
        f"Expected scores [0.9, 0.7] for class 0, got {[d['score'] for d in result[0]]}"
    )
    assert [d["score"] for d in result[1]] == [0.8, 0.6], (
        f"Expected scores [0.8, 0.6] for class 1, got {[d['score'] for d in result[1]]}"
    )
    assert [d["score"] for d in result[2]] == [0.5], (
        f"Expected scores [0.5] for class 2, got {[d['score'] for d in result[2]]}"
    )


def test_empty_input():
    result = group_detections_by_class([])
    assert result == {}, f"Expected {{}}, got {result}"


def test_single_detection():
    detections = [det(0.9, 3)]
    result = group_detections_by_class(detections)
    assert list(result.keys()) == [3]
    assert len(result[3]) == 1
    assert result[3][0]["score"] == 0.9


def test_all_same_class():
    detections = [det(0.9, 0), det(0.5, 0), det(0.1, 0)]
    result = group_detections_by_class(detections)
    assert list(result.keys()) == [0]
    scores = [d["score"] for d in result[0]]
    assert scores == [0.9, 0.5, 0.1], f"Expected [0.9, 0.5, 0.1], got {scores}"


def test_negative_class_id():
    detections = [det(0.8, -1), det(0.7, 0), det(0.6, -1)]
    result = group_detections_by_class(detections)
    assert -1 in result
    scores = [d["score"] for d in result[-1]]
    assert scores == [0.8, 0.6], f"Expected [0.8, 0.6] for class -1, got {scores}"


def test_zero_class_id():
    detections = [det(0.9, 0), det(0.8, 0)]
    result = group_detections_by_class(detections)
    assert 0 in result
    assert len(result[0]) == 2


def test_order_preserved_within_class():
    detections = [det(0.3, 1), det(0.9, 0), det(0.1, 1), det(0.5, 1)]
    result = group_detections_by_class(detections)
    scores = [d["score"] for d in result[1]]
    assert scores == [0.3, 0.1, 0.5], (
        f"Expected original order [0.3, 0.1, 0.5] for class 1, got {scores}"
    )


def test_output_is_plain_dict():
    detections = [det(0.9, 0)]
    result = group_detections_by_class(detections)
    assert type(result) is dict, f"Expected plain dict, got {type(result)}"


def test_input_list_not_mutated():
    detections = [det(0.9, 0), det(0.8, 1), det(0.7, 0)]
    original_ids = [d["class_id"] for d in detections]
    group_detections_by_class(detections)
    after_ids = [d["class_id"] for d in detections]
    assert original_ids == after_ids, (
        f"Input list was mutated: before={original_ids}, after={after_ids}"
    )


def test_detection_dicts_not_mutated():
    detections = [det(0.9, 0), det(0.8, 1)]
    originals = [d.copy() for d in detections]
    group_detections_by_class(detections)
    for original, after in zip(originals, detections):
        assert original == after, (
            f"Detection dict was mutated: before={original}, after={after}"
        )
