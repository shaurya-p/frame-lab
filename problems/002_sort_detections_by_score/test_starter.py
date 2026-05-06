import pytest
from starter import sort_detections_by_score


def det(score: float, class_id: int = 0) -> dict:
    return {"bbox": [0, 0, 10, 10], "score": score, "class_id": class_id}


def test_descending_default():
    detections = [det(0.5), det(0.9), det(0.2)]
    result = sort_detections_by_score(detections)
    scores = [d["score"] for d in result]
    assert scores == [0.9, 0.5, 0.2], f"Expected [0.9, 0.5, 0.2], got {scores}"


def test_ascending():
    detections = [det(0.5), det(0.9), det(0.2)]
    result = sort_detections_by_score(detections, descending=False)
    scores = [d["score"] for d in result]
    assert scores == [0.2, 0.5, 0.9], f"Expected [0.2, 0.5, 0.9], got {scores}"


def test_empty():
    result = sort_detections_by_score([])
    assert result == [], f"Expected [], got {result}"


def test_single_detection():
    detections = [det(0.7)]
    result = sort_detections_by_score(detections)
    assert len(result) == 1
    assert result[0]["score"] == 0.7


def test_equal_score_stability():
    detections = [det(0.5, class_id=1), det(0.5, class_id=2), det(0.5, class_id=3)]
    result = sort_detections_by_score(detections)
    class_ids = [d["class_id"] for d in result]
    assert class_ids == [1, 2, 3], f"Expected stable order [1, 2, 3], got {class_ids}"


def test_already_sorted_descending():
    detections = [det(0.9), det(0.5), det(0.1)]
    result = sort_detections_by_score(detections)
    scores = [d["score"] for d in result]
    assert scores == [0.9, 0.5, 0.1], f"Expected [0.9, 0.5, 0.1], got {scores}"


def test_reverse_sorted_input():
    detections = [det(0.1), det(0.5), det(0.9)]
    result = sort_detections_by_score(detections)
    scores = [d["score"] for d in result]
    assert scores == [0.9, 0.5, 0.1], f"Expected [0.9, 0.5, 0.1], got {scores}"


def test_negative_scores():
    detections = [det(-0.5), det(-0.1), det(-0.9)]
    result = sort_detections_by_score(detections)
    scores = [d["score"] for d in result]
    assert scores == [-0.1, -0.5, -0.9], f"Expected [-0.1, -0.5, -0.9], got {scores}"


def test_input_not_mutated():
    detections = [det(0.5), det(0.9), det(0.2)]
    original_scores = [d["score"] for d in detections]
    sort_detections_by_score(detections)
    after_scores = [d["score"] for d in detections]
    assert original_scores == after_scores, (
        f"Input was mutated: before={original_scores}, after={after_scores}"
    )
