import numpy as np
import pytest
from starter import arrays_to_detections


def test_normal_multiple_detections():
    boxes = np.array([[0, 0, 10, 10], [5, 5, 15, 15], [1, 1, 8, 8]], dtype=float)
    scores = np.array([0.9, 0.6, 0.3], dtype=float)
    class_ids = np.array([0, 2, 1], dtype=int)
    result = arrays_to_detections(boxes, scores, class_ids)
    assert len(result) == 3
    assert result[0] == {"bbox": [0.0, 0.0, 10.0, 10.0], "score": 0.9, "class_id": 0}
    assert result[1] == {"bbox": [5.0, 5.0, 15.0, 15.0], "score": 0.6, "class_id": 2}
    assert result[2] == {"bbox": [1.0, 1.0, 8.0, 8.0], "score": 0.3, "class_id": 1}


def test_empty_input():
    result = arrays_to_detections(
        np.empty((0, 4), dtype=float),
        np.array([], dtype=float),
        np.array([], dtype=int),
    )
    assert result == [], f"Expected [], got {result}"


def test_single_detection():
    boxes = np.array([[2, 3, 12, 13]], dtype=float)
    scores = np.array([0.75], dtype=float)
    class_ids = np.array([5], dtype=int)
    result = arrays_to_detections(boxes, scores, class_ids)
    assert len(result) == 1
    assert result[0] == {"bbox": [2.0, 3.0, 12.0, 13.0], "score": 0.75, "class_id": 5}


def test_bbox_is_plain_list():
    boxes = np.array([[0, 0, 10, 10]], dtype=float)
    scores = np.array([0.9], dtype=float)
    class_ids = np.array([0], dtype=int)
    result = arrays_to_detections(boxes, scores, class_ids)
    assert type(result[0]["bbox"]) is list, (
        f"Expected bbox to be plain list, got {type(result[0]['bbox'])}"
    )


def test_score_is_plain_float():
    boxes = np.array([[0, 0, 10, 10]], dtype=float)
    scores = np.array([0.9], dtype=float)
    class_ids = np.array([0], dtype=int)
    result = arrays_to_detections(boxes, scores, class_ids)
    assert type(result[0]["score"]) is float, (
        f"Expected score to be plain float, got {type(result[0]['score'])}"
    )


def test_class_id_is_plain_int():
    boxes = np.array([[0, 0, 10, 10]], dtype=float)
    scores = np.array([0.9], dtype=float)
    class_ids = np.array([3], dtype=int)
    result = arrays_to_detections(boxes, scores, class_ids)
    assert type(result[0]["class_id"]) is int, (
        f"Expected class_id to be plain int, got {type(result[0]['class_id'])}"
    )


def test_order_preserved():
    boxes = np.array([[0, 0, 1, 1], [2, 2, 3, 3], [4, 4, 5, 5]], dtype=float)
    scores = np.array([0.1, 0.9, 0.5], dtype=float)
    class_ids = np.array([2, 0, 1], dtype=int)
    result = arrays_to_detections(boxes, scores, class_ids)
    assert [d["score"] for d in result] == [0.1, 0.9, 0.5]
    assert [d["class_id"] for d in result] == [2, 0, 1]


def test_input_arrays_not_mutated():
    boxes = np.array([[0, 0, 10, 10], [1, 1, 5, 5]], dtype=float)
    scores = np.array([0.9, 0.5], dtype=float)
    class_ids = np.array([0, 1], dtype=int)
    boxes_copy = boxes.copy()
    scores_copy = scores.copy()
    class_ids_copy = class_ids.copy()
    arrays_to_detections(boxes, scores, class_ids)
    np.testing.assert_array_equal(boxes, boxes_copy)
    np.testing.assert_array_equal(scores, scores_copy)
    np.testing.assert_array_equal(class_ids, class_ids_copy)
