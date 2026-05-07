import numpy as np
import pytest
from solution import detections_to_arrays


def det(bbox: list, score: float, class_id: int) -> dict:
    return {"bbox": bbox, "score": score, "class_id": class_id}


def test_normal_multiple_detections():
    detections = [
        det([0, 0, 10, 10], 0.9, 0),
        det([5, 5, 15, 15], 0.6, 2),
        det([1, 1, 8, 8], 0.3, 1),
    ]
    boxes, scores, class_ids = detections_to_arrays(detections)
    np.testing.assert_allclose(
        boxes, [[0, 0, 10, 10], [5, 5, 15, 15], [1, 1, 8, 8]]
    )
    np.testing.assert_allclose(scores, [0.9, 0.6, 0.3])
    np.testing.assert_array_equal(class_ids, [0, 2, 1])


def test_empty_input_shapes():
    boxes, scores, class_ids = detections_to_arrays([])
    assert boxes.shape == (0, 4), f"Expected boxes shape (0, 4), got {boxes.shape}"
    assert scores.shape == (0,), f"Expected scores shape (0,), got {scores.shape}"
    assert class_ids.shape == (0,), f"Expected class_ids shape (0,), got {class_ids.shape}"


def test_empty_input_dtypes():
    boxes, scores, class_ids = detections_to_arrays([])
    assert np.issubdtype(boxes.dtype, np.floating), f"Expected float dtype for boxes, got {boxes.dtype}"
    assert np.issubdtype(scores.dtype, np.floating), f"Expected float dtype for scores, got {scores.dtype}"
    assert np.issubdtype(class_ids.dtype, np.integer), f"Expected int dtype for class_ids, got {class_ids.dtype}"


def test_single_detection():
    detections = [det([2, 3, 12, 13], 0.75, 5)]
    boxes, scores, class_ids = detections_to_arrays(detections)
    assert boxes.shape == (1, 4), f"Expected shape (1, 4), got {boxes.shape}"
    assert scores.shape == (1,), f"Expected shape (1,), got {scores.shape}"
    assert class_ids.shape == (1,), f"Expected shape (1,), got {class_ids.shape}"
    np.testing.assert_allclose(boxes[0], [2, 3, 12, 13])
    np.testing.assert_allclose(scores[0], 0.75)
    np.testing.assert_array_equal(class_ids[0], 5)


def test_order_preserved():
    detections = [det([0, 0, 1, 1], 0.1, 2), det([2, 2, 3, 3], 0.9, 0), det([4, 4, 5, 5], 0.5, 1)]
    boxes, scores, class_ids = detections_to_arrays(detections)
    np.testing.assert_allclose(scores, [0.1, 0.9, 0.5])
    np.testing.assert_array_equal(class_ids, [2, 0, 1])


def test_boxes_dtype_is_float():
    detections = [det([0, 0, 10, 10], 0.9, 0)]
    boxes, _, _ = detections_to_arrays(detections)
    assert np.issubdtype(boxes.dtype, np.floating), f"Expected float dtype, got {boxes.dtype}"


def test_scores_dtype_is_float():
    detections = [det([0, 0, 10, 10], 0.9, 0)]
    _, scores, _ = detections_to_arrays(detections)
    assert np.issubdtype(scores.dtype, np.floating), f"Expected float dtype, got {scores.dtype}"


def test_class_ids_dtype_is_int():
    detections = [det([0, 0, 10, 10], 0.9, 3)]
    _, _, class_ids = detections_to_arrays(detections)
    assert np.issubdtype(class_ids.dtype, np.integer), f"Expected int dtype, got {class_ids.dtype}"


def test_integer_bbox_values_produce_float_boxes():
    detections = [det([0, 0, 10, 10], 0.9, 0)]
    boxes, _, _ = detections_to_arrays(detections)
    assert np.issubdtype(boxes.dtype, np.floating), (
        f"bbox with int values should still produce float array, got {boxes.dtype}"
    )


def test_input_not_mutated():
    detections = [det([0, 0, 10, 10], 0.9, 0), det([1, 1, 5, 5], 0.5, 1)]
    originals = [d.copy() for d in detections]
    detections_to_arrays(detections)
    for original, after in zip(originals, detections):
        assert original == after, f"Detection was mutated: before={original}, after={after}"
