import numpy as np
import pytest
from starter import iou_one_to_many


def test_identical_box():
    box = np.array([0, 0, 10, 10], dtype=float)
    boxes = np.array([[0, 0, 10, 10]], dtype=float)
    np.testing.assert_allclose(iou_one_to_many(box, boxes), [1.0])


def test_partial_overlap():
    box = np.array([0, 0, 10, 10], dtype=float)
    boxes = np.array([[5, 5, 15, 15]], dtype=float)
    np.testing.assert_allclose(iou_one_to_many(box, boxes), [25 / 175], rtol=1e-6)


def test_no_overlap():
    box = np.array([0, 0, 10, 10], dtype=float)
    boxes = np.array([[20, 20, 30, 30], [50, 50, 60, 60]], dtype=float)
    np.testing.assert_allclose(iou_one_to_many(box, boxes), [0.0, 0.0])


def test_touching_edge_produces_zero():
    box = np.array([0, 0, 10, 10], dtype=float)
    boxes = np.array([[10, 0, 20, 10]], dtype=float)
    np.testing.assert_allclose(iou_one_to_many(box, boxes), [0.0])


def test_candidate_fully_inside_query():
    box = np.array([0, 0, 10, 10], dtype=float)
    boxes = np.array([[2, 2, 8, 8]], dtype=float)
    np.testing.assert_allclose(iou_one_to_many(box, boxes), [36 / 100], rtol=1e-6)


def test_query_fully_inside_candidate():
    box = np.array([2, 2, 8, 8], dtype=float)
    boxes = np.array([[0, 0, 10, 10]], dtype=float)
    np.testing.assert_allclose(iou_one_to_many(box, boxes), [36 / 100], rtol=1e-6)


def test_mixed_results():
    box = np.array([0, 0, 10, 10], dtype=float)
    boxes = np.array([
        [0,  0, 10, 10],
        [5,  5, 15, 15],
        [20, 20, 30, 30],
    ], dtype=float)
    expected = [1.0, 25 / 175, 0.0]
    np.testing.assert_allclose(iou_one_to_many(box, boxes), expected, rtol=1e-6)


def test_zero_area_query_box():
    box = np.array([5, 5, 5, 5], dtype=float)
    boxes = np.array([[0, 0, 10, 10], [5, 5, 15, 15]], dtype=float)
    np.testing.assert_allclose(iou_one_to_many(box, boxes), [0.0, 0.0])


def test_zero_area_candidate():
    box = np.array([0, 0, 10, 10], dtype=float)
    boxes = np.array([[5, 5, 5, 5]], dtype=float)
    np.testing.assert_allclose(iou_one_to_many(box, boxes), [0.0])


def test_inverted_candidate_produces_zero():
    box = np.array([0, 0, 10, 10], dtype=float)
    boxes = np.array([[10, 10, 5, 5]], dtype=float)
    np.testing.assert_allclose(iou_one_to_many(box, boxes), [0.0])


def test_inverted_query_produces_zero():
    box = np.array([10, 10, 5, 5], dtype=float)
    boxes = np.array([[0, 0, 10, 10]], dtype=float)
    np.testing.assert_allclose(iou_one_to_many(box, boxes), [0.0])


def test_empty_boxes_shape():
    box = np.array([0, 0, 10, 10], dtype=float)
    result = iou_one_to_many(box, np.empty((0, 4), dtype=float))
    assert result.shape == (0,), f"Expected shape (0,), got {result.shape}"


def test_float_coordinates():
    box = np.array([0.0, 0.0, 1.0, 1.0], dtype=float)
    boxes = np.array([[0.5, 0.5, 1.5, 1.5]], dtype=float)
    np.testing.assert_allclose(iou_one_to_many(box, boxes), [0.25 / 1.75], rtol=1e-6)


def test_order_preserved():
    box = np.array([0, 0, 10, 10], dtype=float)
    boxes = np.array([
        [20, 20, 30, 30],
        [0,  0, 10, 10],
        [5,  5, 15, 15],
    ], dtype=float)
    result = iou_one_to_many(box, boxes)
    np.testing.assert_allclose(result[0], 0.0)
    np.testing.assert_allclose(result[1], 1.0)
    np.testing.assert_allclose(result[2], 25 / 175, rtol=1e-6)


def test_output_dtype_is_float():
    box = np.array([0, 0, 10, 10], dtype=float)
    boxes = np.array([[0, 0, 10, 10]], dtype=float)
    result = iou_one_to_many(box, boxes)
    assert np.issubdtype(result.dtype, np.floating), f"Expected float dtype, got {result.dtype}"


def test_input_not_mutated():
    box = np.array([0, 0, 10, 10], dtype=float)
    boxes = np.array([[5, 5, 15, 15], [20, 20, 30, 30]], dtype=float)
    box_copy, boxes_copy = box.copy(), boxes.copy()
    iou_one_to_many(box, boxes)
    np.testing.assert_array_equal(box, box_copy)
    np.testing.assert_array_equal(boxes, boxes_copy)
