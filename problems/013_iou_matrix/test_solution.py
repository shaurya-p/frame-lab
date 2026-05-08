import numpy as np
import pytest
from solution import iou_matrix


def test_known_2x2():
    boxes_a = np.array([[0, 0, 10, 10], [5, 5, 15, 15]], dtype=float)
    boxes_b = np.array([[0, 0, 10, 10], [20, 20, 30, 30]], dtype=float)
    result = iou_matrix(boxes_a, boxes_b)
    assert result.shape == (2, 2)
    np.testing.assert_allclose(result[0, 0], 1.0)
    np.testing.assert_allclose(result[0, 1], 0.0)
    np.testing.assert_allclose(result[1, 0], 25 / 175, rtol=1e-6)
    np.testing.assert_allclose(result[1, 1], 0.0)


def test_diagonal_is_ones_for_identical_inputs():
    boxes = np.array([[0, 0, 10, 10], [5, 5, 20, 20], [1, 1, 4, 4]], dtype=float)
    result = iou_matrix(boxes, boxes)
    assert result.shape == (3, 3)
    np.testing.assert_allclose(np.diag(result), [1.0, 1.0, 1.0])


def test_no_overlap():
    boxes_a = np.array([[0, 0, 5, 5]], dtype=float)
    boxes_b = np.array([[10, 10, 20, 20], [30, 30, 40, 40]], dtype=float)
    result = iou_matrix(boxes_a, boxes_b)
    np.testing.assert_allclose(result, [[0.0, 0.0]])


def test_touching_edge_produces_zero():
    boxes_a = np.array([[0, 0, 10, 10]], dtype=float)
    boxes_b = np.array([[10, 0, 20, 10]], dtype=float)
    result = iou_matrix(boxes_a, boxes_b)
    np.testing.assert_allclose(result, [[0.0]])


def test_partial_overlap():
    boxes_a = np.array([[0, 0, 10, 10]], dtype=float)
    boxes_b = np.array([[5, 5, 15, 15]], dtype=float)
    result = iou_matrix(boxes_a, boxes_b)
    np.testing.assert_allclose(result, [[25 / 175]], rtol=1e-6)


def test_one_fully_inside_other():
    boxes_a = np.array([[0, 0, 10, 10]], dtype=float)
    boxes_b = np.array([[2, 2, 8, 8]], dtype=float)
    result = iou_matrix(boxes_a, boxes_b)
    np.testing.assert_allclose(result, [[36 / 100]], rtol=1e-6)


def test_zero_area_box_in_a():
    boxes_a = np.array([[5, 5, 5, 5]], dtype=float)
    boxes_b = np.array([[0, 0, 10, 10], [5, 5, 15, 15]], dtype=float)
    result = iou_matrix(boxes_a, boxes_b)
    np.testing.assert_allclose(result, [[0.0, 0.0]])


def test_zero_area_box_in_b():
    boxes_a = np.array([[0, 0, 10, 10], [5, 5, 15, 15]], dtype=float)
    boxes_b = np.array([[5, 5, 5, 5]], dtype=float)
    result = iou_matrix(boxes_a, boxes_b)
    np.testing.assert_allclose(result, [[0.0], [0.0]])


def test_inverted_box_in_a():
    boxes_a = np.array([[10, 10, 5, 5]], dtype=float)
    boxes_b = np.array([[0, 0, 10, 10]], dtype=float)
    result = iou_matrix(boxes_a, boxes_b)
    np.testing.assert_allclose(result, [[0.0]])


def test_inverted_box_in_b():
    boxes_a = np.array([[0, 0, 10, 10]], dtype=float)
    boxes_b = np.array([[10, 10, 5, 5]], dtype=float)
    result = iou_matrix(boxes_a, boxes_b)
    np.testing.assert_allclose(result, [[0.0]])


def test_float_coordinates():
    boxes_a = np.array([[0.0, 0.0, 1.0, 1.0]], dtype=float)
    boxes_b = np.array([[0.5, 0.5, 1.5, 1.5]], dtype=float)
    result = iou_matrix(boxes_a, boxes_b)
    np.testing.assert_allclose(result, [[0.25 / 1.75]], rtol=1e-6)


def test_empty_boxes_a():
    boxes_a = np.empty((0, 4), dtype=float)
    boxes_b = np.array([[0, 0, 10, 10], [5, 5, 15, 15]], dtype=float)
    result = iou_matrix(boxes_a, boxes_b)
    assert result.shape == (0, 2), f"Expected shape (0, 2), got {result.shape}"


def test_empty_boxes_b():
    boxes_a = np.array([[0, 0, 10, 10], [5, 5, 15, 15]], dtype=float)
    boxes_b = np.empty((0, 4), dtype=float)
    result = iou_matrix(boxes_a, boxes_b)
    assert result.shape == (2, 0), f"Expected shape (2, 0), got {result.shape}"


def test_both_empty():
    result = iou_matrix(np.empty((0, 4), dtype=float), np.empty((0, 4), dtype=float))
    assert result.shape == (0, 0), f"Expected shape (0, 0), got {result.shape}"


def test_output_shape():
    boxes_a = np.array([[0, 0, 5, 5], [1, 1, 4, 4], [2, 2, 6, 6]], dtype=float)
    boxes_b = np.array([[0, 0, 5, 5], [3, 3, 8, 8]], dtype=float)
    result = iou_matrix(boxes_a, boxes_b)
    assert result.shape == (3, 2), f"Expected shape (3, 2), got {result.shape}"


def test_output_dtype_is_float():
    boxes_a = np.array([[0, 0, 10, 10]], dtype=float)
    boxes_b = np.array([[0, 0, 10, 10]], dtype=float)
    result = iou_matrix(boxes_a, boxes_b)
    assert np.issubdtype(result.dtype, np.floating), f"Expected float dtype, got {result.dtype}"


def test_input_not_mutated():
    boxes_a = np.array([[0, 0, 10, 10], [5, 5, 15, 15]], dtype=float)
    boxes_b = np.array([[0, 0, 10, 10], [20, 20, 30, 30]], dtype=float)
    a_copy, b_copy = boxes_a.copy(), boxes_b.copy()
    iou_matrix(boxes_a, boxes_b)
    np.testing.assert_array_equal(boxes_a, a_copy)
    np.testing.assert_array_equal(boxes_b, b_copy)
