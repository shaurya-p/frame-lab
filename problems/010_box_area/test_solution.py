import numpy as np
import pytest
from solution import box_area


def test_normal_boxes():
    boxes = np.array([[0, 0, 10, 20], [5, 5, 15, 10]], dtype=float)
    result = box_area(boxes)
    np.testing.assert_allclose(result, [200.0, 50.0])


def test_float_boxes():
    boxes = np.array([[0.5, 1.5, 4.5, 5.5]], dtype=float)
    result = box_area(boxes)
    np.testing.assert_allclose(result, [16.0])


def test_zero_area_point():
    boxes = np.array([[3, 3, 3, 3]], dtype=float)
    result = box_area(boxes)
    np.testing.assert_allclose(result, [0.0])


def test_zero_area_horizontal_line():
    boxes = np.array([[0, 5, 10, 5]], dtype=float)
    result = box_area(boxes)
    np.testing.assert_allclose(result, [0.0])


def test_zero_area_vertical_line():
    boxes = np.array([[5, 0, 5, 10]], dtype=float)
    result = box_area(boxes)
    np.testing.assert_allclose(result, [0.0])


def test_inverted_box_produces_zero():
    boxes = np.array([[10, 10, 5, 5]], dtype=float)
    result = box_area(boxes)
    np.testing.assert_allclose(result, [0.0])


def test_inverted_width_only():
    boxes = np.array([[10, 0, 5, 20]], dtype=float)
    result = box_area(boxes)
    np.testing.assert_allclose(result, [0.0])


def test_empty_input_shape():
    result = box_area(np.empty((0, 4), dtype=float))
    assert result.shape == (0,), f"Expected shape (0,), got {result.shape}"


def test_output_shape():
    boxes = np.array([[0, 0, 5, 5], [1, 1, 4, 4], [2, 2, 6, 8]], dtype=float)
    result = box_area(boxes)
    assert result.shape == (3,), f"Expected shape (3,), got {result.shape}"


def test_order_preserved():
    boxes = np.array([[0, 0, 2, 3], [0, 0, 5, 5], [0, 0, 1, 10]], dtype=float)
    result = box_area(boxes)
    np.testing.assert_allclose(result, [6.0, 25.0, 10.0])


def test_input_not_mutated():
    boxes = np.array([[0, 0, 10, 20], [5, 5, 15, 10]], dtype=float)
    original = boxes.copy()
    box_area(boxes)
    np.testing.assert_array_equal(boxes, original)
