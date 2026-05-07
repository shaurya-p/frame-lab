import numpy as np
import pytest
from solution import xyxy_to_xywh


def test_normal_integer_boxes():
    boxes = np.array([[10, 20, 50, 80], [0, 0, 30, 40]], dtype=float)
    result = xyxy_to_xywh(boxes)
    np.testing.assert_allclose(result, [[10, 20, 40, 60], [0, 0, 30, 40]])


def test_float_boxes():
    boxes = np.array([[1.5, 2.5, 4.5, 6.5]], dtype=float)
    result = xyxy_to_xywh(boxes)
    np.testing.assert_allclose(result, [[1.5, 2.5, 3.0, 4.0]])


def test_empty_input_shape():
    result = xyxy_to_xywh(np.empty((0, 4), dtype=float))
    assert result.shape == (0, 4), f"Expected shape (0, 4), got {result.shape}"


def test_single_box():
    boxes = np.array([[5, 10, 25, 40]], dtype=float)
    result = xyxy_to_xywh(boxes)
    assert result.shape == (1, 4), f"Expected shape (1, 4), got {result.shape}"
    np.testing.assert_allclose(result, [[5, 10, 20, 30]])


def test_order_preserved():
    boxes = np.array([[0, 0, 10, 10], [5, 5, 15, 20], [2, 3, 8, 9]], dtype=float)
    result = xyxy_to_xywh(boxes)
    np.testing.assert_allclose(result[0], [0, 0, 10, 10])
    np.testing.assert_allclose(result[1], [5, 5, 10, 15])
    np.testing.assert_allclose(result[2], [2, 3, 6, 6])


def test_output_shape():
    boxes = np.array([[0, 0, 10, 10], [1, 2, 5, 8]], dtype=float)
    result = xyxy_to_xywh(boxes)
    assert result.shape == (2, 4), f"Expected shape (2, 4), got {result.shape}"


def test_square_box():
    boxes = np.array([[3, 3, 13, 13]], dtype=float)
    result = xyxy_to_xywh(boxes)
    np.testing.assert_allclose(result, [[3, 3, 10, 10]])
    assert result[0, 2] == result[0, 3], "w and h should be equal for a square box"


def test_input_not_mutated():
    boxes = np.array([[0, 0, 10, 20], [5, 5, 15, 25]], dtype=float)
    original = boxes.copy()
    xyxy_to_xywh(boxes)
    np.testing.assert_array_equal(boxes, original)
