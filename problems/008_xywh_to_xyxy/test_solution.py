import numpy as np
import pytest
from solution import xywh_to_xyxy


def test_normal_integer_boxes():
    boxes = np.array([[10, 20, 40, 60], [0, 0, 30, 40]], dtype=float)
    result = xywh_to_xyxy(boxes)
    np.testing.assert_allclose(result, [[10, 20, 50, 80], [0, 0, 30, 40]])


def test_float_boxes():
    boxes = np.array([[1.5, 2.5, 3.0, 4.0]], dtype=float)
    result = xywh_to_xyxy(boxes)
    np.testing.assert_allclose(result, [[1.5, 2.5, 4.5, 6.5]])


def test_empty_input_shape():
    result = xywh_to_xyxy(np.empty((0, 4), dtype=float))
    assert result.shape == (0, 4), f"Expected shape (0, 4), got {result.shape}"


def test_single_box():
    boxes = np.array([[5, 10, 20, 30]], dtype=float)
    result = xywh_to_xyxy(boxes)
    assert result.shape == (1, 4), f"Expected shape (1, 4), got {result.shape}"
    np.testing.assert_allclose(result, [[5, 10, 25, 40]])


def test_order_preserved():
    boxes = np.array([[0, 0, 10, 10], [5, 5, 10, 15], [2, 3, 6, 6]], dtype=float)
    result = xywh_to_xyxy(boxes)
    np.testing.assert_allclose(result[0], [0, 0, 10, 10])
    np.testing.assert_allclose(result[1], [5, 5, 15, 20])
    np.testing.assert_allclose(result[2], [2, 3, 8, 9])


def test_output_shape():
    boxes = np.array([[0, 0, 10, 10], [1, 2, 4, 6]], dtype=float)
    result = xywh_to_xyxy(boxes)
    assert result.shape == (2, 4), f"Expected shape (2, 4), got {result.shape}"


def test_zero_width_and_height():
    boxes = np.array([[5, 5, 0, 0]], dtype=float)
    result = xywh_to_xyxy(boxes)
    np.testing.assert_allclose(result, [[5, 5, 5, 5]])


def test_input_not_mutated():
    boxes = np.array([[0, 0, 10, 20], [5, 5, 10, 15]], dtype=float)
    original = boxes.copy()
    xywh_to_xyxy(boxes)
    np.testing.assert_array_equal(boxes, original)


def test_round_trip_consistency():
    xyxy = np.array([[10, 20, 50, 80], [0, 0, 30, 40], [3, 7, 9, 15]], dtype=float)
    xywh = np.stack(
        [xyxy[:, 0], xyxy[:, 1], xyxy[:, 2] - xyxy[:, 0], xyxy[:, 3] - xyxy[:, 1]],
        axis=1,
    )
    result = xywh_to_xyxy(xywh)
    np.testing.assert_allclose(result, xyxy)
