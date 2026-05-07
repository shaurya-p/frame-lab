import numpy as np
import pytest
from starter import clip_boxes


def test_boxes_already_inside():
    boxes = np.array([[10, 10, 50, 50], [20, 30, 80, 90]], dtype=float)
    result = clip_boxes(boxes, image_shape=(100, 150))
    np.testing.assert_allclose(result, boxes)


def test_partially_outside():
    boxes = np.array([[-5, 10, 30, 50], [10, 20, 200, 300]], dtype=float)
    result = clip_boxes(boxes, image_shape=(100, 150))
    np.testing.assert_allclose(result, [[0, 10, 30, 50], [10, 20, 150, 100]])


def test_completely_outside():
    boxes = np.array([[-50, -50, -10, -10], [200, 200, 300, 300]], dtype=float)
    result = clip_boxes(boxes, image_shape=(100, 150))
    np.testing.assert_allclose(result, [[0, 0, 0, 0], [150, 100, 150, 100]])


def test_negative_coordinates_clipped_to_zero():
    boxes = np.array([[-20, -30, 40, 50]], dtype=float)
    result = clip_boxes(boxes, image_shape=(100, 100))
    np.testing.assert_allclose(result, [[0, 0, 40, 50]])


def test_coordinates_beyond_image_clipped_to_max():
    boxes = np.array([[0, 0, 500, 400]], dtype=float)
    result = clip_boxes(boxes, image_shape=(200, 300))
    np.testing.assert_allclose(result, [[0, 0, 300, 200]])


def test_float_coordinates():
    boxes = np.array([[-1.5, 2.5, 99.9, 150.7]], dtype=float)
    result = clip_boxes(boxes, image_shape=(100, 80))
    np.testing.assert_allclose(result, [[0.0, 2.5, 80.0, 100.0]])


def test_empty_input_shape():
    result = clip_boxes(np.empty((0, 4), dtype=float), image_shape=(100, 200))
    assert result.shape == (0, 4), f"Expected shape (0, 4), got {result.shape}"


def test_single_box():
    boxes = np.array([[5, 10, 25, 40]], dtype=float)
    result = clip_boxes(boxes, image_shape=(50, 30))
    assert result.shape == (1, 4)
    np.testing.assert_allclose(result, [[5, 10, 25, 40]])


def test_x_uses_width_y_uses_height():
    boxes = np.array([[0, 0, 999, 999]], dtype=float)
    result = clip_boxes(boxes, image_shape=(100, 200))
    np.testing.assert_allclose(result, [[0, 0, 200, 100]])


def test_input_not_mutated():
    boxes = np.array([[-5, -5, 200, 200], [10, 10, 50, 50]], dtype=float)
    original = boxes.copy()
    clip_boxes(boxes, image_shape=(100, 150))
    np.testing.assert_array_equal(boxes, original)
