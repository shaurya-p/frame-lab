import numpy as np
import pytest
from solution import iou_two_boxes


def test_identical_boxes():
    box = np.array([0, 0, 10, 10], dtype=float)
    assert iou_two_boxes(box, box) == pytest.approx(1.0)


def test_partial_overlap():
    box_a = np.array([0, 0, 10, 10], dtype=float)
    box_b = np.array([5, 5, 15, 15], dtype=float)
    # intersection = 25, area_a = 100, area_b = 100, union = 175
    assert iou_two_boxes(box_a, box_b) == pytest.approx(25 / 175)


def test_no_overlap():
    box_a = np.array([0, 0, 10, 10], dtype=float)
    box_b = np.array([20, 20, 30, 30], dtype=float)
    assert iou_two_boxes(box_a, box_b) == pytest.approx(0.0)


def test_touching_edge_produces_zero():
    box_a = np.array([0, 0, 10, 10], dtype=float)
    box_b = np.array([10, 0, 20, 10], dtype=float)
    assert iou_two_boxes(box_a, box_b) == pytest.approx(0.0)


def test_touching_corner_produces_zero():
    box_a = np.array([0, 0, 10, 10], dtype=float)
    box_b = np.array([10, 10, 20, 20], dtype=float)
    assert iou_two_boxes(box_a, box_b) == pytest.approx(0.0)


def test_one_box_inside_another():
    box_a = np.array([0, 0, 10, 10], dtype=float)
    box_b = np.array([2, 2, 8, 8], dtype=float)
    # intersection = 36, area_a = 100, area_b = 36, union = 100
    assert iou_two_boxes(box_a, box_b) == pytest.approx(36 / 100)


def test_zero_area_box_produces_zero():
    box_a = np.array([0, 0, 10, 10], dtype=float)
    box_b = np.array([5, 5, 5, 5], dtype=float)
    assert iou_two_boxes(box_a, box_b) == pytest.approx(0.0)


def test_both_zero_area_produces_zero():
    box_a = np.array([5, 5, 5, 5], dtype=float)
    box_b = np.array([5, 5, 5, 5], dtype=float)
    assert iou_two_boxes(box_a, box_b) == pytest.approx(0.0)


def test_inverted_box_produces_zero():
    # Confirms behavioral contract: inverted box → IoU 0.0.
    # Note: the standard min/max intersection formula always gives 0 for inverted
    # boxes, so this test cannot detect a wrong internal area calculation. Direct
    # area clamping is enforced by Problem 010 box_area.
    box_a = np.array([0, 0, 10, 10], dtype=float)
    box_b = np.array([10, 10, 5, 5], dtype=float)
    assert iou_two_boxes(box_a, box_b) == pytest.approx(0.0)


def test_both_inverted_boxes_returns_zero():
    # Both inverted → area=0, union=0. Tests the union==0 guard specifically;
    # a missing guard would produce NaN (0/0) and fail pytest.approx(0.0).
    box_a = np.array([10, 10, 5, 5], dtype=float)
    box_b = np.array([10, 10, 5, 5], dtype=float)
    assert iou_two_boxes(box_a, box_b) == pytest.approx(0.0)


def test_float_coordinates():
    box_a = np.array([0.0, 0.0, 1.0, 1.0], dtype=float)
    box_b = np.array([0.5, 0.5, 1.5, 1.5], dtype=float)
    # intersection = 0.25, area_a = 1.0, area_b = 1.0, union = 1.75
    assert iou_two_boxes(box_a, box_b) == pytest.approx(0.25 / 1.75)


def test_input_not_mutated():
    box_a = np.array([0, 0, 10, 10], dtype=float)
    box_b = np.array([5, 5, 15, 15], dtype=float)
    a_copy, b_copy = box_a.copy(), box_b.copy()
    iou_two_boxes(box_a, box_b)
    np.testing.assert_array_equal(box_a, a_copy)
    np.testing.assert_array_equal(box_b, b_copy)


def test_return_type_is_float():
    box_a = np.array([0, 0, 10, 10], dtype=float)
    box_b = np.array([5, 5, 15, 15], dtype=float)
    result = iou_two_boxes(box_a, box_b)
    assert isinstance(result, float), f"Expected float, got {type(result)}"
