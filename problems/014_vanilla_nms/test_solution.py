import numpy as np
import pytest
from solution import vanilla_nms


def test_empty_input():
    result = vanilla_nms(np.empty((0, 4), dtype=float), np.array([]), iou_threshold=0.5)
    assert result == []


def test_single_box():
    boxes = np.array([[0, 0, 10, 10]], dtype=float)
    scores = np.array([0.9])
    assert vanilla_nms(boxes, scores, iou_threshold=0.5) == [0]


def test_non_overlapping_all_kept():
    boxes = np.array([
        [0,  0, 10, 10],
        [20, 0, 30, 10],
        [40, 0, 50, 10],
    ], dtype=float)
    scores = np.array([0.9, 0.7, 0.5])
    result = vanilla_nms(boxes, scores, iou_threshold=0.5)
    assert result == [0, 1, 2]


def test_overlapping_suppresses_lower_score():
    boxes = np.array([
        [0, 0, 10, 10],  # index 0, score 0.9 — kept
        [1, 1, 11, 11],  # index 1, score 0.8 — high overlap, suppressed
    ], dtype=float)
    scores = np.array([0.9, 0.8])
    result = vanilla_nms(boxes, scores, iou_threshold=0.5)
    assert result == [0]


def test_higher_score_selected_first():
    boxes = np.array([
        [0,  0, 10, 10],  # index 0, score 0.5
        [0,  0, 10, 10],  # index 1, score 0.9 — selected first
    ], dtype=float)
    scores = np.array([0.5, 0.9])
    result = vanilla_nms(boxes, scores, iou_threshold=0.5)
    assert result == [1]


def test_chain_suppression():
    boxes = np.array([
        [0,  0, 10, 10],   # index 0, score 0.9 — kept
        [1,  1, 11, 11],   # index 1, score 0.8 — high overlap with 0, suppressed
        [20, 20, 30, 30],  # index 2, score 0.7 — no overlap with 0, kept
    ], dtype=float)
    scores = np.array([0.9, 0.8, 0.7])
    result = vanilla_nms(boxes, scores, iou_threshold=0.5)
    assert result == [0, 2]


def test_iou_equal_to_threshold_not_suppressed():
    # boxes_a[0] and boxes_b[0]: intersection=25, union=175, IoU=25/175
    boxes = np.array([
        [0, 0, 10, 10],   # index 0, score 0.9
        [5, 5, 15, 15],   # index 1, score 0.5 — IoU = 25/175 with box 0
    ], dtype=float)
    scores = np.array([0.9, 0.5])
    threshold = 25 / 175
    result = vanilla_nms(boxes, scores, iou_threshold=threshold)
    assert result == [0, 1], f"Box at IoU==threshold should not be suppressed, got {result}"


def test_equal_score_preserves_original_order():
    boxes = np.array([
        [0,  0,  5,  5],   # index 0, score 0.5 — no overlap with 1
        [10, 10, 15, 15],  # index 1, score 0.5 — no overlap with 0
    ], dtype=float)
    scores = np.array([0.5, 0.5])
    result = vanilla_nms(boxes, scores, iou_threshold=0.5)
    assert result == [0, 1], f"Equal scores should preserve original order, got {result}"


def test_equal_score_tie_lower_index_selected_first():
    boxes = np.array([
        [0, 0, 10, 10],  # index 0, score 0.5 — identical to index 1
        [0, 0, 10, 10],  # index 1, score 0.5
    ], dtype=float)
    scores = np.array([0.5, 0.5])
    result = vanilla_nms(boxes, scores, iou_threshold=0.5)
    assert result == [0], f"Lower original index should win the tie, got {result}"


def test_returned_indices_are_original():
    # Highest score is at index 1, not index 0
    boxes = np.array([
        [20, 20, 30, 30],  # index 0, score 0.5
        [0,  0, 10, 10],   # index 1, score 0.9 — selected first
        [1,  1, 11, 11],   # index 2, score 0.8 — suppressed by index 1
    ], dtype=float)
    scores = np.array([0.5, 0.9, 0.8])
    result = vanilla_nms(boxes, scores, iou_threshold=0.5)
    assert result == [1, 0], f"Expected original indices [1, 0], got {result}"


def test_output_is_list_of_python_ints():
    boxes = np.array([[0, 0, 10, 10], [20, 20, 30, 30]], dtype=float)
    scores = np.array([0.9, 0.5])
    result = vanilla_nms(boxes, scores, iou_threshold=0.5)
    assert isinstance(result, list), f"Expected list, got {type(result)}"
    for item in result:
        assert type(item) is int, f"Expected Python int, got {type(item)}"


def test_input_not_mutated():
    boxes = np.array([[0, 0, 10, 10], [1, 1, 11, 11]], dtype=float)
    scores = np.array([0.9, 0.8])
    boxes_copy, scores_copy = boxes.copy(), scores.copy()
    vanilla_nms(boxes, scores, iou_threshold=0.5)
    np.testing.assert_array_equal(boxes, boxes_copy)
    np.testing.assert_array_equal(scores, scores_copy)
