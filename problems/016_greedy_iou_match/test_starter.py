import numpy as np
import pytest
from starter import greedy_iou_match


def test_simple_one_to_one():
    iou = np.array([[0.8]], dtype=float)
    assert greedy_iou_match(iou, iou_threshold=0.5) == [(0, 0)]


def test_multi_track_multi_detection_diagonal():
    iou = np.array([
        [0.9, 0.1],
        [0.2, 0.8],
    ], dtype=float)
    assert greedy_iou_match(iou, iou_threshold=0.5) == [(0, 0), (1, 1)]


def test_below_threshold_not_matched():
    iou = np.array([[0.3]], dtype=float)
    assert greedy_iou_match(iou, iou_threshold=0.5) == []


def test_no_valid_matches():
    iou = np.array([[0.1, 0.2], [0.3, 0.4]], dtype=float)
    assert greedy_iou_match(iou, iou_threshold=0.5) == []


def test_empty_rows():
    iou = np.empty((0, 3), dtype=float)
    assert greedy_iou_match(iou, iou_threshold=0.5) == []


def test_empty_cols():
    iou = np.empty((3, 0), dtype=float)
    assert greedy_iou_match(iou, iou_threshold=0.5) == []


def test_both_empty():
    iou = np.empty((0, 0), dtype=float)
    assert greedy_iou_match(iou, iou_threshold=0.5) == []


def test_competing_detections_for_one_track():
    iou = np.array([[0.9, 0.7]], dtype=float)
    result = greedy_iou_match(iou, iou_threshold=0.5)
    assert result == [(0, 0)], f"Expected [(0, 0)], got {result}"


def test_competing_tracks_for_one_detection():
    iou = np.array([[0.9], [0.7]], dtype=float)
    result = greedy_iou_match(iou, iou_threshold=0.5)
    assert result == [(0, 0)], f"Expected [(0, 0)], got {result}"


def test_greedy_propagates_second_best():
    iou = np.array([
        [0.9, 0.8],
        [0.7, 0.6],
    ], dtype=float)
    result = greedy_iou_match(iou, iou_threshold=0.5)
    assert result == [(0, 0), (1, 1)], f"Expected [(0, 0), (1, 1)], got {result}"


def test_tiebreak_by_lower_track_index():
    iou = np.array([[0.8], [0.8]], dtype=float)
    result = greedy_iou_match(iou, iou_threshold=0.5)
    assert result == [(0, 0)], f"Expected [(0, 0)], got {result}"


def test_tiebreak_by_lower_detection_index():
    iou = np.array([[0.8, 0.8]], dtype=float)
    result = greedy_iou_match(iou, iou_threshold=0.5)
    assert result == [(0, 0)], f"Expected [(0, 0)], got {result}"


def test_threshold_is_inclusive():
    iou = np.array([[0.5]], dtype=float)
    assert greedy_iou_match(iou, iou_threshold=0.5) == [(0, 0)]


def test_output_is_list_of_tuples_of_python_ints():
    iou = np.array([[0.9, 0.1], [0.2, 0.8]], dtype=float)
    result = greedy_iou_match(iou, iou_threshold=0.5)
    assert isinstance(result, list), f"Expected list, got {type(result)}"
    for pair in result:
        assert isinstance(pair, tuple), f"Expected tuple, got {type(pair)}"
        assert type(pair[0]) is int and type(pair[1]) is int, (
            f"Expected Python ints, got {type(pair[0])}, {type(pair[1])}"
        )


def test_input_not_mutated():
    iou = np.array([[0.9, 0.1], [0.2, 0.8]], dtype=float)
    iou_copy = iou.copy()
    greedy_iou_match(iou, iou_threshold=0.5)
    np.testing.assert_array_equal(iou, iou_copy)
