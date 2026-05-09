import numpy as np


def greedy_iou_match(
    iou_matrix: np.ndarray,
    iou_threshold: float,
) -> list[tuple[int, int]]:
    """Greedily match tracks to detections using a precomputed IoU matrix.

    Args:
        iou_matrix: (N, M) array where entry [i, j] is IoU of track i and detection j.
        iou_threshold: Minimum IoU required to form a match.

    Returns:
        List of (track_index, detection_index) pairs in selection order.
    """
    raise NotImplementedError
