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
    if iou_matrix.size == 0:
        return []

    rows, cols = np.where(iou_matrix >= iou_threshold)
    if len(rows) == 0:
        return []

    ious = iou_matrix[rows, cols]
    # Primary: descending IoU; secondary: ascending track index; tertiary: ascending detection index
    order = np.lexsort((cols, rows, -ious))

    matched_tracks: set[int] = set()
    matched_detections: set[int] = set()
    matches: list[tuple[int, int]] = []

    for idx in order:
        track_idx = int(rows[idx])
        det_idx = int(cols[idx])
        if track_idx in matched_tracks or det_idx in matched_detections:
            continue
        matches.append((track_idx, det_idx))
        matched_tracks.add(track_idx)
        matched_detections.add(det_idx)

    return matches
