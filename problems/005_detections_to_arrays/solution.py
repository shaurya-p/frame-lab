import numpy as np


def detections_to_arrays(
    detections: list[dict],
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Convert a list of detection dicts to parallel NumPy arrays.

    Args:
        detections: List of dicts with 'bbox', 'score', and 'class_id' keys.

    Returns:
        Tuple of (boxes, scores, class_ids) where boxes is (N, 4) float,
        scores is (N,) float, and class_ids is (N,) int.
    """
    if not detections:
        return (
            np.empty((0, 4), dtype=float),
            np.array([], dtype=float),
            np.array([], dtype=int),
        )
    boxes = np.array([d["bbox"] for d in detections], dtype=float)
    scores = np.array([d["score"] for d in detections], dtype=float)
    class_ids = np.array([d["class_id"] for d in detections], dtype=int)
    return boxes, scores, class_ids
