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
    raise NotImplementedError
