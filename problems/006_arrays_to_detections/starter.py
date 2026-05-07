import numpy as np


def arrays_to_detections(
    boxes: np.ndarray,
    scores: np.ndarray,
    class_ids: np.ndarray,
) -> list[dict]:
    """Convert parallel NumPy arrays into a list of detection dicts.

    Args:
        boxes: (N, 4) float array of bounding boxes in [x1, y1, x2, y2] format.
        scores: (N,) float array of confidence scores.
        class_ids: (N,) int array of class IDs.

    Returns:
        List of dicts with 'bbox' (list), 'score' (float), and 'class_id' (int).
    """
    raise NotImplementedError
