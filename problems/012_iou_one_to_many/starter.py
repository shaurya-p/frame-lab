import numpy as np


def iou_one_to_many(box: np.ndarray, boxes: np.ndarray) -> np.ndarray:
    """Compute IoU between a single query box and each box in an array.

    Args:
        box: (4,) array in [x1, y1, x2, y2] format.
        boxes: (N, 4) array in [x1, y1, x2, y2] format.

    Returns:
        (N,) float array of IoU values. Elements are 0.0 where union is zero.
    """
    raise NotImplementedError
