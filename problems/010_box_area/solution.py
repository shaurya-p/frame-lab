import numpy as np


def box_area(boxes: np.ndarray) -> np.ndarray:
    """Compute the area of each box.

    Args:
        boxes: (N, 4) array of boxes in [x1, y1, x2, y2] format.

    Returns:
        (N,) array of areas. Inverted or degenerate boxes produce area 0.
    """
    w = np.clip(boxes[:, 2] - boxes[:, 0], 0, None)
    h = np.clip(boxes[:, 3] - boxes[:, 1], 0, None)
    return w * h
