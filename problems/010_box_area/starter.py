import numpy as np


def box_area(boxes: np.ndarray) -> np.ndarray:
    """Compute the area of each box.

    Args:
        boxes: (N, 4) array of boxes in [x1, y1, x2, y2] format.

    Returns:
        (N,) array of areas. Inverted or degenerate boxes produce area 0.
    """
    raise NotImplementedError
