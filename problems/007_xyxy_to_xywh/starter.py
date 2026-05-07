import numpy as np


def xyxy_to_xywh(boxes: np.ndarray) -> np.ndarray:
    """Convert boxes from [x1, y1, x2, y2] to [x, y, w, h] format.

    Args:
        boxes: (N, 4) array of boxes in [x1, y1, x2, y2] format.

    Returns:
        (N, 4) array of boxes in [x, y, w, h] format.
    """
    raise NotImplementedError
