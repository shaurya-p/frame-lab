import numpy as np


def xywh_to_xyxy(boxes: np.ndarray) -> np.ndarray:
    """Convert boxes from [x, y, w, h] to [x1, y1, x2, y2] format.

    Args:
        boxes: (N, 4) array of boxes in [x, y, w, h] format.

    Returns:
        (N, 4) array of boxes in [x1, y1, x2, y2] format.
    """
    raise NotImplementedError
