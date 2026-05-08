import numpy as np


def iou_two_boxes(box_a: np.ndarray, box_b: np.ndarray) -> float:
    """Compute Intersection over Union between two boxes.

    Args:
        box_a: (4,) array in [x1, y1, x2, y2] format.
        box_b: (4,) array in [x1, y1, x2, y2] format.

    Returns:
        IoU as a float in [0.0, 1.0]. Returns 0.0 if union is zero.
    """
    raise NotImplementedError
