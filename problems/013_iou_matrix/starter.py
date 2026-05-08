import numpy as np


def iou_matrix(boxes_a: np.ndarray, boxes_b: np.ndarray) -> np.ndarray:
    """Compute pairwise IoU between two sets of boxes.

    Args:
        boxes_a: (N, 4) array in [x1, y1, x2, y2] format.
        boxes_b: (M, 4) array in [x1, y1, x2, y2] format.

    Returns:
        (N, M) float array where output[i, j] is IoU of boxes_a[i] and boxes_b[j].
        Elements are 0.0 where union is zero.
    """
    raise NotImplementedError
