import numpy as np


def vanilla_nms(
    boxes: np.ndarray,
    scores: np.ndarray,
    iou_threshold: float,
) -> list[int]:
    """Apply greedy Non-Maximum Suppression.

    Args:
        boxes: (N, 4) array of boxes in [x1, y1, x2, y2] format.
        scores: (N,) array of confidence scores.
        iou_threshold: Boxes with IoU strictly greater than this are suppressed.

    Returns:
        List of original input indices of kept boxes, in selection order.
    """
    raise NotImplementedError
