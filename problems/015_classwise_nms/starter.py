import numpy as np


def classwise_nms(
    boxes: np.ndarray,
    scores: np.ndarray,
    class_ids: np.ndarray,
    iou_threshold: float,
) -> list[int]:
    """Apply NMS independently per class and return kept original indices.

    Args:
        boxes: (N, 4) array of boxes in [x1, y1, x2, y2] format.
        scores: (N,) array of confidence scores.
        class_ids: (N,) array of integer class labels.
        iou_threshold: Boxes with IoU strictly greater than this are suppressed.

    Returns:
        List of original input indices of kept boxes, sorted by descending score.
        Ties in score preserve original input order.
    """
    raise NotImplementedError
