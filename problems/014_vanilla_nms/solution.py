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
    if len(boxes) == 0:
        return []

    order = np.argsort(-scores, kind="stable")
    kept = []

    while len(order) > 0:
        idx = order[0]
        kept.append(int(idx))
        order = order[1:]

        if len(order) == 0:
            break

        box = boxes[idx]
        remaining = boxes[order]

        inter_w = np.maximum(0.0, np.minimum(box[2], remaining[:, 2]) - np.maximum(box[0], remaining[:, 0]))
        inter_h = np.maximum(0.0, np.minimum(box[3], remaining[:, 3]) - np.maximum(box[1], remaining[:, 1]))
        intersection = inter_w * inter_h

        area_box = max(0.0, box[2] - box[0]) * max(0.0, box[3] - box[1])
        area_remaining = (
            np.maximum(0.0, remaining[:, 2] - remaining[:, 0])
            * np.maximum(0.0, remaining[:, 3] - remaining[:, 1])
        )
        union = area_box + area_remaining - intersection

        iou = np.divide(
            intersection,
            union,
            out=np.zeros_like(intersection, dtype=float),
            where=union != 0,
        )

        order = order[iou <= iou_threshold]

    return kept
