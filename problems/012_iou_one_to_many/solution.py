import numpy as np


def iou_one_to_many(box: np.ndarray, boxes: np.ndarray) -> np.ndarray:
    """Compute IoU between a single query box and each box in an array.

    Args:
        box: (4,) array in [x1, y1, x2, y2] format.
        boxes: (N, 4) array in [x1, y1, x2, y2] format.

    Returns:
        (N,) float array of IoU values. Elements are 0.0 where union is zero.
    """
    query_box_area = max(0.0, box[2] - box[0]) * max(0.0, box[3] - box[1])
    boxes_area = np.maximum(0.0, boxes[:, 2] - boxes[:, 0]) * np.maximum(0.0, boxes[:, 3] - boxes[:, 1])

    inter_w = np.maximum(
        0.0,
        np.minimum(boxes[:,2], box[2]) - np.maximum(boxes[:, 0], box[0])
    )
    inter_h = np.maximum(
        0.0,
        np.minimum(boxes[:, 3], box[3]) - np.maximum(boxes[:, 1], box[1])
    )

    inter = inter_w * inter_h
    union = boxes_area + query_box_area - inter

    return np.divide(
        inter,
        union,
        out=np.zeros_like(inter, dtype=float),
        where=union!=0
    )