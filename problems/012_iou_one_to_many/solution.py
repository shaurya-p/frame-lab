import numpy as np


def iou_one_to_many(box: np.ndarray, boxes: np.ndarray) -> np.ndarray:
    """Compute IoU between a single query box and each box in an array.

    Args:
        box: (4,) array in [x1, y1, x2, y2] format.
        boxes: (N, 4) array in [x1, y1, x2, y2] format.

    Returns:
        (N,) float array of IoU values. Elements are 0.0 where union is zero.
    """
    inter_w = np.clip(np.minimum(box[2], boxes[:, 2]) - np.maximum(box[0], boxes[:, 0]), 0, None)
    inter_h = np.clip(np.minimum(box[3], boxes[:, 3]) - np.maximum(box[1], boxes[:, 1]), 0, None)
    intersection = inter_w * inter_h

    area_box = max(0.0, box[2] - box[0]) * max(0.0, box[3] - box[1])
    area_boxes = np.clip(boxes[:, 2] - boxes[:, 0], 0, None) * np.clip(boxes[:, 3] - boxes[:, 1], 0, None)
    union = area_box + area_boxes - intersection

    return np.where(union == 0, 0.0, intersection / np.where(union == 0, 1.0, union))
