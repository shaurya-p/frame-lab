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
    a = boxes_a[:, None, :]  # (N, 1, 4)
    b = boxes_b[None, :, :]  # (1, M, 4)

    inter_w = np.maximum(0.0, np.minimum(a[:, :, 2], b[:, :, 2]) - np.maximum(a[:, :, 0], b[:, :, 0]))
    inter_h = np.maximum(0.0, np.minimum(a[:, :, 3], b[:, :, 3]) - np.maximum(a[:, :, 1], b[:, :, 1]))
    intersection = inter_w * inter_h  # (N, M)

    area_a = np.maximum(0.0, boxes_a[:, 2] - boxes_a[:, 0]) * np.maximum(0.0, boxes_a[:, 3] - boxes_a[:, 1])  # (N,)
    area_b = np.maximum(0.0, boxes_b[:, 2] - boxes_b[:, 0]) * np.maximum(0.0, boxes_b[:, 3] - boxes_b[:, 1])  # (M,)
    union = area_a[:, None] + area_b[None, :] - intersection  # (N, M)

    return np.divide(
        intersection,
        union,
        out=np.zeros_like(intersection, dtype=float),
        where=union != 0,
    )
