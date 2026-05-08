import numpy as np


def iou_two_boxes(box_a: np.ndarray, box_b: np.ndarray) -> float:
    """Compute Intersection over Union between two boxes.

    Args:
        box_a: (4,) array in [x1, y1, x2, y2] format.
        box_b: (4,) array in [x1, y1, x2, y2] format.

    Returns:
        IoU as a float in [0.0, 1.0]. Returns 0.0 if union is zero.
    """
    def area(box: np.ndarray) -> float:
        return (box[2] - box[0]) * (box[3] - box[1])

    def intersection(box_a: np.ndarray, box_b: np.ndarray) -> float:
        inter_w = max(0.0, min(box_a[2], box_b[2]) - max(box_a[0], box_b[0]))
        inter_h = max(0.0, min(box_a[3], box_b[3]) - max(box_a[1], box_b[1]))

        return inter_h * inter_w

    area_a = area(box_a)
    area_b = area(box_b)
    inter = intersection(box_a, box_b)
    union = area_a + area_b - inter

    if union == 0:
        return 0.0
    return inter/union

    raise NotImplementedError
