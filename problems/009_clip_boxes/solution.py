import numpy as np


def clip_boxes(boxes: np.ndarray, image_shape: tuple[int, int]) -> np.ndarray:
    """Clip boxes to image boundaries.

    Args:
        boxes: (N, 4) array of boxes in [x1, y1, x2, y2] format.
        image_shape: (height, width) of the image.

    Returns:
        (N, 4) array with coordinates clipped to [0, width] for x and [0, height] for y.
    """
    height, width = image_shape
    return np.stack(
        [
            np.clip(boxes[:, 0], 0, width),
            np.clip(boxes[:, 1], 0, height),
            np.clip(boxes[:, 2], 0, width),
            np.clip(boxes[:, 3], 0, height),
        ],
        axis=1,
    )
