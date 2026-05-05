from typing import Any


def filter_detections_by_score(
    detections: list[dict[str, Any]],
    min_score: float,
) -> list[dict[str, Any]]:
    """Return detections with score >= min_score.

    Args:
        detections: list of dicts each containing at least {"score": float}.
        min_score: inclusive minimum score threshold.

    Returns:
        Filtered list in original order.
    """
    return [d for d in detections if d["score"] >= min_score]
