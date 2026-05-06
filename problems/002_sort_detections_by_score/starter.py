def sort_detections_by_score(
    detections: list[dict],
    descending: bool = True,
) -> list[dict]:
    """Sort detections by score.

    Args:
        detections: List of dicts with 'bbox', 'score', and 'class_id' keys.
        descending: If True, highest score first. If False, lowest score first.

    Returns:
        New list of detection dicts sorted by score.
    """

    raise NotImplementedError
