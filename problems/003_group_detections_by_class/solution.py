def group_detections_by_class(detections: list[dict]) -> dict[int, list[dict]]:
    """Group detections by class_id.

    Args:
        detections: List of dicts with 'bbox', 'score', and 'class_id' keys.

    Returns:
        Dict mapping each class_id to its list of detections in original order.
    """
    grouped_detections: dict[int, list[dict]] = {}

    for detection in detections:
        grouped_detections.setdefault(detection["class_id"], []).append(detection)
    return grouped_detections
