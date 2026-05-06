def delete_stale_tracks(tracks: dict[int, dict], max_age: int) -> dict[int, dict]:
    """Remove tracks whose age exceeds max_age.

    Args:
        tracks: Dict mapping track ID to a track dict with 'bbox', 'age', and 'hits'.
        max_age: Maximum allowed age. Tracks with age > max_age are removed.

    Returns:
        New dict containing only tracks with age <= max_age.
    """
    return {tid: track for tid, track in tracks.items() if track["age"] <= max_age}
