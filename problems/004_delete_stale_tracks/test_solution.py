import pytest
from solution import delete_stale_tracks


def track(age: int, hits: int = 1) -> dict:
    return {"bbox": [0, 0, 10, 10], "age": age, "hits": hits}


def test_normal_some_removed():
    tracks = {1: track(1), 2: track(4), 3: track(2)}
    result = delete_stale_tracks(tracks, max_age=2)
    assert set(result.keys()) == {1, 3}, (
        f"Expected track IDs {{1, 3}}, got {set(result.keys())}"
    )


def test_empty_input():
    result = delete_stale_tracks({}, max_age=5)
    assert result == {}, f"Expected {{}}, got {result}"


def test_all_kept():
    tracks = {1: track(1), 2: track(2), 3: track(3)}
    result = delete_stale_tracks(tracks, max_age=5)
    assert set(result.keys()) == {1, 2, 3}, (
        f"Expected all track IDs {{1, 2, 3}}, got {set(result.keys())}"
    )


def test_all_removed():
    tracks = {1: track(3), 2: track(5)}
    result = delete_stale_tracks(tracks, max_age=2)
    assert result == {}, f"Expected {{}}, got {result}"


def test_age_equal_to_max_age_is_kept():
    tracks = {1: track(2), 2: track(3)}
    result = delete_stale_tracks(tracks, max_age=2)
    assert 1 in result, "Track with age == max_age should be kept"
    assert 2 not in result, "Track with age > max_age should be removed"


def test_max_age_zero():
    tracks = {1: track(0), 2: track(1), 3: track(0)}
    result = delete_stale_tracks(tracks, max_age=0)
    assert set(result.keys()) == {1, 3}, (
        f"Expected only age-0 tracks {{1, 3}}, got {set(result.keys())}"
    )


def test_non_contiguous_track_ids():
    tracks = {10: track(1), 42: track(5), 99: track(2)}
    result = delete_stale_tracks(tracks, max_age=3)
    assert set(result.keys()) == {10, 99}, (
        f"Expected {{10, 99}}, got {set(result.keys())}"
    )


def test_input_dict_not_mutated():
    tracks = {1: track(1), 2: track(4)}
    original_keys = set(tracks.keys())
    delete_stale_tracks(tracks, max_age=2)
    assert set(tracks.keys()) == original_keys, (
        f"Input dict was mutated: expected keys {original_keys}, got {set(tracks.keys())}"
    )


def test_nested_track_dicts_not_mutated():
    tracks = {1: track(1, hits=7), 2: track(2, hits=3)}
    originals = {tid: t.copy() for tid, t in tracks.items()}
    delete_stale_tracks(tracks, max_age=5)
    for tid, original in originals.items():
        assert tracks[tid] == original, (
            f"Track {tid} was mutated: before={original}, after={tracks[tid]}"
        )


def test_output_is_plain_dict():
    tracks = {1: track(1)}
    result = delete_stale_tracks(tracks, max_age=5)
    assert type(result) is dict, f"Expected plain dict, got {type(result)}"
