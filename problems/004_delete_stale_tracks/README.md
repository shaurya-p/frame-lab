# 004 — Delete Stale Tracks

**Difficulty:** Easy
**Category:** Data Structures

## Goal

Return a new dictionary containing only the tracks whose `age` is less than or equal to `max_age`.

## Function signature

```python
def delete_stale_tracks(tracks: dict[int, dict], max_age: int) -> dict[int, dict]:
```

## Input format

- `tracks`: dict mapping track ID (`int`) to a track dict containing:
  - `bbox`: `[x1, y1, x2, y2]` — bounding box coordinates
  - `age`: `int` — number of frames since the track was last matched
  - `hits`: `int` — total number of times the track has been matched
- `max_age`: `int` — maximum allowed age; tracks with `age > max_age` are removed

## Output format

- `dict[int, dict]` — new dict with the same structure, containing only tracks where `age <= max_age`

## Examples

```python
tracks = {
    1: {"bbox": [0, 0, 10, 10], "age": 1, "hits": 5},
    2: {"bbox": [5, 5, 15, 15], "age": 4, "hits": 2},
    3: {"bbox": [2, 2,  8,  8], "age": 2, "hits": 8},
}
delete_stale_tracks(tracks, max_age=2)
# {
#   1: {"bbox": [0, 0, 10, 10], "age": 1, "hits": 5},
#   3: {"bbox": [2, 2,  8,  8], "age": 2, "hits": 8},
# }
```

**All removed:**
```python
delete_stale_tracks(tracks, max_age=0)
# {}
```

## Constraints

- `0 <= len(tracks) <= 10_000`
- `age` and `max_age` are non-negative integers
- A track with `age == max_age` is kept
- Do not mutate the input dictionary or any nested track dictionary
- Return a plain `dict`, not a copy of the input
- Track IDs are not guaranteed to be contiguous

## Edge cases

- Empty input → return `{}`
- All tracks within age limit → return all tracks
- All tracks exceed age limit → return `{}`
- `age == max_age` → track is kept
- `max_age = 0` → only tracks with `age == 0` are kept
- Non-contiguous track IDs → IDs are preserved as-is

## Skills practiced

- Filtering a dictionary with a comprehension
- Preserving original keys and values
- Avoiding mutation of nested structures
- Boundary condition handling (`<=` vs `<`)
