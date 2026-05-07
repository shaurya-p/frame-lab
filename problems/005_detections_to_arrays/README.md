# 005 — Detections to Arrays

**Difficulty:** Easy
**Category:** Data Structures

## Goal

Convert a list of detection dictionaries into three parallel NumPy arrays: bounding boxes, scores, and class IDs.

## Function signature

```python
def detections_to_arrays(
    detections: list[dict],
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
```

## Input format

- `detections`: list of dicts, each containing:
  - `bbox`: sequence of 4 numbers in `[x1, y1, x2, y2]` format
  - `score`: `float` — confidence score
  - `class_id`: `int` — predicted class

## Output format

Returns a 3-tuple `(boxes, scores, class_ids)`:

| Array | Shape | dtype |
|---|---|---|
| `boxes` | `(N, 4)` | `float` |
| `scores` | `(N,)` | `float` |
| `class_ids` | `(N,)` | `int` |

Where `N` is the number of detections.

## Examples

```python
detections = [
    {"bbox": [0, 0, 10, 10], "score": 0.9, "class_id": 0},
    {"bbox": [5, 5, 15, 15], "score": 0.6, "class_id": 2},
]
boxes, scores, class_ids = detections_to_arrays(detections)
# boxes     → np.array([[0, 0, 10, 10], [5, 5, 15, 15]], dtype=float)  shape (2, 4)
# scores    → np.array([0.9, 0.6], dtype=float)                         shape (2,)
# class_ids → np.array([0, 2], dtype=int)                               shape (2,)
```

**Empty input:**
```python
boxes, scores, class_ids = detections_to_arrays([])
# boxes     → shape (0, 4), dtype float
# scores    → shape (0,),   dtype float
# class_ids → shape (0,),   dtype int
```

## Constraints

- `0 <= len(detections) <= 10_000`
- Preserve the original order of detections across all three output arrays
- Do not mutate the input list or any detection dictionary
- Empty input must return arrays with the correct shapes — not flat empty arrays
- Use only NumPy; no other dependencies

## Edge cases

- Empty list → `(0, 4)` boxes, `(0,)` scores, `(0,)` class_ids — shapes must be exact
- Single detection → shapes `(1, 4)`, `(1,)`, `(1,)`
- All detections same class → class_ids array contains repeated values
- `bbox` values may be integers in the input — output dtype must still be `float`

## Skills practiced

- Converting structured Python data to NumPy arrays
- Specifying explicit dtypes
- Handling empty-input shape requirements
- Keeping parallel arrays aligned
