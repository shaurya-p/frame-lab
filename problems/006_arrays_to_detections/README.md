# 006 — Arrays to Detections

**Difficulty:** Easy
**Category:** Data Structures

## Goal

Convert three parallel NumPy arrays into a list of detection dictionaries, one per row.

## Function signature

```python
def arrays_to_detections(
    boxes: np.ndarray,
    scores: np.ndarray,
    class_ids: np.ndarray,
) -> list[dict]:
```

## Input format

| Array | Shape | dtype |
|---|---|---|
| `boxes` | `(N, 4)` | float — `[x1, y1, x2, y2]` per row |
| `scores` | `(N,)` | float |
| `class_ids` | `(N,)` | int |

## Output format

- `list[dict]` — one dict per detection, in input order:

```python
{
    "bbox": [x1, y1, x2, y2],  # plain Python list of floats
    "score": float,             # plain Python float
    "class_id": int,            # plain Python int
}
```

## Examples

```python
boxes     = np.array([[0, 0, 10, 10], [5, 5, 15, 15]], dtype=float)
scores    = np.array([0.9, 0.6], dtype=float)
class_ids = np.array([0, 2], dtype=int)

arrays_to_detections(boxes, scores, class_ids)
# [
#   {"bbox": [0.0, 0.0, 10.0, 10.0], "score": 0.9, "class_id": 0},
#   {"bbox": [5.0, 5.0, 15.0, 15.0], "score": 0.6, "class_id": 2},
# ]
```

**Empty input:**
```python
arrays_to_detections(
    np.empty((0, 4), dtype=float),
    np.array([], dtype=float),
    np.array([], dtype=int),
)
# []
```

## Constraints

- `N >= 0`; empty arrays return `[]`
- Preserve the original row order across all three arrays
- Each `bbox` in the output must be a plain Python `list`, not a NumPy array
- Each `score` must be a plain Python `float`, not a NumPy scalar
- Each `class_id` must be a plain Python `int`, not a NumPy scalar
- Do not mutate the input arrays
- Use only NumPy

## Edge cases

- Empty arrays → `[]`
- Single row → one-element list
- NumPy scalar types in output are not acceptable — values must be native Python types

## Skills practiced

- Iterating over parallel arrays
- Converting NumPy scalars and rows to plain Python types
- Reconstructing structured data from array form
- Inverse of `detections_to_arrays` (Problem 005)
