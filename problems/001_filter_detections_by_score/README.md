# 001 — Filter Detections by Score

**Difficulty:** Easy
**Category:** Data Structures

## Goal

Return detections whose score is greater than or equal to `min_score`, preserving original order.

## Function signature

```python
def filter_detections_by_score(
    detections: list[dict[str, Any]],
    min_score: float,
) -> list[dict[str, Any]]:
```

## Input

| Parameter | Type | Description |
|---|---|---|
| `detections` | `list[dict[str, Any]]` | List of detection dicts, each containing at least `{"score": float}` |
| `min_score` | `float` | Minimum score threshold, inclusive. Range: `[0.0, 1.0]` |

Each detection dict has at minimum:
```python
{"score": float, "class_id": int, "box": [x1, y1, x2, y2]}
```

## Output

`list[dict[str, Any]]` — filtered list in original order.

## Examples

**Example 1**
```
Input:
  detections = [
      {"score": 0.9, "class_id": 0, "box": [0, 0, 10, 10]},
      {"score": 0.4, "class_id": 1, "box": [5, 5, 15, 15]},
      {"score": 0.7, "class_id": 0, "box": [2, 2, 8, 8]},
  ]
  min_score = 0.6

Output:
  [
      {"score": 0.9, "class_id": 0, "box": [0, 0, 10, 10]},
      {"score": 0.7, "class_id": 0, "box": [2, 2, 8, 8]},
  ]
```

**Example 2**
```
Input:
  detections = [{"score": 0.5, "class_id": 0, "box": [0, 0, 1, 1]}]
  min_score = 0.5

Output:
  [{"score": 0.5, "class_id": 0, "box": [0, 0, 1, 1]}]
```

## Constraints

- Each dict contains at least a `"score"` key with a `float` value.
- `min_score` is in `[0.0, 1.0]`.
- Do not mutate the input list.
- Preserve original order.

## Edge cases

- Empty input list → return `[]`
- All scores below threshold → return `[]`
- All scores above threshold → return all detections
- `score == min_score` → include (threshold is inclusive)
- `min_score = 0.0` → all detections pass
- `min_score = 1.0` → only detections with `score == 1.0` pass

## Skills practiced

- List filtering
- Dict field access
- Threshold comparison with inclusive boundary
