# 002 — Sort Detections by Score

**Difficulty:** Easy
**Category:** Data Structures

## Goal

Return a new list of detection dictionaries sorted by their `score` field.

## Function signature

```python
def sort_detections_by_score(
    detections: list[dict],
    descending: bool = True,
) -> list[dict]:
```

## Input format

- `detections`: list of dicts, each containing:
  - `bbox`: `[x1, y1, x2, y2]` — bounding box coordinates
  - `score`: `float` — confidence score
  - `class_id`: `int` — predicted class
- `descending`: `bool` — if `True`, highest score first; if `False`, lowest score first (default: `True`)

## Output format

- `list[dict]` — new list of the same detection dicts, sorted by `score`

## Examples

**Descending (default):**
```python
detections = [
    {"bbox": [0, 0, 10, 10], "score": 0.5, "class_id": 0},
    {"bbox": [5, 5, 15, 15], "score": 0.9, "class_id": 1},
    {"bbox": [2, 2,  8,  8], "score": 0.2, "class_id": 0},
]
sort_detections_by_score(detections)
# [{"bbox": [5, 5, 15, 15], "score": 0.9, "class_id": 1},
#  {"bbox": [0, 0, 10, 10], "score": 0.5, "class_id": 0},
#  {"bbox": [2, 2,  8,  8], "score": 0.2, "class_id": 0}]
```

**Ascending:**
```python
sort_detections_by_score(detections, descending=False)
# [{"bbox": [2, 2,  8,  8], "score": 0.2, "class_id": 0},
#  {"bbox": [0, 0, 10, 10], "score": 0.5, "class_id": 0},
#  {"bbox": [5, 5, 15, 15], "score": 0.9, "class_id": 1}]
```

## Constraints

- `0 <= len(detections) <= 10_000`
- `score` is a float; negative scores are valid
- Do not mutate the input list
- Preserve relative order for detections with equal scores (stable sort)
- NumPy is not required

## Edge cases

- Empty list → return `[]`
- Single detection → return a new list containing that one detection
- All scores equal → preserve original relative order
- Already sorted input → return correctly sorted result
- Reverse-sorted input → return correctly sorted result
- Negative scores → treated as any other float

## Skills practiced

- Sorting with a custom key
- Stable sort guarantees in Python
- Returning a new list without mutating the input
- Handling edge cases in list operations
