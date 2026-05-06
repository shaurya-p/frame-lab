# 003 — Group Detections by Class

**Difficulty:** Easy
**Category:** Data Structures

## Goal

Return a dictionary mapping each `class_id` to the list of detections belonging to that class.

## Function signature

```python
def group_detections_by_class(detections: list[dict]) -> dict[int, list[dict]]:
```

## Input format

- `detections`: list of dicts, each containing:
  - `bbox`: `[x1, y1, x2, y2]` — bounding box coordinates
  - `score`: `float` — confidence score
  - `class_id`: `int` — predicted class

## Output format

- `dict[int, list[dict]]` — keys are `class_id` values; values are lists of detection dicts in their original relative order

## Examples

```python
detections = [
    {"bbox": [0, 0, 10, 10], "score": 0.9, "class_id": 0},
    {"bbox": [5, 5, 15, 15], "score": 0.8, "class_id": 1},
    {"bbox": [1, 1, 11, 11], "score": 0.7, "class_id": 0},
]
group_detections_by_class(detections)
# {
#   0: [{"bbox": [0,0,10,10], "score": 0.9, "class_id": 0},
#       {"bbox": [1,1,11,11], "score": 0.7, "class_id": 0}],
#   1: [{"bbox": [5,5,15,15], "score": 0.8, "class_id": 1}],
# }
```

**Empty input:**
```python
group_detections_by_class([])
# {}
```

## Constraints

- `0 <= len(detections) <= 10_000`
- `class_id` is an integer; negative values and zero are valid
- Do not mutate the input list or any detection dictionary
- Return a plain `dict`, not a `defaultdict`
- Preserve the original relative order of detections within each class group
- NumPy is not required

## Edge cases

- Empty list → return `{}`
- Single detection → one key, one-element list
- All detections share the same `class_id` → one key, all detections in original order
- Negative or zero `class_id` → treated as any other integer key
- Detections already grouped by class in input → order still preserved

## Skills practiced

- Grouping with a dict
- Preserving insertion order
- Avoiding mutation of inputs
- Returning plain built-in types
