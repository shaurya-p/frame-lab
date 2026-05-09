# 015 — Classwise NMS

**Difficulty:** Medium
**Category:** Selection & Suppression

## Goal

Apply Non-Maximum Suppression independently for each class and return the original indices of all kept boxes, sorted by descending score globally.

## Function signature

```python
def classwise_nms(
    boxes: np.ndarray,
    scores: np.ndarray,
    class_ids: np.ndarray,
    iou_threshold: float,
) -> list[int]:
```

## Input format

- `boxes`: `np.ndarray` of shape `(N, 4)` — boxes in `[x1, y1, x2, y2]` format
- `scores`: `np.ndarray` of shape `(N,)` — confidence scores
- `class_ids`: `np.ndarray` of shape `(N,)` — integer class labels
- `iou_threshold`: `float` — boxes with IoU **strictly greater than** this value are suppressed

## Output format

- `list[int]` — original input indices of kept boxes, sorted by **descending score globally**. Ties in score preserve original input order.

## Algorithm

1. For each unique `class_id`, run greedy NMS independently:
   - Sort that class's boxes by descending score (stable — preserves input order for ties).
   - Select the highest-scoring remaining box.
   - Suppress same-class boxes with IoU **strictly greater than** `iou_threshold`.
   - Repeat until no boxes remain for that class.
2. Combine kept indices from all classes.
3. Sort the combined list by descending score. Break ties by original input index (lower index first).

**Boxes from different classes never suppress each other.**

## Suppression convention

| Condition | Action |
|---|---|
| `IoU > iou_threshold` | Suppress |
| `IoU == iou_threshold` | Keep |
| `IoU < iou_threshold` | Keep |

## Examples

```python
boxes = np.array([
    [0,  0, 10, 10],   # index 0, class 0, score 0.9 — kept
    [1,  1, 11, 11],   # index 1, class 0, score 0.8 — same class, high IoU → suppressed
    [1,  1, 11, 11],   # index 2, class 1, score 0.7 — different class → kept
], dtype=float)
scores    = np.array([0.9, 0.8, 0.7])
class_ids = np.array([0,   0,   1  ])

classwise_nms(boxes, scores, class_ids, iou_threshold=0.5)
# [0, 2]
```

**Multi-class output ordering:**
```python
# Boxes kept: index 2 (score 0.9), index 0 (score 0.5)
# Sorted by descending score → [2, 0]
```

## Constraints

- Empty input returns `[]`
- Returned indices are original input indices, not per-class positions
- Output is a plain Python `list` of Python `int` values
- Output is sorted by descending score globally; ties preserve original input order
- Boxes from different classes are never compared or suppressed against each other
- For equal scores within a class, lower original index is selected first
- Do not mutate input arrays
- Use continuous box geometry — not `+1` pixel geometry
- Use only NumPy

## Edge cases

- Empty input → `[]`
- Single box → `[0]`
- All boxes different classes with high overlap → all kept
- All boxes same class with high overlap → only highest-scoring kept
- `IoU == iou_threshold` → box is **not** suppressed

## Skills practiced

- Decomposing a problem into per-class subproblems
- Running independent NMS per class
- Merging and re-sorting results by global score
- Stable sort for consistent tie-breaking across classes
