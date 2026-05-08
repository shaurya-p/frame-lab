# 014 — Vanilla NMS

**Difficulty:** Medium
**Category:** Selection & Suppression

## Goal

Apply greedy Non-Maximum Suppression (NMS) to a set of detections and return the original indices of the kept boxes.

## Function signature

```python
def vanilla_nms(
    boxes: np.ndarray,
    scores: np.ndarray,
    iou_threshold: float,
) -> list[int]:
```

## Input format

- `boxes`: `np.ndarray` of shape `(N, 4)` — boxes in `[x1, y1, x2, y2]` format
- `scores`: `np.ndarray` of shape `(N,)` — confidence scores
- `iou_threshold`: `float` — boxes with IoU **strictly greater than** this value are suppressed

## Output format

- `list[int]` — original input indices of kept boxes, in selection order (highest score first)

## Algorithm

1. Sort boxes by score descending. For equal scores, preserve original input order.
2. Select the highest-scoring remaining box. Add its original index to the kept list.
3. Compute IoU between the selected box and all remaining boxes.
4. Remove (suppress) any remaining box whose IoU with the selected box is **strictly greater than** `iou_threshold`.
5. Repeat from step 2 until no boxes remain.

## Suppression convention

| Condition | Action |
|---|---|
| `IoU > iou_threshold` | Suppress |
| `IoU == iou_threshold` | Keep |
| `IoU < iou_threshold` | Keep |

## Examples

```python
boxes = np.array([
    [0,  0, 10, 10],  # index 0, score 0.9
    [1,  1, 11, 11],  # index 1, score 0.8  — high overlap with 0 → suppressed
    [20, 20, 30, 30], # index 2, score 0.7  — no overlap → kept
], dtype=float)
scores = np.array([0.9, 0.8, 0.7])

vanilla_nms(boxes, scores, iou_threshold=0.5)
# [0, 2]
```

**Empty input:**
```python
vanilla_nms(np.empty((0, 4)), np.array([]), iou_threshold=0.5)
# []
```

## Constraints

- Empty input returns `[]`
- Returned indices are original input indices, not positions in sorted order
- Output is a plain Python `list` of Python `int` values
- For equal scores, lower original index is selected first
- Do not mutate input arrays
- Use continuous box geometry — not `+1` pixel geometry
- Use only NumPy

## Edge cases

- Empty input → `[]`
- Single box → `[0]`
- No overlapping boxes → all indices kept, in descending score order
- `IoU == iou_threshold` → box is **not** suppressed
- All boxes identical → only the highest-scoring index is kept
- Equal scores → lower original index selected first

## Skills practiced

- Greedy selection loop
- Vectorized IoU computation inside an iterative algorithm
- Stable sort to handle tie-breaking by input order
- Mapping sorted positions back to original input indices
