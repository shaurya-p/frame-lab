# 012 — IoU: One to Many

**Difficulty:** Easy
**Category:** Similarity & Association

## Goal

Compute the IoU between a single query box and each box in an array of candidates.

## Function signature

```python
def iou_one_to_many(box: np.ndarray, boxes: np.ndarray) -> np.ndarray:
```

## Input format

- `box`: `np.ndarray` of shape `(4,)` — query box in `[x1, y1, x2, y2]` format
- `boxes`: `np.ndarray` of shape `(N, 4)` — candidate boxes in `[x1, y1, x2, y2]` format

## Output format

- `np.ndarray` of shape `(N,)` — IoU of `box` against each row of `boxes`

## Formula

```
inter_w      = max(0, min(x2_box, x2_boxes) - max(x1_box, x1_boxes))
inter_h      = max(0, min(y2_box, y2_boxes) - max(y1_box, y1_boxes))
intersection = inter_w * inter_h

area_box   = max(0, x2_box - x1_box) * max(0, y2_box - y1_box)
area_boxes = max(0, x2_boxes - x1_boxes) * max(0, y2_boxes - y1_boxes)
union      = area_box + area_boxes - intersection

IoU = intersection / union  (0.0 where union == 0)
```

## Examples

```python
box   = np.array([0, 0, 10, 10], dtype=float)
boxes = np.array([
    [ 0,  0, 10, 10],   # identical  → 1.0
    [ 5,  5, 15, 15],   # partial    → 25/175 ≈ 0.143
    [20, 20, 30, 30],   # no overlap → 0.0
], dtype=float)

iou_one_to_many(box, boxes)
# np.array([1.0, 0.1429, 0.0])
```

**Empty candidates:**
```python
iou_one_to_many(box, np.empty((0, 4)))
# np.array([])  — shape (0,)
```

## Constraints

- `N >= 0`; empty `boxes` of shape `(0, 4)` must return shape `(0,)`
- Where `union == 0`, return `0.0` for that element
- Inverted or degenerate boxes have area `0` — use `max(0, w) * max(0, h)`, not `abs`
- Use continuous box geometry — not `+1` pixel geometry
- Do not mutate input arrays
- Output dtype must be float
- Use only NumPy

> **Note on test coverage:** For inverted boxes, the standard min/max intersection formula always produces zero intersection, so these tests verify behavioral contracts. Direct area-clamping behavior is enforced by Problem 010 `box_area`.

## Edge cases

- Empty `boxes` → shape `(0,)`
- Query box is a point or line → all IoUs are `0.0`
- Candidate box is inverted → IoU for that row is `0.0`
- Query box fully inside a candidate → IoU = area_query / area_candidate
- Candidate fully inside query box → IoU = area_candidate / area_query

## Skills practiced

- Vectorized min/max operations with NumPy broadcasting
- Per-element zero-union handling with `np.where`
- Scaling single-box IoU to one-vs-many
