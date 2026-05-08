# 013 — IoU Matrix

**Difficulty:** Medium
**Category:** Similarity & Association

## Goal

Compute the full pairwise IoU matrix between two sets of bounding boxes.

## Function signature

```python
def iou_matrix(boxes_a: np.ndarray, boxes_b: np.ndarray) -> np.ndarray:
```

## Input format

- `boxes_a`: `np.ndarray` of shape `(N, 4)` — boxes in `[x1, y1, x2, y2]` format
- `boxes_b`: `np.ndarray` of shape `(M, 4)` — boxes in `[x1, y1, x2, y2]` format

## Output format

- `np.ndarray` of shape `(N, M)` — `output[i, j]` is the IoU between `boxes_a[i]` and `boxes_b[j]`

## Formula

For each pair `(i, j)`:

```
inter_w = max(0, min(x2_a[i], x2_b[j]) - max(x1_a[i], x1_b[j]))
inter_h = max(0, min(y2_a[i], y2_b[j]) - max(y1_a[i], y1_b[j]))
intersection = inter_w * inter_h

area_a[i] = max(0, x2_a[i] - x1_a[i]) * max(0, y2_a[i] - y1_a[i])
area_b[j] = max(0, x2_b[j] - x1_b[j]) * max(0, y2_b[j] - y1_b[j])
union = area_a[i] + area_b[j] - intersection

output[i, j] = intersection / union  (0.0 where union == 0)
```

## Examples

```python
boxes_a = np.array([
    [0,  0, 10, 10],
    [5,  5, 15, 15],
], dtype=float)
boxes_b = np.array([
    [0,  0, 10, 10],
    [20, 20, 30, 30],
], dtype=float)

iou_matrix(boxes_a, boxes_b)
# np.array([
#   [1.0,  0.0],   # boxes_a[0] vs each of boxes_b
#   [25/175, 0.0], # boxes_a[1] vs each of boxes_b
# ])
```

**Empty input:**
```python
iou_matrix(np.empty((0, 4)), boxes_b)  # shape (0, 2)
iou_matrix(boxes_a, np.empty((0, 4)))  # shape (2, 0)
```

## Constraints

- Output shape is always `(N, M)`, including when `N == 0` or `M == 0`
- Where `union == 0`, return `0.0` for that element
- Inverted or degenerate boxes have area `0` — use `max(0, w) * max(0, h)`, not `abs`
- Use continuous box geometry — not `+1` pixel geometry
- Do not mutate input arrays
- Output dtype must be float
- Use only NumPy

> **Note on test coverage:** For inverted boxes, the standard min/max intersection formula always produces zero intersection. Direct area-clamping behavior is enforced by Problem 010 `box_area`.

## Edge cases

- `boxes_a` empty `(0, 4)` → output shape `(0, M)`
- `boxes_b` empty `(0, 4)` → output shape `(N, 0)`
- Both empty → output shape `(0, 0)`
- Diagonal of `iou_matrix(boxes, boxes)` is all `1.0` for valid boxes
- Zero-area or inverted box → that entire row or column is `0.0`

## Skills practiced

- 2D NumPy broadcasting (`[:, None, :]` vs `[None, :, :]`)
- Vectorized pairwise computation without loops
- Safe division with `np.divide` and a zero-filled output
- Understanding output shape from two input arrays
