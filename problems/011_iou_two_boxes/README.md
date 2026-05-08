# 011 — IoU: Two Boxes

**Difficulty:** Easy
**Category:** Similarity & Association

## Goal

Compute the Intersection over Union (IoU) between two bounding boxes.

## Function signature

```python
def iou_two_boxes(box_a: np.ndarray, box_b: np.ndarray) -> float:
```

## Input format

- `box_a`: `np.ndarray` of shape `(4,)` — `[x1, y1, x2, y2]`
- `box_b`: `np.ndarray` of shape `(4,)` — `[x1, y1, x2, y2]`

## Output format

- `float` — IoU value in `[0.0, 1.0]`

## Formula

```
inter_w     = max(0, min(x2_a, x2_b) - max(x1_a, x1_b))
inter_h     = max(0, min(y2_a, y2_b) - max(y1_a, y1_b))
intersection = inter_w * inter_h

area_a = max(0, x2_a - x1_a) * max(0, y2_a - y1_a)
area_b = max(0, x2_b - x1_b) * max(0, y2_b - y1_b)
union  = area_a + area_b - intersection

IoU = intersection / union
```

If `union == 0`, return `0.0`.

## Examples

**Partial overlap:**
```python
box_a = np.array([0, 0, 10, 10], dtype=float)
box_b = np.array([5, 5, 15, 15], dtype=float)
iou_two_boxes(box_a, box_b)
# intersection = 5*5 = 25
# area_a = 100, area_b = 100, union = 175
# IoU ≈ 0.1429
```

**No overlap:**
```python
box_a = np.array([0, 0, 10, 10], dtype=float)
box_b = np.array([20, 20, 30, 30], dtype=float)
iou_two_boxes(box_a, box_b)
# 0.0
```

## Constraints

- If `union == 0`, return `0.0`
- Inverted or degenerate boxes (`x2 <= x1` or `y2 <= y1`) have area `0` — use `max(0, w) * max(0, h)`, not `abs(w) * abs(h)`
- Use continuous box geometry — not `+1` pixel geometry
- Do not mutate input arrays
- Return a Python `float`, not a NumPy scalar
- Use only NumPy

> **Note on test coverage:** For inverted boxes, the standard min/max intersection formula always produces zero intersection, so black-box IoU tests alone cannot detect every incorrect area implementation. Direct area-clamping behavior is verified independently by Problem 010 `box_area`.

## Edge cases

- Identical boxes → `1.0`
- Touching edges (zero-width intersection) → `0.0`
- One box fully inside another → IoU = area_inner / area_outer
- Zero-area box (point or line) → `0.0`
- Inverted box → treated as zero-area → `0.0`
- Both boxes zero-area → `0.0`

## Skills practiced

- Intersection computation with coordinate clamping
- Union formula: `area_a + area_b - intersection`
- Degenerate case handling (`union == 0`)
- Translating a geometric formula directly into code
