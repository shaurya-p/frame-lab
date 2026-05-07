# 008 — xywh to xyxy

**Difficulty:** Easy
**Category:** Spatial Geometry

## Goal

Convert an array of bounding boxes from `[x, y, w, h]` format to `[x1, y1, x2, y2]` format.

## Function signature

```python
def xywh_to_xyxy(boxes: np.ndarray) -> np.ndarray:
```

## Input format

- `boxes`: `np.ndarray` of shape `(N, 4)` — each row is `[x, y, w, h]`

## Output format

- `np.ndarray` of shape `(N, 4)` — each row is `[x1, y1, x2, y2]`

## Conversion formula

| Output | Formula |
|---|---|
| `x1` | `x` |
| `y1` | `y` |
| `x2` | `x + w` |
| `y2` | `y + h` |

## Examples

```python
boxes = np.array([
    [10, 20, 40, 60],
    [ 0,  0, 30, 40],
], dtype=float)

xywh_to_xyxy(boxes)
# np.array([
#   [10, 20, 50, 80],
#   [ 0,  0, 30, 40],
# ])
```

**Empty input:**
```python
xywh_to_xyxy(np.empty((0, 4)))
# np.empty((0, 4))  — shape (0, 4)
```

## Constraints

- `N >= 0`; empty input of shape `(0, 4)` must return shape `(0, 4)`
- Input may be integer or float dtype
- Do not mutate the input array
- Use only NumPy

## Edge cases

- Empty array `(0, 4)` → output shape `(0, 4)`
- Single box → output shape `(1, 4)`
- Integer input → conversion formula still applies
- Zero width or height → `x2 == x1` or `y2 == y1`

## Skills practiced

- NumPy column slicing
- Coordinate system conversion
- Inverse of `xyxy_to_xywh` (Problem 007)
- Avoiding in-place mutation with array operations
