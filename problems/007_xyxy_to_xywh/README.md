# 007 — xyxy to xywh

**Difficulty:** Easy
**Category:** Spatial Geometry

## Goal

Convert an array of bounding boxes from `[x1, y1, x2, y2]` format to `[x, y, w, h]` format.

## Function signature

```python
def xyxy_to_xywh(boxes: np.ndarray) -> np.ndarray:
```

## Input format

- `boxes`: `np.ndarray` of shape `(N, 4)` — each row is `[x1, y1, x2, y2]`

## Output format

- `np.ndarray` of shape `(N, 4)` — each row is `[x, y, w, h]`

## Conversion formula

| Output | Formula |
|---|---|
| `x` | `x1` |
| `y` | `y1` |
| `w` | `x2 - x1` |
| `h` | `y2 - y1` |

## Examples

```python
boxes = np.array([
    [10, 20, 50, 80],
    [ 0,  0, 30, 40],
], dtype=float)

xyxy_to_xywh(boxes)
# np.array([
#   [10, 20, 40, 60],
#   [ 0,  0, 30, 40],
# ])
```

**Empty input:**
```python
xyxy_to_xywh(np.empty((0, 4)))
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
- Square box (`x2 - x1 == y2 - y1`) → `w == h`

## Skills practiced

- NumPy column slicing
- Coordinate system conversion
- Avoiding in-place mutation with array operations
- Handling empty array shapes
