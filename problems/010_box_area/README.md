# 010 — Box Area

**Difficulty:** Easy
**Category:** Spatial Geometry

## Goal

Compute the area of each bounding box in an array of `[x1, y1, x2, y2]` boxes.

## Function signature

```python
def box_area(boxes: np.ndarray) -> np.ndarray:
```

## Input format

- `boxes`: `np.ndarray` of shape `(N, 4)` — each row is `[x1, y1, x2, y2]`

## Output format

- `np.ndarray` of shape `(N,)` — one area value per box

## Formula

```
width  = x2 - x1
height = y2 - y1
area   = width * height
```

Negative widths or heights are clamped to `0` before multiplying. Inverted or degenerate boxes produce area `0`, not a negative value.

## Examples

```python
boxes = np.array([
    [0,  0, 10, 20],   # area = 10 * 20 = 200
    [5,  5, 15, 10],   # area = 10 * 5  = 50
    [3,  3,  3,  3],   # area = 0 (point)
], dtype=float)

box_area(boxes)
# np.array([200., 50., 0.])
```

**Inverted box:**
```python
box_area(np.array([[10, 10, 5, 5]], dtype=float))
# np.array([0.])  — negative width and height clamped to 0
```

**Empty input:**
```python
box_area(np.empty((0, 4)))
# np.array([])  — shape (0,)
```

## Constraints

- `N >= 0`; empty input of shape `(0, 4)` must return shape `(0,)`
- Negative widths or heights are clamped to `0`
- Input may be integer or float dtype
- Do not mutate the input array
- Use only NumPy

## Edge cases

- Zero-area box (point: `x1==x2` and `y1==y2`) → `0`
- Zero-area box (line: only one dimension is zero) → `0`
- Inverted box (`x2 < x1` or `y2 < y1`) → `0`
- Empty array `(0, 4)` → output shape `(0,)`

## Skills practiced

- NumPy column arithmetic
- Clamping with `np.clip`
- Handling degenerate geometry
- Output shape reduction from `(N, 4)` to `(N,)`
