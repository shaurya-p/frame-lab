# 009 — Clip Boxes

**Difficulty:** Easy
**Category:** Spatial Geometry

## Goal

Clip bounding box coordinates so they stay within the image boundaries.

## Function signature

```python
def clip_boxes(boxes: np.ndarray, image_shape: tuple[int, int]) -> np.ndarray:
```

## Input format

- `boxes`: `np.ndarray` of shape `(N, 4)` — each row is `[x1, y1, x2, y2]`
- `image_shape`: `(height, width)` — dimensions of the image

## Output format

- `np.ndarray` of shape `(N, 4)` — same format as input, with coordinates clipped to image bounds

## Clipping rules

| Coordinate | Clipped to |
|---|---|
| `x1`, `x2` | `[0, width]` |
| `y1`, `y2` | `[0, height]` |

## Examples

```python
boxes = np.array([
    [ -5,  10,  30,  50],   # x1 negative — needs clipping
    [ 10,  20, 200, 300],   # x2, y2 exceed image — needs clipping
    [ 10,  10,  50,  50],   # fully inside — no change
], dtype=float)

clip_boxes(boxes, image_shape=(100, 150))
# np.array([
#   [  0,  10,  30,  50],
#   [ 10,  20, 150, 100],
#   [ 10,  10,  50,  50],
# ])
```

**Empty input:**
```python
clip_boxes(np.empty((0, 4)), image_shape=(100, 200))
# np.empty((0, 4))  — shape (0, 4)
```

## Constraints

- `N >= 0`; empty input of shape `(0, 4)` must return shape `(0, 4)`
- Input may be integer or float dtype
- Do not mutate the input array
- Use only NumPy

## Edge cases

- Coordinates already within bounds → no change
- Negative coordinates → clipped to `0`
- Coordinates exceeding image size → clipped to `width` or `height`
- Boxes completely outside image → all coordinates clipped to boundary
- Float coordinates are clipped using the same rules

## Skills practiced

- Applying `np.clip` to specific array columns
- Working with `(height, width)` image shape convention
- Avoiding in-place mutation
- Handling empty array shapes
