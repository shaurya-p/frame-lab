# 016 — Greedy IoU Match

**Difficulty:** Medium
**Category:** Similarity & Association

## Goal

Given a precomputed IoU matrix between tracks and detections, greedily assign the best-IoU track-detection pairs subject to one-to-one constraints.

## Function signature

```python
def greedy_iou_match(
    iou_matrix: np.ndarray,
    iou_threshold: float,
) -> list[tuple[int, int]]:
```

## Input format

- `iou_matrix`: `np.ndarray` of shape `(N, M)`
  - Rows are tracks (indices `0…N-1`)
  - Columns are detections (indices `0…M-1`)
  - `iou_matrix[i, j]` is the IoU between track `i` and detection `j`
- `iou_threshold`: `float` — minimum IoU required to form a match

## Output format

- `list[tuple[int, int]]` — matched pairs `(track_index, detection_index)` in the order they were selected (highest IoU first)

## Algorithm

1. Collect all candidate pairs `(i, j)` where `iou_matrix[i, j] >= iou_threshold`.
2. Sort candidates by **descending IoU**. Break ties by **ascending track index**, then **ascending detection index**.
3. Iterate through sorted candidates:
   - Skip if track `i` is already matched.
   - Skip if detection `j` is already matched.
   - Otherwise record the pair and mark both as matched.
4. Return the list of matched pairs.

## Examples

```python
iou_matrix = np.array([
    [0.9, 0.1],
    [0.2, 0.8],
], dtype=float)

greedy_iou_match(iou_matrix, iou_threshold=0.5)
# [(0, 0), (1, 1)]
```

**Competing detections — one track, two candidates:**
```python
iou_matrix = np.array([[0.9, 0.7]], dtype=float)  # shape (1, 2)
greedy_iou_match(iou_matrix, iou_threshold=0.5)
# [(0, 0)]  — track 0 takes detection 0 (higher IoU); detection 1 is unmatched
```

**Empty input:**
```python
greedy_iou_match(np.empty((0, 3)), iou_threshold=0.5)
# []
```

## Constraints

- Empty `iou_matrix` (any dimension is 0) returns `[]`
- Pairs with `IoU < iou_threshold` are never matched
- Each track and each detection appears in at most one pair
- Output pairs are in selection order (highest IoU first)
- Output is a plain Python `list` of `tuple[int, int]` — not NumPy types
- Do not use `scipy` or Hungarian matching
- Do not mutate the input array
- Use only NumPy

## Edge cases

- Shape `(0, M)` or `(N, 0)` or `(0, 0)` → `[]`
- All pairs below threshold → `[]`
- Two tracks competing for one detection → higher-IoU track wins; tie goes to lower track index
- One track competing for two detections → higher-IoU detection wins; tie goes to lower detection index

## Skills practiced

- Greedy assignment from a score matrix
- Lexicographic sort with `np.lexsort` for stable tie-breaking
- One-to-one constraint tracking with sets
- Mapping flat candidate lists back to (row, col) indices
