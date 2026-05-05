# frame-lab

A hands-on coding lab for practicing robotic perception algorithms through small, testable Python exercises.

---

## What is frame-lab?

frame-lab is a LeetCode-style local practice repo for robotic perception implementation fluency. Each problem is a self-contained folder with a README, a starter file, a reference solution, and tests.

## Who is this for?

Engineers and students who want to build fluency implementing perception algorithms from scratch — bounding box math, IoU, NMS, tracking utilities, and more.

## Why does it exist?

Most perception knowledge lives in large libraries. frame-lab isolates the core algorithms as small, testable exercises so you can implement and verify them directly.

---

## Setup

Requires Python >=3.10 and [uv](https://docs.astral.sh/uv/).

```bash
git clone https://github.com/your-username/frame-lab
cd frame-lab
uv sync
```

---

## How to run tests

Run all tests:
```bash
uv run pytest -q
```

Run one problem:
```bash
uv run pytest problems/011_iou_two_boxes/test_starter.py -q
```

Run with full output:
```bash
uv run pytest problems/011_iou_two_boxes/test_starter.py -vv
```

---

## How to solve a problem

1. Open `problems/NNN_problem_name/README.md` and read the problem statement.
2. Implement your solution in `starter.py`.
3. Run the tests.
4. If stuck, consult `solution.py` as a reference.

---

## v0.1 problems

### Data Structures
| # | Problem | Difficulty |
|---|---|---|
| 001 | filter_detections_by_score | Easy |
| 002 | sort_detections_by_score | Easy |
| 003 | group_detections_by_class | Easy |
| 004 | delete_stale_tracks | Easy |
| 005 | detections_to_arrays | Easy |
| 006 | arrays_to_detections | Easy |

### Spatial Geometry
| # | Problem | Difficulty |
|---|---|---|
| 007 | xyxy_to_xywh | Easy |
| 008 | xywh_to_xyxy | Easy |
| 009 | clip_boxes | Easy |
| 010 | box_area | Easy |

### Similarity & Association
| # | Problem | Difficulty |
|---|---|---|
| 011 | iou_two_boxes | Easy |
| 012 | iou_one_to_many | Medium |
| 013 | iou_matrix | Medium |

### Selection & Suppression
| # | Problem | Difficulty |
|---|---|---|
| 014 | vanilla_nms | Medium |
| 015 | classwise_nms | Medium |

---

## Roadmap

Future categories planned:
- Image processing
- Geometry
- Tracking
- SLAM / SfM fundamentals
- LiDAR and point-cloud algorithms
- Sensor fusion basics
