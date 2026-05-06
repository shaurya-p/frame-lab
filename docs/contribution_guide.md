# Contribution Guide

## Philosophy

frame-lab problems are small, focused, and independently testable. Each problem exercises one implementation skill. Keep that constraint when adding or modifying problems.

## Adding a problem

1. Pick the next available number and a descriptive name: `NNN_problem_name`.
2. Create the folder under `problems/`.
3. Add the five required files: `README.md`, `starter.py`, `solution.py`, `test_starter.py`, `test_solution.py`.
4. Use `docs/problem_template.md` as the README template.
5. Write `test_solution.py` first and verify it passes against `solution.py`.
6. Write `test_starter.py` (identical logic, different import).
7. Confirm `starter.py` raises `NotImplementedError` and the tests fail on it.

## Problem quality checklist

- [ ] README is a problem statement, not a tutorial.
- [ ] README does not leak the solution algorithm.
- [ ] Starter file has no hints, pseudocode, or partial implementation.
- [ ] Solution file is simple, readable, and handles edge cases.
- [ ] Tests cover correctness, edge cases, mutation, shape, and dtype.
- [ ] Tests pass with `uv run pytest problems/*/test_solution.py -q` against `solution.py`.
- [ ] All files pass `uv run ruff check .`.

## Tooling commands

```bash
# Install dependencies
uv sync

# Validate all reference solutions (CI)
uv run pytest problems/*/test_solution.py -q

# Test one problem's starter (learner practice)
uv run pytest problems/NNN_problem_name/test_starter.py -q

# Lint
uv run ruff check .
```

## Dependencies

Do not add runtime dependencies beyond `numpy` without prior discussion.
Dev dependencies: `pytest`, `pytest-timeout`, `ruff`.

## Style

- Type hints required in all problem files.
- Docstrings: one-line description + `Args:` + `Returns:` only.
- No comments explaining what the code does — use clear names.
