import sys

import pytest


def pytest_collectstart(collector):
    """Make each problem's `solution`/`starter` importable in isolation.

    Test files across problems share basenames and import sibling modules named
    `solution`/`starter`. Before a test module is imported, point sys.path at its
    own directory and drop any cached sibling modules from a previous problem.
    """
    if isinstance(collector, pytest.Module):
        test_dir = str(collector.path.parent)
        for key in ("solution", "starter"):
            sys.modules.pop(key, None)
        if test_dir in sys.path:
            sys.path.remove(test_dir)
        sys.path.insert(0, test_dir)
