import sys
import pytest


class _ProblemModule(pytest.Module):
    def collect(self):
        test_dir = str(self.path.parent)
        for key in ("solution", "starter"):
            sys.modules.pop(key, None)
        if test_dir in sys.path:
            sys.path.remove(test_dir)
        sys.path.insert(0, test_dir)
        yield from super().collect()


def pytest_collect_file(parent, file_path):
    if file_path.suffix == ".py" and file_path.name.startswith("test_"):
        return _ProblemModule.from_parent(parent, path=file_path)
