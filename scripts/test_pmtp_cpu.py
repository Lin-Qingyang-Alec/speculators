"""Run the PMTP unit tests on CPU, including on Windows.

The project imports Unix file locking at package import time. On Windows only,
provide an import shim which raises if a test actually requests file locking.
This runner cannot validate online transport, vLLM, or distributed training.
"""

import os
import sys
from pathlib import Path
from types import ModuleType


def _unsupported_file_lock(*args, **kwargs):  # noqa: ARG001
    raise RuntimeError(
        "Unix file transfer is unavailable in the Windows CPU test runner"
    )


def main() -> int:
    os.environ["CUDA_VISIBLE_DEVICES"] = ""
    root = Path(__file__).resolve().parents[1]
    sys.path.insert(0, str(root))
    if sys.platform == "win32":
        file_lock = ModuleType("fcntl")
        file_lock.__dict__.update(LOCK_EX=2, LOCK_NB=4, flock=_unsupported_file_lock)
        sys.modules["fcntl"] = file_lock
        print("Windows CPU tests: Unix transport is disabled and fails if invoked.")
    import pytest  # noqa: PLC0415

    return pytest.main([str(root / "tests/unit/models/test_pmtp.py"), *sys.argv[1:]])


if __name__ == "__main__":
    raise SystemExit(main())
