"""I/O file utilities re-export for Q1 Subproject.

Re-exports atomic CSV file writing and I/O helpers from the root
`src.utils.io` single source of truth via explicit file path loading.
"""

import importlib.util
from pathlib import Path

_ROOT_IO_PATH = Path(__file__).resolve().parent.parent.parent.parent / "src" / "utils" / "io.py"
_spec = importlib.util.spec_from_file_location("_root_io", _ROOT_IO_PATH)
_root_io = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_root_io)

for _attr in dir(_root_io):
    if not _attr.startswith("_"):
        globals()[_attr] = getattr(_root_io, _attr)
