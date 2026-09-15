"""Project path resolution utilities re-export for Q1 Subproject.

Re-exports all path resolution constants and helpers from the root
`src.utils.paths` single source of truth via explicit file path loading.
"""

import importlib.util
from pathlib import Path

_ROOT_PATHS_PATH = Path(__file__).resolve().parent.parent.parent.parent / "src" / "utils" / "paths.py"
_spec = importlib.util.spec_from_file_location("_root_paths", _ROOT_PATHS_PATH)
_root_paths = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_root_paths)

for _attr in dir(_root_paths):
    if not _attr.startswith("_"):
        globals()[_attr] = getattr(_root_paths, _attr)
