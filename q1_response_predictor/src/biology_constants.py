"""Centralized Biological Constants re-export for Q1 Subproject.

Re-exports all biological constants, gene panels, and RECIST response mappings from the
root `src.biology_constants` single source of truth via explicit file path loading.
"""

import importlib.util
from pathlib import Path

_ROOT_BIOLOGY_CONSTANTS_PATH = Path(__file__).resolve().parent.parent.parent / "src" / "biology_constants.py"
_spec = importlib.util.spec_from_file_location("_root_biology_constants", _ROOT_BIOLOGY_CONSTANTS_PATH)
_root_bio = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_root_bio)

for _attr in dir(_root_bio):
    if not _attr.startswith("_"):
        globals()[_attr] = getattr(_root_bio, _attr)
