"""Logging utilities re-export for Q1 Subproject.

Re-exports `TeeStream` and logging helpers from the root
`src.utils.logging` single source of truth via explicit file path loading.
"""

import importlib.util
from pathlib import Path

_ROOT_LOGGING_PATH = Path(__file__).resolve().parent.parent.parent.parent / "src" / "utils" / "logging.py"
_spec = importlib.util.spec_from_file_location("_root_logging", _ROOT_LOGGING_PATH)
_root_logging = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_root_logging)

for _attr in dir(_root_logging):
    if not _attr.startswith("_"):
        globals()[_attr] = getattr(_root_logging, _attr)
