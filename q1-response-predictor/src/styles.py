"""Centralized Visualisation Styles re-export for Q1 Subproject.

Re-exports all style standards, Okabe-Ito palettes, and formatting functions from the
root `src.styles` single source of truth via explicit file path loading.
"""

import importlib.util
from pathlib import Path

_ROOT_STYLES_PATH = Path(__file__).resolve().parent.parent.parent / "src" / "styles.py"
_spec = importlib.util.spec_from_file_location("_root_styles", _ROOT_STYLES_PATH)
_root_styles = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_root_styles)

for _attr in dir(_root_styles):
    if not _attr.startswith("_"):
        globals()[_attr] = getattr(_root_styles, _attr)
