"""Reusable formatting utilities re-export for Q1 Subproject.

Re-exports numerical, statistical, and Obsidian YAML frontmatter formatting helpers from
the root `src.utils.formatting` single source of truth via explicit file path loading.
"""

import importlib.util
from pathlib import Path

_ROOT_FORMATTING_PATH = Path(__file__).resolve().parent.parent.parent.parent / "src" / "utils" / "formatting.py"
_spec = importlib.util.spec_from_file_location("_root_formatting", _ROOT_FORMATTING_PATH)
_root_formatting = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_root_formatting)

for _attr in dir(_root_formatting):
    if not _attr.startswith("_"):
        globals()[_attr] = getattr(_root_formatting, _attr)
