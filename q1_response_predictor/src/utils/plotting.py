"""Plotting utilities re-export for Q1 Subproject.

Re-exports `save_fig` and plotting helpers from the root
`src.utils.plotting` single source of truth via explicit file path loading.
"""

import importlib.util
from pathlib import Path

_ROOT_PLOTTING_PATH = Path(__file__).resolve().parent.parent.parent.parent / "src" / "utils" / "plotting.py"
_spec = importlib.util.spec_from_file_location("_root_plotting", _ROOT_PLOTTING_PATH)
_root_plotting = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_root_plotting)

for _attr in dir(_root_plotting):
    if not _attr.startswith("_"):
        globals()[_attr] = getattr(_root_plotting, _attr)
