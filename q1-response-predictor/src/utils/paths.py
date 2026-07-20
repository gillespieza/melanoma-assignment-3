"""Path-resolution helpers shared across pipeline scripts.

Note: top-level scripts that need to add the project root to sys.path
before they can `import src.*` (which is required to reach this module)
still need a small inline copy of this search loop for that first step.
Once BASE_DIR is on sys.path, everything else in the project (including
those same scripts, for any later path lookups) can import and reuse
find_project_root() directly rather than re-deriving paths ad hoc.
"""

from pathlib import Path


def find_project_root(start: Path, markers=("src", "data")) -> Path:
    """
    Walk upward from `start` until a directory containing all `markers`
    is found. Used instead of a fixed parent.parent... depth so scripts
    keep working if they're moved to a different folder depth.
    """
    for candidate in [start] + list(start.parents):
        if all((candidate / m).exists() for m in markers):
            return candidate
    raise FileNotFoundError(f"Could not locate project root (looked for {markers}) above {start}")