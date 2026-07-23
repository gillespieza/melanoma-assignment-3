"""Project path resolution utilities.

Project structure::

    melanoma-assignment-3/              <- PROJECT_ROOT
    ├── data/
    │   ├── raw/
    │   └── processed/
    │
    └── q1-response-predictor/          <- SUBPROJECT_ROOT
        ├── config/
        ├── logs/
        ├── models/
        ├── plots/
        ├── reports/
        ├── scripts/
        └── src/

The top-level project root contains shared datasets. The subproject contains
code and outputs specific to the current assignment question.
"""

from pathlib import Path


def find_directory_with_markers(
    start: Path,
    markers: tuple[str, ...],
) -> Path:
    """Find the nearest ancestor containing all specified markers.

    Args:
        start:
            File or directory from which to begin searching.

        markers:
            Files or directories that must exist in the candidate directory.

    Returns:
        The nearest matching ancestor directory.

    Raises:
        FileNotFoundError:
            If no matching directory is found.
    """
    start = start.resolve()

    if start.is_file():
        start = start.parent

    for candidate in (start, *start.parents):
        if all((candidate / marker).exists() for marker in markers):
            return candidate

    raise FileNotFoundError(
        f"Could not find a directory containing all markers "
        f"{markers!r} starting from {start}"
    )


def find_subproject_root(start: Path) -> Path:
    """Find the root directory of the current subproject."""
    return find_directory_with_markers(
        start,
        markers=("src", "config"),
    )


def find_project_root(start: Path) -> Path:
    """Find the top-level project root.

    The project root is the parent directory of the subproject root and must
    contain the shared ``data`` directory.
    """
    subproject_root = find_subproject_root(start)
    project_root = subproject_root.parent

    data_dir = project_root / "data"

    if not data_dir.is_dir():
        raise FileNotFoundError(
            f"Expected shared data directory not found: {data_dir}"
        )

    return project_root


# ============================================================================
# Project paths
# ============================================================================

SUBPROJECT_ROOT = find_subproject_root(Path(__file__).resolve())
PROJECT_ROOT = find_project_root(SUBPROJECT_ROOT)

DATA_DIR = PROJECT_ROOT / "data"
RAW_DIR = DATA_DIR / "raw"
PROCESSED_DIR = DATA_DIR / "processed"

CONFIG_DIR = SUBPROJECT_ROOT / "config"
LOG_DIR = SUBPROJECT_ROOT / "logs"
REPORTS_DIR = SUBPROJECT_ROOT / "reports"
MODELS_DIR = SUBPROJECT_ROOT / "models"
PLOTS_DIR = SUBPROJECT_ROOT / "plots"


def format_relative_path(path: Path) -> str:
    """Formats a path relative to PROJECT_ROOT or SUBPROJECT_ROOT for logging.

    Args:
        path: Absolute or relative Path object.

    Returns:
        Posix string representation relative to project root.
    """
    path = Path(path).resolve()
    try:
        return path.relative_to(PROJECT_ROOT).as_posix()
    except ValueError:
        try:
            return path.relative_to(SUBPROJECT_ROOT).as_posix()
        except ValueError:
            return path.as_posix()


# Canonical shorthand alias
rel_path = format_relative_path