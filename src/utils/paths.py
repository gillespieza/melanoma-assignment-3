"""Project path resolution utilities.

Project structure::

    melanoma-assignment-3/              <- PROJECT_ROOT
    ├── data/
    │   ├── raw/
    │   └── processed/
    ├── src/
    ├── q1-response-predictor/          <- SUBPROJECT_ROOT
    └── q5-patient-stratification/      <- SUBPROJECT_ROOT
"""

from pathlib import Path


def find_directory_with_markers(
    start: Path,
    markers: tuple[str, ...],
) -> Path:
    """Find the nearest ancestor containing all specified markers.

    Args:
        start: File or directory from which to begin searching.
        markers: Files or directories that must exist in the candidate directory.

    Returns:
        The nearest matching ancestor directory.

    Raises:
        FileNotFoundError: If no matching directory is found.
    """
    start = start.resolve()
    if start.is_file():
        start = start.parent

    for candidate in (start, *start.parents):
        if all((candidate / marker).exists() for marker in markers):
            return candidate

    raise FileNotFoundError(
        f"Could not find a directory containing all markers {markers!r} starting from {start}"
    )


def find_project_root(start: Path) -> Path:
    """Find the top-level project root containing data and src."""
    start = start.resolve()
    if start.is_file():
        start = start.parent

    for candidate in (start, *start.parents):
        if (candidate / "data").is_dir() and (candidate / "src").is_dir():
            return candidate

    raise FileNotFoundError(f"Could not find top-level project root from {start}")


def find_subproject_root(start: Path) -> Path:
    """Find the root directory of the current subproject or fall back to project root."""
    start = start.resolve()
    if start.is_file():
        start = start.parent

    for candidate in (start, *start.parents):
        if (candidate / "scripts").is_dir() or (candidate / "models").is_dir():
            return candidate

    return find_project_root(start)


# Project path definitions
PROJECT_ROOT = find_project_root(Path(__file__).resolve())
SUBPROJECT_ROOT = find_subproject_root(Path(__file__).resolve())

DATA_DIR = PROJECT_ROOT / "data"
RAW_DIR = DATA_DIR / "raw"
PROCESSED_DIR = DATA_DIR / "processed"
LOG_DIR = PROJECT_ROOT / "logs"
PLOTS_DIR = PROJECT_ROOT / "plots"
REPORTS_DIR = PROJECT_ROOT / "reports"


def format_relative_path(path: Path) -> str:
    """Formats a path relative to PROJECT_ROOT for logging."""
    path = Path(path).resolve()
    try:
        return path.relative_to(PROJECT_ROOT).as_posix()
    except ValueError:
        return path.as_posix()


rel_path = format_relative_path
