"""Shared project path-resolution utilities.

Provides:
- dynamic project-root discovery
- strongly typed standard project-directory paths

Dataset-specific paths should not be defined here. Those should be derived
from the dataset configuration loaded from config/datasets.yaml.
"""

from dataclasses import dataclass
from pathlib import Path


# ---------------------------------------------------------------------------
# Project structure
# ---------------------------------------------------------------------------

DEFAULT_PROJECT_ROOT_MARKERS = ("src", "data")


@dataclass(frozen=True)
class ProjectPaths:
    """Standard directory paths for the project.

    Attributes:
        root: Project root directory.
        data: Root data directory.
        raw: Raw, unmodified input data directory.
        processed: Cleaned and transformed data directory.
        config: Project configuration directory.
        logs: Pipeline log directory.
        reports: Generated reports directory.
    """

    root: Path
    data: Path
    raw: Path
    processed: Path
    config: Path
    logs: Path
    reports: Path


def find_project_root(
    start: Path,
    markers: tuple[str, ...] = DEFAULT_PROJECT_ROOT_MARKERS,
) -> Path:
    """Finds the project root by searching upwards from a starting path.

    A directory is considered the project root when it contains all
    configured marker directories.

    Args:
        start: Path from which to begin searching. This may be a file or
            directory.
        markers: Directory names expected to exist at the project root.

    Returns:
        Resolved project root path.

    Raises:
        FileNotFoundError: If no matching project root can be found.
    """
    start = start.resolve()

    # If the starting path is a file, search from its parent directory.
    search_start = start.parent if start.is_file() else start

    for candidate in (search_start, *search_start.parents):
        if all((candidate / marker).is_dir() for marker in markers):
            return candidate

    raise FileNotFoundError(
        f"Could not locate project root. "
        f"Expected directories {markers} above: {start}"
    )


def get_project_paths(
    project_root: Path,
) -> ProjectPaths:
    """Builds standard project paths from a project root.

    This function only defines stable project-level directories. Dataset-
    specific directories should be derived from the dataset configuration.

    Args:
        project_root: Root directory of the project.

    Returns:
        A ProjectPaths instance containing standard project directories.
    """
    project_root = project_root.resolve()
    data_dir = project_root / "data"

    return ProjectPaths(
        root=project_root,
        data=data_dir,
        raw=data_dir / "raw",
        processed=data_dir / "processed",
        config=project_root / "config",
        logs=project_root / "logs",
        reports=project_root / "reports",
    )