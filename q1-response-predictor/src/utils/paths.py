"""Path-resolution helpers shared across project and subproject scripts.

Project structure:

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

The top-level project root contains shared data and logs. The subproject
contains code, configuration, models, plots, and reports specific to the
current assignment question.
"""

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable


@dataclass(frozen=True)
class ProjectPaths:
    """Resolved filesystem paths for the project and current subproject."""

    # ------------------------------------------------------------------
    # Root directories
    # ------------------------------------------------------------------

    project_root: Path
    subproject_root: Path

    # ------------------------------------------------------------------
    # Shared project-level resources
    # ------------------------------------------------------------------

    data: Path
    raw: Path
    processed: Path
    

    # ------------------------------------------------------------------
    # Subproject-level resources
    # ------------------------------------------------------------------

    logs: Path
    config: Path
    reports: Path
    models: Path
    plots: Path

    # ------------------------------------------------------------------
    # Convenience helpers for structured output directories
    # ------------------------------------------------------------------

    def report_dir(self, *parts: str) -> Path:
        """Return a path within the subproject reports directory."""
        return self.reports.joinpath(*parts)

    def model_dir(self, *parts: str) -> Path:
        """Return a path within the subproject models directory."""
        return self.models.joinpath(*parts)

    def plot_dir(self, *parts: str) -> Path:
        """Return a path within the subproject plots directory."""
        return self.plots.joinpath(*parts)

    def ensure_directories(self) -> None:
        """Create the standard project and subproject directories."""
        directories = (
            self.data,
            self.raw,
            self.processed,
            self.logs,
            self.config,
            self.reports,
            self.models,
            self.plots,
        )

        for directory in directories:
            directory.mkdir(parents=True, exist_ok=True)


def find_directory_with_markers(
    start: Path,
    markers: Iterable[str],
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

    markers = tuple(markers)

    for candidate in (start, *start.parents):
        if all((candidate / marker).exists() for marker in markers):
            return candidate

    raise FileNotFoundError(
        f"Could not find a directory containing all markers "
        f"{markers!r} starting from {start}"
    )


def find_subproject_root(start: Path) -> Path:
    """Find the root directory of the current subproject.

    A subproject is identified by the presence of both ``src`` and
    ``config`` directories.

    Args:
        start:
            Any file or directory within the subproject.

    Returns:
        The resolved subproject root.
    """
    return find_directory_with_markers(
        start,
        markers=("src", "config"),
    )


def find_project_root(subproject_root: Path) -> Path:
    """Find the top-level project root containing a subproject.

    The top-level project root is expected to contain the shared ``data``
    directory alongside the subproject directory.

    Args:
        subproject_root:
            Path to the current subproject root.

    Returns:
        The resolved top-level project root.

    Raises:
        NotADirectoryError:
            If ``subproject_root`` is not a directory.

        FileNotFoundError:
            If the expected shared ``data`` directory is not found.
    """
    subproject_root = subproject_root.resolve()

    if not subproject_root.is_dir():
        raise NotADirectoryError(
            f"Subproject root is not a directory: {subproject_root}"
        )

    project_root = subproject_root.parent
    data_dir = project_root / "data"

    if not data_dir.is_dir():
        raise FileNotFoundError(
            f"Expected shared data directory not found: {data_dir}"
        )

    return project_root


def get_project_paths(start: Path) -> ProjectPaths:
    """Resolve all important project and subproject paths.

    Args:
        start:
            Any file or directory within the current subproject.

    Returns:
        A ``ProjectPaths`` instance containing resolved paths for:

        - the overall project root;
        - the current subproject root;
        - shared raw and processed data;
        - shared logs;
        - subproject configuration;
        - subproject reports;
        - subproject models;
        - subproject plots.
    """
    subproject_root = find_subproject_root(start)
    project_root = find_project_root(subproject_root)

    return ProjectPaths(
        # Roots
        project_root=project_root,
        subproject_root=subproject_root,

        # Shared project-level resources
        data=project_root / "data",
        raw=project_root / "data" / "raw",
        processed=project_root / "data" / "processed",
        
        # Subproject-level resources
        logs=subproject_root / "logs",
        config=subproject_root / "config",
        reports=subproject_root / "reports",
        models=subproject_root / "models",
        plots=subproject_root / "plots",
    )