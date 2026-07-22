"""Path-resolution helpers shared across project and subproject scripts.

Project structure::

    melanoma-assignment-3/              <- PROJECT_ROOT
    ├── data/                           <- Shared project-level data
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
code, configuration, logs, models, plots, and reports specific to the current
assignment question.

This module provides both:

1. ``ProjectPaths`` and ``get_project_paths()`` for structured path access.
2. Module-level path constants for convenient use by project scripts.
"""

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable


# ============================================================================
# Path dataclass
# ============================================================================


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


# ============================================================================
# Root discovery
# ============================================================================


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
    directory alongside the current subproject.

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


# ============================================================================
# Structured path resolution
# ============================================================================


def get_project_paths(start: Path) -> ProjectPaths:
    """Resolve all important project and subproject paths.

    Args:
        start:
            Any file or directory within the current subproject.

    Returns:
        A ``ProjectPaths`` instance containing resolved paths for:

        - the overall project root;
        - the current subproject root;
        - shared data, raw, and processed data;
        - subproject configuration;
        - subproject logs;
        - subproject reports;
        - subproject models;
        - subproject plots.
    """
    subproject_root = find_subproject_root(start)
    project_root = find_project_root(subproject_root)

    data_dir = project_root / "data"

    return ProjectPaths(
        project_root=project_root,
        subproject_root=subproject_root,
        data=data_dir,
        raw=data_dir / "raw",
        processed=data_dir / "processed",
        logs=subproject_root / "logs",
        config=subproject_root / "config",
        reports=subproject_root / "reports",
        models=subproject_root / "models",
        plots=subproject_root / "plots",
    )


# ============================================================================
# Module-level convenience paths
# ============================================================================

# Resolve paths relative to this file. This allows any module in the
# subproject to import these constants without manually reconstructing paths.
PATHS = get_project_paths(Path(__file__).resolve())

PROJECT_ROOT = PATHS.project_root
SUBPROJECT_ROOT = PATHS.subproject_root

DATA_DIR = PATHS.data
RAW_DIR = PATHS.raw
PROCESSED_DIR = PATHS.processed

CONFIG_DIR = PATHS.config
LOG_DIR = PATHS.logs
REPORTS_DIR = PATHS.reports
MODELS_DIR = PATHS.models
PLOTS_DIR = PATHS.plots