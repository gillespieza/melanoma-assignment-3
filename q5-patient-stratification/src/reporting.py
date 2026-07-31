"""Reporting module for Q5 patient stratification.

Provides functions to format Markdown tables and write the consolidated
Q5 stratification report.  Obsidian YAML frontmatter is generated via the
shared ``src.utils.formatting.generate_obsidian_frontmatter`` utility;
the local duplicate has been removed (see tech-debt resolution 2026-07-31).
"""

from pathlib import Path  # noqa: F401 — retained for downstream callers

import pandas as pd

# ---------------------------------------------------------------------------
# Project Imports
# ---------------------------------------------------------------------------

from src.utils.formatting import generate_obsidian_frontmatter  # re-exported
from src.utils.paths import PROJECT_ROOT, rel_path  # noqa: F401 — retained for callers

__all__ = ["generate_obsidian_frontmatter", "format_markdown_table"]


def format_markdown_table(df: pd.DataFrame, float_format: str = "%.3f") -> str:
    """Convert a pandas DataFrame into GitHub-flavoured Markdown table syntax.

    Args:
        df: DataFrame to render.
        float_format: printf-style format string for floating-point values.

    Returns:
        Markdown-formatted table string.
    """
    return df.to_markdown(index=False, floatfmt=float_format)
