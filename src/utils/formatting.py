"""Reusable formatting utilities for reports, tables, logs, and plots."""

from __future__ import annotations

from typing import Any, Dict, List, Optional
import numpy as np
import pandas as pd


def format_count_percentage(
    count: int,
    total: int,
    decimals: int = 1,
) -> str:
    """Formats a count with its percentage."""
    if total == 0:
        return "N/A"
    percentage = (count / total) * 100.0
    return f"{count} ({percentage:.{decimals}f}%)"


def format_median(
    value: Optional[float | int],
    decimals: int = 1,
    not_reached: str = "NR",
) -> str:
    """Formats a numerical median value."""
    if value is None or not np.isfinite(value):
        return not_reached
    return f"{value:.{decimals}f}"


def format_value(
    value: Any,
    decimals: int = 1,
    missing: str = "N/A",
) -> str:
    """Formats a general numerical value."""
    if value is None:
        return missing
    if isinstance(value, float) and np.isnan(value):
        return missing
    if isinstance(value, (int, float)):
        return f"{value:.{decimals}f}"
    return str(value)


def format_median_iqr(statistics: Dict[str, float]) -> str:
    """Formats median and IQR as: 58.0 (47.0-70.5)."""
    median = statistics.get("median_age", np.nan)
    q1 = statistics.get("q1_age", np.nan)
    q3 = statistics.get("q3_age", np.nan)
    if pd.isna(median):
        return "N/A"
    return f"{median:.1f} ({q1:.1f}-{q3:.1f})"


def generate_obsidian_frontmatter(
    title: Optional[str] = None,
    aliases: Optional[List[str]] = None,
    tags: Optional[List[str]] = None,
    created: Optional[str] = None,
    updated: Optional[str] = None,
    extra_css_classes: Optional[List[str]] = None,
) -> str:
    """Generates standard Obsidian-compliant YAML frontmatter for reports.

    Args:
        title: Optional title for the report.
        aliases: Optional list of aliases.
        tags: Optional list of tags.
        created: Optional creation timestamp (YYYY-MM-DD HH:MM). Defaults to current time.
        updated: Optional update timestamp (YYYY-MM-DD HH:MM). Defaults to current time.
        extra_css_classes: Additional cssclass entries appended after ``table-small``
            (e.g. ``["table-center", "row-alt"]`` for Q5 reports).

    Returns:
        Formatted YAML frontmatter block enclosed by ``---``.
    """
    now_str = pd.Timestamp.now().strftime("%Y-%m-%d %H:%M")
    created_ts = created or now_str
    updated_ts = updated or now_str

    lines = ["---"]
    if title:
        clean_title = title.strip('"\'')
        lines.append(f'title: "{clean_title}"')
    if aliases:
        lines.append("aliases:")
        for alias in aliases:
            lines.append(f"  - {alias}")
    if tags:
        lines.append("tags:")
        for tag in tags:
            lines.append(f"  - {tag}")

    css_classes = ["table-small"] + (extra_css_classes or [])
    css_lines = ["cssclasses:"] + [f"  - {cls}" for cls in css_classes]
    lines.extend(css_lines)
    lines.extend([
        "obsidianEditingMode: preview",
        "obsidianUIMode: source",
        f"updated: {updated_ts}",
        "---",
    ])
    # Insert created timestamp immediately after the last tag/alias block
    # (before cssclasses) to maintain consistent key ordering.
    created_line = f"created: {created_ts}"
    css_start = next(i for i, ln in enumerate(lines) if ln == "cssclasses:")
    lines.insert(css_start, created_line)

    return "\n".join(lines)
