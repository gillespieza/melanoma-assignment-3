"""
Reusable formatting utilities.

These functions provide consistent formatting of numerical
and statistical values for reports, tables, logs, and plots.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd


def format_count_percentage(
    count: int,
    total: int,
    decimals: int = 1,
) -> str:
    """
    Formats a count with its percentage.

    Args:
        count:
            Numerator/count.

        total:
            Denominator/total.

        decimals:
            Number of decimal places for the percentage.

    Returns:
        Formatted string.

    Examples:
        >>> format_count_percentage(54, 104)
        '54 (51.9%)'

        >>> format_count_percentage(0, 0)
        'N/A'
    """

    if total == 0:
        return "N/A"

    percentage = (
        count
        / total
        * 100
    )

    return (
        f"{count} "
        f"({percentage:.{decimals}f}%)"
    )


def format_median(
    value: float | int | None,
    decimals: int = 1,
    not_reached: str = "NR",
) -> str:
    """
    Formats a numerical median value.

    NaN and infinite values are represented using
    the specified not-reached value.

    Args:
        value:
            Numerical value to format.

        decimals:
            Number of decimal places.

        not_reached:
            Value to return for missing or non-finite values.

    Returns:
        Formatted value as a string.

    Examples:
        >>> format_median(22.437)
        '22.4'

        >>> format_median(np.nan)
        'NR'
    """

    if value is None:
        return not_reached

    if not np.isfinite(value):
        return not_reached

    return f"{value:.{decimals}f}"


def format_value(
    value: Any,
    decimals: int = 1,
    missing: str = "N/A",
) -> str:
    """
    Formats a general numerical value.

    Args:
        value:
            Value to format.

        decimals:
            Number of decimal places.

        missing:
            Representation for missing values.

    Returns:
        Formatted value as a string.
    """

    if value is None:
        return missing

    if isinstance(value, float) and np.isnan(value):
        return missing

    if isinstance(value, (int, float)):
        return f"{value:.{decimals}f}"

    return str(value)


def format_median_iqr(
    statistics: dict,
) -> str:
    """
    Formats median and IQR as:

        58.0 (47.0–70.5)

    Returns N/A when no valid data are available.
    """

    median = statistics[
        "median_age"
    ]

    q1 = statistics[
        "q1_age"
    ]

    q3 = statistics[
        "q3_age"
    ]

    if pd.isna(median):

        return "N/A"

    return (
        f"{median:.1f} "
        f"({q1:.1f}–{q3:.1f})"
    )


def generate_obsidian_frontmatter(
    title: str | None = None,
    aliases: list[str] | None = None,
    tags: list[str] | None = None,
    created: str | None = None,
    updated: str | None = None,
) -> str:
    """Generates standard Obsidian-compliant YAML frontmatter for reports.

    Args:
        title: Optional title for the report.
        aliases: Optional list of aliases.
        tags: Optional list of tags.
        created: Optional creation timestamp (YYYY-MM-DD HH:MM). Defaults to now.
        updated: Optional update timestamp (YYYY-MM-DD HH:MM). Defaults to now.

    Returns:
        Formatted YAML frontmatter block starting and ending with '---'.
    """
    now_str = pd.Timestamp.now().strftime("%Y-%m-%d %H:%M")
    created_ts = created or now_str
    updated_ts = updated or now_str

    lines = ["---"]
    if title:
        lines.append(f"title: {title}")
    if aliases:
        lines.append("aliases:")
        for alias in aliases:
            lines.append(f"  - {alias}")
    if tags:
        lines.append("tags:")
        for tag in tags:
            lines.append(f"  - {tag}")

    lines.extend([
        f"created: {created_ts}",
        "cssclasses:",
        "  - table-small",
        "obsidianEditingMode: preview",
        "obsidianUIMode: source",
        f"updated: {updated_ts}",
        "---",
    ])

    return "\n".join(lines)