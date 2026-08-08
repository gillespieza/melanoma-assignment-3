"""Reusable formatting utilities for reports, tables, logs, and plots."""

from __future__ import annotations

from typing import Any, Dict, List, Optional
from pathlib import Path
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
        f"updated: {updated_ts}",
        "---",
    ])
    # Insert created timestamp immediately after the last tag/alias block
    # (before cssclasses) to maintain consistent key ordering.
    created_line = f"created: {created_ts}"
    css_start = next(i for i, ln in enumerate(lines) if ln == "cssclasses:")
    lines.insert(css_start, created_line)

    return "\n".join(lines)


def generate_script_reference_callout(
    script_entries: List[Dict[str, Any]] | List[Tuple[str, Any, str]],
    base_dir: Optional[Any] = None,
    callout_type: str = "[!formula]+",
    title: str = "Script Reference",
) -> str:
    """Generates a standard Obsidian Script Reference callout box for reports.

    Args:
        script_entries: List of dictionaries (with keys ``name``, ``path``, ``description``)
            or tuples of ``(name, path, description)`` describing each script.
        base_dir: Optional base directory to resolve relative paths against.
            If omitted, attempts to find the top-level project root.
        callout_type: Obsidian callout type string, e.g. ``"[!NOTE]"``,
            ``"![formula]+"`` or ``"[!formula]"``. Defaults to ``"[!NOTE]"``.
        title: Title text appended after the callout type. Defaults to
            ``"Script Reference"``.

    Returns:
        Formatted Markdown callout block string starting with a horizontal rule ``---``
        followed by the specified callout header line.
    """
    from src.utils.paths import find_project_root, format_relative_path

    if base_dir is not None:
        root_path = Path(base_dir).resolve()
    else:
        root_path = find_project_root(Path(__file__).resolve()).resolve()

    root_str = str(root_path).replace("\\", "/").rstrip("/")

    lines = ["---", "", f"> {callout_type} {title}", ">"]
    for entry in script_entries:
        if isinstance(entry, tuple):
            name, path, desc = entry
        elif isinstance(entry, dict):
            name = entry["name"]
            path = entry["path"]
            desc = entry["description"]
        else:
            continue

        p_str = str(Path(path).resolve()).replace("\\", "/")
        if p_str.lower().startswith(root_str.lower()):
            rel_str = p_str[len(root_str):].lstrip("/")
        else:
            rel_str = format_relative_path(Path(path))

        lines.append(f"> - [`{name}`]({rel_str}): {desc}")

    return "\n".join(lines) + "\n"
