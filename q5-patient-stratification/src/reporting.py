"""Reporting module for Q5 patient stratification.

Provides functions to format Obsidian-compliant YAML frontmatter, compile phase statistics,
format Markdown tables, and write the consolidated Q5 stratification report.
"""

from pathlib import Path
from typing import Dict, List, Optional
import pandas as pd

# ---------------------------------------------------------------------------
# Project Imports
# ---------------------------------------------------------------------------

from src.utils.paths import PROJECT_ROOT, rel_path

# ---------------------------------------------------------------------------
# Module-level Constants & Definitions
# ---------------------------------------------------------------------------


def generate_obsidian_frontmatter(
    title: Optional[str] = None,
    aliases: Optional[List[str]] = None,
    tags: Optional[List[str]] = None,
    created: Optional[str] = None,
    updated: Optional[str] = None,
) -> str:
    """Generates standard Obsidian-compliant YAML frontmatter matching Q1 conventions.

    Args:
        title: Optional title for the report.
        aliases: Optional list of aliases.
        tags: Optional list of tags.
        created: Optional creation timestamp (YYYY-MM-DD HH:MM). Defaults to current time.
        updated: Optional update timestamp (YYYY-MM-DD HH:MM). Defaults to current time.

    Returns:
        Formatted YAML frontmatter block enclosed by '---'.
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

    lines.extend([
        f"created: {created_ts}",
        "cssclasses:",
        "  - table-small",
        "  - table-center",
        "  - row-alt",
        "obsidianEditingMode: preview",
        "obsidianUIMode: source",
        f"updated: {updated_ts}",
        "---",
    ])

    return "\n".join(lines)


def format_markdown_table(df: pd.DataFrame, float_format: str = "%.3f") -> str:
    """Converts a pandas DataFrame into GitHub-flavored Markdown table syntax."""
    return df.to_markdown(index=False, floatfmt=float_format)
