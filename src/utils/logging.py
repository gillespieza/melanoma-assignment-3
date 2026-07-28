"""Logging and stream utilities for the project."""

from __future__ import annotations

import sys
from pathlib import Path
from typing import IO, Any

from src.utils.paths import PROJECT_ROOT


def display_path(path: Path) -> str:
    """Return a project-relative path for logging where possible."""
    try:
        return path.relative_to(PROJECT_ROOT).as_posix()
    except ValueError:
        return path.as_posix()


class TeeStream:
    """Writes output to multiple streams simultaneously."""

    def __init__(self, *streams: IO[str]) -> None:
        if not streams:
            raise ValueError("TeeStream requires at least one output stream.")
        self.streams = streams

    def write(self, data: str) -> int:
        for stream in self.streams:
            stream.write(data)
            stream.flush()
        return len(data)

    def flush(self) -> None:
        for stream in self.streams:
            stream.flush()

    def isatty(self) -> bool:
        return any(getattr(stream, "isatty", lambda: False)() for stream in self.streams)

    def fileno(self) -> int:
        for stream in self.streams:
            try:
                return stream.fileno()
            except (AttributeError, OSError):
                continue
        raise OSError("No wrapped stream exposes a valid file descriptor.")

    def __enter__(self) -> TeeStream:
        return self

    def __exit__(self, exc_type: Any, exc_value: Any, traceback: Any) -> None:
        self.flush()
