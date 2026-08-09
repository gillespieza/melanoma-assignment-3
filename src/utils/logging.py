"""Logging and stream utilities for the project."""

from __future__ import annotations

import contextlib
import datetime
import sys
import time
from pathlib import Path
from typing import IO, Any, Generator

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


@contextlib.contextmanager
def setup_logging(
    log_path: Path, relative_to: Path | None = None
) -> Generator[TeeStream, None, None]:
    """Context manager to set up stdout/stderr logging with execution timing and timestamps.

    Args:
        log_path: Target path for the output log file.
        relative_to: Optional base directory to format relative log path display.

    Yields:
        TeeStream: The active stdout TeeStream instance.
    """
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")

    log_path.parent.mkdir(parents=True, exist_ok=True)

    if relative_to is not None:
        try:
            path_str = log_path.relative_to(relative_to).as_posix()
        except ValueError:
            path_str = display_path(log_path)
    else:
        path_str = display_path(log_path)

    start_time = datetime.datetime.now()
    start_perf = time.perf_counter()
    start_str = start_time.strftime("%Y-%m-%d %H:%M:%S")

    with log_path.open("w", encoding="utf-8", buffering=1) as log_file:
        stdout_tee = TeeStream(sys.stdout, log_file)
        stderr_tee = TeeStream(sys.stderr, log_file)

        with (
            contextlib.redirect_stdout(stdout_tee),
            contextlib.redirect_stderr(stderr_tee),
        ):
            print(f"[{start_str}] Logging console output to {path_str}")
            try:
                yield stdout_tee
            finally:
                end_time = datetime.datetime.now()
                end_perf = time.perf_counter()
                end_str = end_time.strftime("%Y-%m-%d %H:%M:%S")
                duration_sec = end_perf - start_perf

                if duration_sec < 60:
                    runtime_str = f"{duration_sec:.2f}s"
                elif duration_sec < 3600:
                    mins = int(duration_sec // 60)
                    secs = duration_sec % 60
                    runtime_str = f"{mins}m {secs:.2f}s"
                else:
                    hrs = int(duration_sec // 3600)
                    rem = duration_sec % 3600
                    mins = int(rem // 60)
                    secs = rem % 60
                    runtime_str = f"{hrs}h {mins}m {secs:.2f}s"

                print(f"[{end_str}] Execution completed. Total runtime: {runtime_str}")

