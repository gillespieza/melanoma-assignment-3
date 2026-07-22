"""Logging and stream utilities for the project."""

from __future__ import annotations

from typing import IO, Any


class TeeStream:
    """Writes output to multiple streams simultaneously.

    This is primarily used to duplicate console output into a log file while
    preserving normal terminal output.

    Example:
        with open(log_path, "w", encoding="utf-8") as log_file:
            stdout_tee = TeeStream(sys.stdout, log_file)
            stderr_tee = TeeStream(sys.stderr, log_file)

            with contextlib.redirect_stdout(stdout_tee):
                print("This appears in both the console and the log file.")
    """

    def __init__(
        self,
        *streams: IO[str],
    ) -> None:
        """Initialises the tee stream.

        Args:
            *streams: Text streams that should receive all written output.

        Raises:
            ValueError: If no streams are provided.
        """
        if not streams:
            raise ValueError(
                "TeeStream requires at least one output stream."
            )

        self.streams = streams

    def write(
        self,
        data: str,
    ) -> int:
        """Writes data to every wrapped stream.

        Args:
            data: Text to write.

        Returns:
            Number of characters written.
        """
        for stream in self.streams:
            stream.write(data)
            stream.flush()

        return len(data)

    def flush(self) -> None:
        """Flushes every wrapped stream."""
        for stream in self.streams:
            stream.flush()

    def isatty(self) -> bool:
        """Returns whether any wrapped stream is attached to a terminal."""
        return any(
            getattr(
                stream,
                "isatty",
                lambda: False,
            )()
            for stream in self.streams
        )

    def fileno(self) -> int:
        """Returns the file descriptor of the first suitable stream.

        This supports compatibility with code that expects a file-like object
        to expose ``fileno()``.

        Raises:
            OSError: If none of the wrapped streams exposes a valid file
                descriptor.
        """
        for stream in self.streams:
            try:
                return stream.fileno()
            except (AttributeError, OSError):
                continue

        raise OSError(
            "No wrapped stream exposes a valid file descriptor."
        )

    def __enter__(self) -> TeeStream:
        """Returns the tee stream as a context manager."""
        return self

    def __exit__(
        self,
        exc_type: Any,
        exc_value: Any,
        traceback: Any,
    ) -> None:
        """Flushes all wrapped streams on context-manager exit."""
        self.flush()