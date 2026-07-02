"""Cooperative cancellation shared by TUI, loop, providers, and tools."""
from __future__ import annotations

import threading


class CancelledError(Exception):
    """Raised when a running agent turn is cancelled by the caller."""


class CancellationToken:
    """Small thread-safe cancellation token.

    The project intentionally stays synchronous for now, so this is just enough
    to let the TUI, streaming model call, and subprocess tools agree on one
    stop signal without introducing an async framework.
    """

    def __init__(self) -> None:
        self._event = threading.Event()

    def cancel(self) -> None:
        self._event.set()

    def reset(self) -> None:
        self._event.clear()

    @property
    def is_cancelled(self) -> bool:
        return self._event.is_set()

    def throw_if_cancelled(self) -> None:
        if self.is_cancelled:
            raise CancelledError("cancelled")
