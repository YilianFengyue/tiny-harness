"""Tiny immutable-ish runtime state store.

This is the Python analogue of the tiny Zustand-like store discussed in CH05:
callers update by returning a new state object; returning the same object means
"no change" and subscribers are not notified.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable, Generic, TypeVar

from .features import feature_snapshot

T = TypeVar("T")
Listener = Callable[[], None]
OnChange = Callable[[T, T], None]


class Store(Generic[T]):
    def __init__(self, initial_state: T, on_change: OnChange[T] | None = None) -> None:
        self._state = initial_state
        self._on_change = on_change
        self._listeners: set[Listener] = set()

    def get_state(self) -> T:
        return self._state

    def set_state(self, updater: Callable[[T], T]) -> None:
        old = self._state
        new = updater(old)
        if new is old:
            return
        self._state = new
        if self._on_change:
            self._on_change(new, old)
        for listener in tuple(self._listeners):
            listener()

    def subscribe(self, listener: Listener) -> Callable[[], None]:
        self._listeners.add(listener)

        def unsubscribe() -> None:
            self._listeners.discard(listener)

        return unsubscribe


def create_store(initial_state: T, on_change: OnChange[T] | None = None) -> Store[T]:
    return Store(initial_state, on_change)


@dataclass(frozen=True)
class AppState:
    model: str
    status: str = "ready"
    settings_sources: tuple[str, ...] = ()
    settings_policy_origin: str | None = None
    features: dict[str, object] = field(default_factory=dict)
    permission_context: object | None = None


def build_app_state(cfg, *, permission_context: object | None = None,
                    status: str = "ready") -> AppState:
    snapshot = getattr(cfg, "settings_snapshot", None)
    sources = tuple(layer.source for layer in snapshot.sources) if snapshot else ()
    policy_origin = snapshot.policy_origin if snapshot else None
    return AppState(
        model=cfg.model,
        status=status,
        settings_sources=sources,
        settings_policy_origin=policy_origin,
        features=feature_snapshot(cfg),
        permission_context=permission_context,
    )
