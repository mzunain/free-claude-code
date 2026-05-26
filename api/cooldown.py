"""In-memory provider/model cooldown so the chain skips recently-failed candidates.

When a candidate raises a retryable error during preflight, pre-commit streaming,
or mid-stream forwarding, we mark its ``provider_id/provider_model`` key as
cooled-down for a short window. Subsequent chain resolutions partition the
candidate list into ``[healthy, cooled]`` (preserving the configured order
within each partition) and try healthy candidates first, so the next request
prefers a known-good provider rather than re-hitting the same upstream that
just failed.

This is per-process state only. With the default single-worker uvicorn setup
that the launchd LaunchAgent runs, that is the entire installation. If a future
deployment scales to multiple workers each worker independently learns about a
failing upstream, which is acceptable graceful degradation.
"""

from __future__ import annotations

import time
from collections.abc import Callable, Iterable
from threading import Lock
from typing import TypeVar

DEFAULT_COOLDOWN_SECONDS = 30.0

T = TypeVar("T")


class CooldownStore:
    """Thread-safe ``key -> cooldown_until_monotonic`` store."""

    def __init__(self, default_seconds: float = DEFAULT_COOLDOWN_SECONDS) -> None:
        self._default = max(0.0, float(default_seconds))
        self._lock = Lock()
        self._until: dict[str, float] = {}

    @property
    def default_seconds(self) -> float:
        return self._default

    def mark(
        self,
        key: str,
        *,
        seconds: float | None = None,
        now: float | None = None,
    ) -> float:
        """Record that ``key`` is cooled-down. Returns the absolute monotonic expiry.

        If the key is already cooled-down for a longer period, the existing
        expiry wins (we never shorten an existing cooldown).
        """
        seconds = seconds if seconds is not None and seconds > 0 else self._default
        if seconds <= 0:
            return 0.0
        now_ts = now if now is not None else time.monotonic()
        until = now_ts + seconds
        with self._lock:
            existing = self._until.get(key, 0.0)
            if until > existing:
                self._until[key] = until
                return until
            return existing

    def is_cooled(self, key: str, *, now: float | None = None) -> bool:
        """True if ``key`` is currently in cooldown. Expired entries are evicted."""
        now_ts = now if now is not None else time.monotonic()
        with self._lock:
            until = self._until.get(key)
            if until is None:
                return False
            if now_ts >= until:
                self._until.pop(key, None)
                return False
            return True

    def remaining(self, key: str, *, now: float | None = None) -> float:
        """Seconds remaining on the cooldown for ``key``, or 0.0 if not cooled."""
        now_ts = now if now is not None else time.monotonic()
        with self._lock:
            until = self._until.get(key)
            if until is None or now_ts >= until:
                return 0.0
            return until - now_ts

    def partition(
        self,
        items: Iterable[T],
        key: Callable[[T], str],
        *,
        now: float | None = None,
    ) -> tuple[list[T], list[T]]:
        """Split ``items`` into ``(healthy, cooled)`` while preserving order."""
        now_ts = now if now is not None else time.monotonic()
        healthy: list[T] = []
        cooled: list[T] = []
        for item in items:
            if self.is_cooled(key(item), now=now_ts):
                cooled.append(item)
            else:
                healthy.append(item)
        return healthy, cooled

    def reorder(
        self,
        items: Iterable[T],
        key: Callable[[T], str],
        *,
        now: float | None = None,
    ) -> list[T]:
        """Return ``items`` reordered as healthy-first, cooled-last.

        Both partitions preserve their input order, so the configured chain
        priority is honoured among healthy providers and among cooled providers.
        """
        healthy, cooled = self.partition(items, key, now=now)
        return healthy + cooled

    def reset(self) -> None:
        """Clear all cooldowns. Primarily for tests."""
        with self._lock:
            self._until.clear()


def candidate_key(provider_id: str, provider_model: str) -> str:
    """Canonical key for ``CooldownStore`` lookups."""
    return f"{provider_id}/{provider_model}"
