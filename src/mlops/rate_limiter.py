from __future__ import annotations

import time
from collections import defaultdict


class RateLimiter:
    """Fixed-window rate limiter, keyed by client identifier (e.g. IP address).

    A minimal dependency-free stand-in for `slowapi`/Redis-backed limiting —
    fine for a single-process portfolio demo; a real multi-instance deployment
    would need a shared backing store instead of the in-memory dict here.
    See TODO.md section 20.
    """

    def __init__(self, max_requests: int, window_seconds: float, clock=time.monotonic) -> None:
        self.max_requests = max_requests
        self.window_seconds = window_seconds
        self._clock = clock
        self._hits: dict[str, list[float]] = defaultdict(list)

    def allow(self, key: str) -> bool:
        now = self._clock()
        window_start = now - self.window_seconds
        recent = [t for t in self._hits[key] if t > window_start]

        if len(recent) >= self.max_requests:
            self._hits[key] = recent
            return False

        recent.append(now)
        self._hits[key] = recent
        return True
