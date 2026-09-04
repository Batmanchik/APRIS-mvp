from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone
from threading import Lock
from typing import Any


def _percentile(values: list[float], q: float) -> float:
    if not values:
        return 0.0
    if q <= 0:
        return float(min(values))
    if q >= 1:
        return float(max(values))
    ordered = sorted(values)
    idx = int(round((len(ordered) - 1) * q))
    idx = max(0, min(idx, len(ordered) - 1))
    return float(ordered[idx])


@dataclass
class EndpointStats:
    requests: int = 0
    errors: int = 0
    latencies_ms: list[float] = field(default_factory=list)

    def record(self, latency_ms: float, *, is_error: bool) -> None:
        self.requests += 1
        if is_error:
            self.errors += 1
        self.latencies_ms.append(float(latency_ms))
        # Bound memory while preserving a representative window.
        if len(self.latencies_ms) > 2000:
            self.latencies_ms = self.latencies_ms[-1000:]

    def snapshot(self) -> dict[str, float | int]:
        avg_latency = 0.0
        if self.latencies_ms:
            avg_latency = float(sum(self.latencies_ms) / len(self.latencies_ms))
        error_rate = float(self.errors / self.requests) if self.requests > 0 else 0.0
        return {
            "requests": int(self.requests),
            "errors": int(self.errors),
            "error_rate": error_rate,
            "latency_avg_ms": avg_latency,
            "latency_p50_ms": _percentile(self.latencies_ms, 0.50),
            "latency_p95_ms": _percentile(self.latencies_ms, 0.95),
            "latency_max_ms": _percentile(self.latencies_ms, 1.00),
        }


class RuntimeObservability:
    def __init__(self) -> None:
        self._lock = Lock()
        self._started_at = datetime.now(timezone.utc)
        self._requests_total = 0
        self._errors_total = 0
        self._by_endpoint: dict[str, EndpointStats] = {}

    def reset(self) -> None:
        with self._lock:
            self._started_at = datetime.now(timezone.utc)
            self._requests_total = 0
            self._errors_total = 0
            self._by_endpoint = {}

    def record(self, *, method: str, path: str, status_code: int, latency_ms: float) -> None:
        key = f"{method.upper()} {path}"
        with self._lock:
            self._requests_total += 1
            is_error = status_code >= 400
            if is_error:
                self._errors_total += 1
            if key not in self._by_endpoint:
                self._by_endpoint[key] = EndpointStats()
            self._by_endpoint[key].record(latency_ms, is_error=is_error)

    def snapshot(self) -> dict[str, Any]:
        with self._lock:
            now = datetime.now(timezone.utc)
            uptime_seconds = float((now - self._started_at).total_seconds())
            endpoints = {
                endpoint: stats.snapshot()
                for endpoint, stats in sorted(self._by_endpoint.items(), key=lambda item: item[0])
            }
            return {
                "started_at": self._started_at.isoformat(),
                "uptime_seconds": uptime_seconds,
                "requests_total": int(self._requests_total),
                "errors_total": int(self._errors_total),
                "error_rate_total": (
                    float(self._errors_total / self._requests_total) if self._requests_total > 0 else 0.0
                ),
                "endpoints": endpoints,
            }
