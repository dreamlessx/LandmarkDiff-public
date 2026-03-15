"""Health check and system readiness utilities.

Provides structured health status reporting for production deployments,
including GPU availability, model loading status, and dependency checks.

Usage:
    from landmarkdiff.health import HealthChecker

    checker = HealthChecker()
    checker.add_check("gpu", check_gpu_available)
    checker.add_check("model", check_model_loaded)
    status = checker.run()
    print(status.to_dict())
"""

from __future__ import annotations

import logging
import platform
import time
from collections.abc import Callable
from dataclasses import dataclass, field
from enum import Enum
from typing import Any

logger = logging.getLogger(__name__)

_VERSION = "0.2.0"


class HealthStatus(str, Enum):
    """Overall system health status."""

    HEALTHY = "healthy"
    DEGRADED = "degraded"
    UNHEALTHY = "unhealthy"


@dataclass
class CheckResult:
    """Result of a single health check."""

    name: str
    healthy: bool
    message: str = ""
    duration_ms: float = 0.0
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class SystemHealth:
    """Aggregated health status of the system."""

    status: HealthStatus
    checks: list[CheckResult]
    version: str = _VERSION
    uptime_seconds: float = 0.0
    timestamp: float = field(default_factory=time.time)

    def to_dict(self) -> dict[str, Any]:
        return {
            "status": self.status.value,
            "version": self.version,
            "uptime_seconds": round(self.uptime_seconds, 1),
            "timestamp": self.timestamp,
            "checks": {
                c.name: {
                    "healthy": c.healthy,
                    "message": c.message,
                    "duration_ms": round(c.duration_ms, 2),
                    **({"metadata": c.metadata} if c.metadata else {}),
                }
                for c in self.checks
            },
        }

    @property
    def is_healthy(self) -> bool:
        return self.status == HealthStatus.HEALTHY


class HealthChecker:
    """Configurable health check runner.

    Runs registered health checks and aggregates results into
    an overall system health status.

    Args:
        required_checks: Names of checks that must pass for
            HEALTHY status. Other checks can fail with DEGRADED.
    """

    def __init__(self, required_checks: list[str] | None = None) -> None:
        self._checks: dict[str, Callable[[], CheckResult]] = {}
        self._required = set(required_checks or [])
        self._start_time = time.monotonic()

    def add_check(
        self,
        name: str,
        check_fn: Callable[[], CheckResult],
        required: bool = False,
    ) -> None:
        """Register a health check.

        Args:
            name: Unique name for this check.
            check_fn: Callable that returns a CheckResult.
            required: If True, this check must pass for HEALTHY status.
        """
        self._checks[name] = check_fn
        if required:
            self._required.add(name)

    def run(self) -> SystemHealth:
        """Run all registered health checks.

        Returns:
            SystemHealth with aggregated status and individual check results.
        """
        results: list[CheckResult] = []

        for name, check_fn in self._checks.items():
            start = time.monotonic()
            try:
                result = check_fn()
                result.duration_ms = (time.monotonic() - start) * 1000
            except Exception as e:
                result = CheckResult(
                    name=name,
                    healthy=False,
                    message=f"Check raised exception: {e}",
                    duration_ms=(time.monotonic() - start) * 1000,
                )
            result.name = name
            results.append(result)

        status = self._compute_status(results)
        uptime = time.monotonic() - self._start_time

        return SystemHealth(
            status=status,
            checks=results,
            uptime_seconds=uptime,
        )

    def _compute_status(self, results: list[CheckResult]) -> HealthStatus:
        """Determine overall status from individual check results."""
        if not results:
            return HealthStatus.HEALTHY

        failed = {r.name for r in results if not r.healthy}

        if not failed:
            return HealthStatus.HEALTHY

        # If any required check failed, system is unhealthy
        if failed & self._required:
            return HealthStatus.UNHEALTHY

        return HealthStatus.DEGRADED

    @property
    def check_names(self) -> list[str]:
        """Names of all registered checks."""
        return list(self._checks.keys())


# ------------------------------------------------------------------
# Built-in checks
# ------------------------------------------------------------------


def check_python_version() -> CheckResult:
    """Check that Python version is supported."""
    version = platform.python_version()
    major, minor = int(version.split(".")[0]), int(version.split(".")[1])
    healthy = major == 3 and minor >= 9
    return CheckResult(
        name="python_version",
        healthy=healthy,
        message=f"Python {version}" + ("" if healthy else " (requires >= 3.9)"),
        metadata={"version": version},
    )


def check_numpy() -> CheckResult:
    """Check that numpy is available and functional."""
    try:
        import numpy as np

        _ = np.ones(10)  # verify numpy works
        return CheckResult(
            name="numpy",
            healthy=True,
            message=f"numpy {np.__version__}",
            metadata={"version": np.__version__},
        )
    except Exception as e:
        return CheckResult(
            name="numpy",
            healthy=False,
            message=str(e),
        )


def check_opencv() -> CheckResult:
    """Check that OpenCV is available."""
    try:
        import cv2

        return CheckResult(
            name="opencv",
            healthy=True,
            message=f"cv2 {cv2.__version__}",
            metadata={"version": cv2.__version__},
        )
    except Exception as e:
        return CheckResult(
            name="opencv",
            healthy=False,
            message=str(e),
        )


def check_gpu() -> CheckResult:
    """Check GPU availability via torch."""
    try:
        import torch

        available = torch.cuda.is_available()
        if available:
            name = torch.cuda.get_device_name(0)
            vram_gb = torch.cuda.get_device_properties(0).total_mem / 1e9
            return CheckResult(
                name="gpu",
                healthy=True,
                message=f"{name} ({vram_gb:.1f} GB)",
                metadata={"device": name, "vram_gb": round(vram_gb, 1)},
            )
        return CheckResult(
            name="gpu",
            healthy=False,
            message="No CUDA GPU available",
        )
    except ImportError:
        return CheckResult(
            name="gpu",
            healthy=False,
            message="PyTorch not installed",
        )
    except Exception as e:
        return CheckResult(
            name="gpu",
            healthy=False,
            message=str(e),
        )
