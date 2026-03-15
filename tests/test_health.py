"""Tests for health check utilities."""

from __future__ import annotations

import json

from landmarkdiff.health import (
    CheckResult,
    HealthChecker,
    HealthStatus,
    SystemHealth,
    check_numpy,
    check_opencv,
    check_python_version,
)


class TestHealthStatus:
    def test_values(self):
        assert HealthStatus.HEALTHY.value == "healthy"
        assert HealthStatus.DEGRADED.value == "degraded"
        assert HealthStatus.UNHEALTHY.value == "unhealthy"


class TestCheckResult:
    def test_defaults(self):
        r = CheckResult(name="test", healthy=True)
        assert r.message == ""
        assert r.metadata == {}

    def test_with_metadata(self):
        r = CheckResult(name="gpu", healthy=True, metadata={"device": "A6000"})
        assert r.metadata["device"] == "A6000"


class TestSystemHealth:
    def test_to_dict(self):
        checks = [
            CheckResult(name="c1", healthy=True, message="ok"),
            CheckResult(name="c2", healthy=False, message="fail"),
        ]
        health = SystemHealth(
            status=HealthStatus.DEGRADED,
            checks=checks,
            uptime_seconds=120.5,
        )
        d = health.to_dict()
        assert d["status"] == "degraded"
        assert d["checks"]["c1"]["healthy"] is True
        assert d["checks"]["c2"]["healthy"] is False
        assert d["uptime_seconds"] == 120.5

    def test_to_dict_is_json_serializable(self):
        health = SystemHealth(
            status=HealthStatus.HEALTHY,
            checks=[CheckResult(name="test", healthy=True)],
        )
        serialized = json.dumps(health.to_dict())
        assert isinstance(serialized, str)

    def test_is_healthy_property(self):
        h1 = SystemHealth(status=HealthStatus.HEALTHY, checks=[])
        assert h1.is_healthy is True
        h2 = SystemHealth(status=HealthStatus.DEGRADED, checks=[])
        assert h2.is_healthy is False


class TestHealthChecker:
    def test_no_checks_is_healthy(self):
        checker = HealthChecker()
        result = checker.run()
        assert result.status == HealthStatus.HEALTHY

    def test_all_pass_is_healthy(self):
        checker = HealthChecker()
        checker.add_check("ok1", lambda: CheckResult(name="ok1", healthy=True))
        checker.add_check("ok2", lambda: CheckResult(name="ok2", healthy=True))
        result = checker.run()
        assert result.status == HealthStatus.HEALTHY

    def test_optional_fail_is_degraded(self):
        checker = HealthChecker()
        checker.add_check("ok", lambda: CheckResult(name="ok", healthy=True))
        checker.add_check(
            "fail",
            lambda: CheckResult(name="fail", healthy=False),
        )
        result = checker.run()
        assert result.status == HealthStatus.DEGRADED

    def test_required_fail_is_unhealthy(self):
        checker = HealthChecker()
        checker.add_check(
            "critical",
            lambda: CheckResult(name="critical", healthy=False),
            required=True,
        )
        result = checker.run()
        assert result.status == HealthStatus.UNHEALTHY

    def test_required_pass_optional_fail_is_degraded(self):
        checker = HealthChecker()
        checker.add_check(
            "req",
            lambda: CheckResult(name="req", healthy=True),
            required=True,
        )
        checker.add_check(
            "opt",
            lambda: CheckResult(name="opt", healthy=False),
        )
        result = checker.run()
        assert result.status == HealthStatus.DEGRADED

    def test_exception_in_check_reports_unhealthy(self):
        def bad_check():
            raise RuntimeError("boom")

        checker = HealthChecker()
        checker.add_check("bad", bad_check)
        result = checker.run()
        bad_result = result.checks[0]
        assert bad_result.healthy is False
        assert "exception" in bad_result.message.lower()

    def test_uptime_positive(self):
        checker = HealthChecker()
        result = checker.run()
        assert result.uptime_seconds >= 0

    def test_check_names(self):
        checker = HealthChecker()
        checker.add_check("a", lambda: CheckResult(name="a", healthy=True))
        checker.add_check("b", lambda: CheckResult(name="b", healthy=True))
        assert checker.check_names == ["a", "b"]

    def test_duration_recorded(self):
        def slow_check():
            import time

            time.sleep(0.01)
            return CheckResult(name="slow", healthy=True)

        checker = HealthChecker()
        checker.add_check("slow", slow_check)
        result = checker.run()
        assert result.checks[0].duration_ms >= 5  # at least 5ms

    def test_required_via_constructor(self):
        checker = HealthChecker(required_checks=["critical"])
        checker.add_check(
            "critical",
            lambda: CheckResult(name="critical", healthy=False),
        )
        result = checker.run()
        assert result.status == HealthStatus.UNHEALTHY


class TestBuiltInChecks:
    def test_python_version(self):
        result = check_python_version()
        assert result.healthy is True  # we're running >= 3.9
        assert "Python" in result.message

    def test_numpy_check(self):
        result = check_numpy()
        assert result.healthy is True
        assert "numpy" in result.message

    def test_opencv_check(self):
        result = check_opencv()
        assert result.healthy is True
        assert "cv2" in result.message
