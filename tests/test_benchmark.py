"""Tests for landmarkdiff.benchmark."""

from __future__ import annotations

import json
import math
import time

import pytest

from landmarkdiff.benchmark import BenchmarkResult, InferenceBenchmark, Timer

# ---------------------------------------------------------------------------
# BenchmarkResult
# ---------------------------------------------------------------------------


class TestBenchmarkResult:
    def test_fields(self):
        r = BenchmarkResult(config_name="a6000", latency_ms=100.0)
        assert r.config_name == "a6000"
        assert r.latency_ms == 100.0
        assert r.batch_size == 1
        assert r.resolution == 512

    def test_defaults(self):
        r = BenchmarkResult(config_name="test", latency_ms=50.0)
        assert r.vram_gb == 0.0
        assert r.device == ""
        assert r.metadata == {}


# ---------------------------------------------------------------------------
# InferenceBenchmark
# ---------------------------------------------------------------------------


class TestInferenceBenchmark:
    def test_add_result(self):
        bench = InferenceBenchmark()
        bench.add_result("gpu", latency_ms=100.0)
        assert len(bench.results) == 1

    def test_auto_throughput(self):
        bench = InferenceBenchmark()
        bench.add_result("gpu", latency_ms=100.0, batch_size=2)
        # 1000/100 * 2 = 20 FPS
        assert bench.results[0].throughput_fps == pytest.approx(20.0)

    def test_explicit_throughput(self):
        bench = InferenceBenchmark()
        bench.add_result("gpu", latency_ms=100.0, throughput_fps=15.0)
        assert bench.results[0].throughput_fps == 15.0


class TestStatistics:
    @pytest.fixture
    def bench(self):
        b = InferenceBenchmark()
        b.add_result("a6000", latency_ms=100.0, vram_gb=4.0)
        b.add_result("a6000", latency_ms=120.0, vram_gb=4.2)
        b.add_result("a6000", latency_ms=110.0, vram_gb=4.1)
        b.add_result("h100", latency_ms=50.0, vram_gb=3.0)
        b.add_result("h100", latency_ms=55.0, vram_gb=3.1)
        return b

    def test_mean_latency(self, bench):
        assert bench.mean_latency("a6000") == pytest.approx(110.0)

    def test_mean_latency_all(self, bench):
        assert bench.mean_latency() == pytest.approx(87.0)

    def test_p99_latency(self, bench):
        p99 = bench.p99_latency("a6000")
        assert p99 >= 110.0  # Should be near max

    def test_mean_throughput(self, bench):
        t = bench.mean_throughput("h100")
        assert t > 0

    def test_max_vram(self, bench):
        assert bench.max_vram("a6000") == pytest.approx(4.2)

    def test_empty_config(self, bench):
        assert math.isnan(bench.mean_latency("nonexistent"))
        assert bench.max_vram("nonexistent") == 0.0

    def test_config_names(self, bench):
        assert bench.config_names == ["a6000", "h100"]


class TestSummary:
    def test_summary_text(self):
        bench = InferenceBenchmark(model_name="TestModel")
        bench.add_result("a6000", latency_ms=100.0, vram_gb=4.0)
        text = bench.summary()
        assert "TestModel" in text
        assert "a6000" in text
        assert "100.0" in text

    def test_empty_summary(self):
        bench = InferenceBenchmark()
        assert "No benchmark" in bench.summary()


class TestJson:
    def test_to_json(self):
        bench = InferenceBenchmark()
        bench.add_result("gpu", latency_ms=100.0, vram_gb=4.0)
        j = bench.to_json()
        data = json.loads(j)
        assert data["model_name"] == "LandmarkDiff-ControlNet"
        assert len(data["results"]) == 1
        assert "gpu" in data["summary"]

    def test_to_json_file(self, tmp_path):
        bench = InferenceBenchmark()
        bench.add_result("gpu", latency_ms=100.0)
        bench.to_json(tmp_path / "bench.json")
        assert (tmp_path / "bench.json").exists()


# ---------------------------------------------------------------------------
# Timer
# ---------------------------------------------------------------------------


class TestTimer:
    def test_timer_measures_time(self):
        with Timer() as t:
            time.sleep(0.01)
        assert t.elapsed_ms >= 5  # At least 5ms (accounting for system variance)
        assert t.elapsed_s >= 0.005

    def test_timer_zero_on_init(self):
        t = Timer()
        assert t.start_time == 0.0
        assert t.end_time == 0.0
