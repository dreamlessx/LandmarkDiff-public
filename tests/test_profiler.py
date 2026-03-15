"""Tests for pipeline profiling utilities."""

from __future__ import annotations

import time

from landmarkdiff.benchmark import PipelineProfiler, StageProfile


class TestStageProfile:
    def test_empty_profile(self):
        sp = StageProfile(name="test")
        assert sp.mean_ms == 0.0
        assert sp.min_ms == 0.0
        assert sp.max_ms == 0.0
        assert sp.count == 0

    def test_single_timing(self):
        sp = StageProfile(name="test", times_ms=[10.0])
        assert sp.mean_ms == 10.0
        assert sp.min_ms == 10.0
        assert sp.max_ms == 10.0
        assert sp.count == 1

    def test_multiple_timings(self):
        sp = StageProfile(name="test", times_ms=[10.0, 20.0, 30.0])
        assert sp.mean_ms == 20.0
        assert sp.min_ms == 10.0
        assert sp.max_ms == 30.0
        assert sp.total_ms == 60.0
        assert sp.count == 3


class TestPipelineProfiler:
    def test_empty_profiler(self):
        p = PipelineProfiler()
        assert p.total_ms == 0.0
        assert p.bottleneck() is None
        assert p.stages == []

    def test_stage_context_manager(self):
        p = PipelineProfiler()
        with p.stage("fast"):
            time.sleep(0.001)
        assert len(p.stages) == 1
        assert p.stages[0].name == "fast"
        assert p.stages[0].count == 1
        assert p.stages[0].mean_ms > 0

    def test_multiple_stages(self):
        p = PipelineProfiler()
        with p.stage("step_a"):
            time.sleep(0.001)
        with p.stage("step_b"):
            time.sleep(0.001)
        assert len(p.stages) == 2
        assert p.stages[0].name == "step_a"
        assert p.stages[1].name == "step_b"

    def test_repeated_stage(self):
        p = PipelineProfiler()
        for _ in range(3):
            with p.stage("repeated"):
                time.sleep(0.001)
        assert len(p.stages) == 1
        assert p.stages[0].count == 3

    def test_manual_record(self):
        p = PipelineProfiler()
        p.record("manual", 5.0)
        p.record("manual", 15.0)
        assert p.stages[0].mean_ms == 10.0
        assert p.stages[0].count == 2

    def test_bottleneck(self):
        p = PipelineProfiler()
        p.record("fast", 1.0)
        p.record("slow", 100.0)
        p.record("medium", 10.0)
        assert p.bottleneck() == "slow"

    def test_summary_format(self):
        p = PipelineProfiler()
        p.record("extraction", 50.0)
        p.record("manipulation", 5.0)
        p.record("masking", 8.0)
        summary = p.summary()
        assert "Pipeline Profile" in summary
        assert "extraction" in summary
        assert "Bottleneck: extraction" in summary

    def test_to_dict(self):
        p = PipelineProfiler()
        p.record("step_a", 10.0)
        p.record("step_b", 20.0)
        d = p.to_dict()
        assert "stages" in d
        assert "step_a" in d["stages"]
        assert "step_b" in d["stages"]
        assert d["total_ms"] == 30.0
        assert d["bottleneck"] == "step_b"
        assert d["stages"]["step_a"]["mean_ms"] == 10.0

    def test_reset(self):
        p = PipelineProfiler()
        p.record("step", 10.0)
        assert len(p.stages) == 1
        p.reset()
        assert len(p.stages) == 0
        assert p.total_ms == 0.0

    def test_stage_order_preserved(self):
        p = PipelineProfiler()
        p.record("c", 1.0)
        p.record("a", 2.0)
        p.record("b", 3.0)
        names = [s.name for s in p.stages]
        assert names == ["c", "a", "b"]

    def test_percentage_in_summary(self):
        p = PipelineProfiler()
        p.record("dominant", 90.0)
        p.record("minor", 10.0)
        summary = p.summary()
        assert "90.0%" in summary
        assert "10.0%" in summary
