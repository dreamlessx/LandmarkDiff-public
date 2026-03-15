"""Tests for progress reporting utilities."""

from __future__ import annotations

import time

from landmarkdiff.benchmark import PIPELINE_STAGES, ProgressReporter


class TestPipelineStages:
    def test_stages_defined(self):
        assert len(PIPELINE_STAGES) >= 5

    def test_stages_have_three_elements(self):
        for stage in PIPELINE_STAGES:
            assert len(stage) == 3
            name, desc, pct = stage
            assert isinstance(name, str)
            assert isinstance(desc, str)
            assert isinstance(pct, int)

    def test_percentages_monotonic(self):
        pcts = [pct for _, _, pct in PIPELINE_STAGES]
        for i in range(1, len(pcts)):
            assert pcts[i] >= pcts[i - 1]

    def test_starts_above_zero(self):
        assert PIPELINE_STAGES[0][2] > 0

    def test_ends_at_100(self):
        assert PIPELINE_STAGES[-1][2] == 100


class TestProgressReporter:
    def test_no_callback(self):
        reporter = ProgressReporter()
        reporter.update("landmark_extraction")
        reporter.update("complete")
        assert reporter.elapsed_ms > 0

    def test_callback_called(self):
        calls = []
        reporter = ProgressReporter(callback=lambda s, d, p: calls.append((s, d, p)))
        reporter.update("landmark_extraction")
        reporter.update("manipulation")
        assert len(calls) == 2
        assert calls[0][0] == "landmark_extraction"
        assert calls[1][0] == "manipulation"

    def test_percentage_reported(self):
        calls = []
        reporter = ProgressReporter(callback=lambda s, d, p: calls.append(p))
        reporter.update("landmark_extraction")
        reporter.update("inference")
        reporter.update("complete")
        assert calls[0] == 10  # landmark_extraction
        assert calls[1] == 70  # inference
        assert calls[2] == 100  # complete

    def test_custom_description(self):
        calls = []
        reporter = ProgressReporter(callback=lambda s, d, p: calls.append(d))
        reporter.update("inference", description="Running custom model")
        assert calls[0] == "Running custom model"

    def test_unknown_stage_accepted(self):
        calls = []
        reporter = ProgressReporter(callback=lambda s, d, p: calls.append((s, p)))
        reporter.update("custom_stage")
        assert calls[0] == ("custom_stage", 0)

    def test_elapsed_ms(self):
        reporter = ProgressReporter()
        reporter.update("landmark_extraction")
        time.sleep(0.01)
        reporter.update("complete")
        assert reporter.elapsed_ms > 5

    def test_stage_times(self):
        reporter = ProgressReporter()
        reporter.update("landmark_extraction")
        time.sleep(0.01)
        reporter.update("manipulation")
        time.sleep(0.01)
        reporter.update("complete")
        times = reporter.stage_times
        assert "landmark_extraction" in times
        assert "manipulation" in times
        assert times["landmark_extraction"] > 0

    def test_gradio_style_callback(self):
        """Gradio Progress() takes (fraction, desc=...) instead of (name, desc, pct)."""
        calls = []

        def gradio_like(fraction, desc=""):
            calls.append((fraction, desc))

        # Simulate TypeError on first call format, fallback to gradio style
        reporter = ProgressReporter(callback=gradio_like)
        reporter.update("landmark_extraction")
        # Should have called with fraction format
        assert len(calls) == 1

    def test_no_callback_does_not_error(self):
        reporter = ProgressReporter(callback=None)
        reporter.update("landmark_extraction")
        reporter.update("complete")

    def test_initial_elapsed_is_zero(self):
        reporter = ProgressReporter()
        assert reporter.elapsed_ms == 0.0
