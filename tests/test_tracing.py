"""Tests for OpenTelemetry tracing utilities."""

from __future__ import annotations

import pytest

from landmarkdiff.tracing import (
    PIPELINE_SPAN_NAMES,
    _NoOpSpan,
    _NoOpTracer,
    get_tracer,
    trace_stage,
)


class TestNoOpSpan:
    def test_set_attribute_noop(self) -> None:
        span = _NoOpSpan()
        span.set_attribute("key", "val")  # should not raise

    def test_set_status_noop(self) -> None:
        span = _NoOpSpan()
        span.set_status("OK")

    def test_record_exception_noop(self) -> None:
        span = _NoOpSpan()
        span.record_exception(ValueError("test"))

    def test_end_noop(self) -> None:
        span = _NoOpSpan()
        span.end()

    def test_context_manager(self) -> None:
        with _NoOpSpan() as span:
            span.set_attribute("inside", True)


class TestNoOpTracer:
    def test_start_span_returns_noop(self) -> None:
        tracer = _NoOpTracer()
        span = tracer.start_span("test")
        assert isinstance(span, _NoOpSpan)

    def test_start_as_current_span_returns_noop(self) -> None:
        tracer = _NoOpTracer()
        span = tracer.start_as_current_span("test")
        assert isinstance(span, _NoOpSpan)


class TestGetTracer:
    def test_returns_tracer(self) -> None:
        tracer = get_tracer()
        # Without otel installed, should get NoOpTracer
        assert hasattr(tracer, "start_span")

    def test_custom_name(self) -> None:
        tracer = get_tracer("custom")
        assert hasattr(tracer, "start_span")


class TestTraceStage:
    def test_basic_stage(self) -> None:
        tracer = _NoOpTracer()
        with trace_stage(tracer, "test_stage") as span:
            assert isinstance(span, _NoOpSpan)

    def test_with_procedure(self) -> None:
        tracer = _NoOpTracer()
        with trace_stage(tracer, "extract", procedure="rhinoplasty"):
            pass  # should not raise

    def test_with_all_attributes(self) -> None:
        tracer = _NoOpTracer()
        with trace_stage(
            tracer,
            "inference",
            procedure="blepharoplasty",
            resolution=512,
            intensity=65.0,
            attributes={"batch_size": 4, "seed": 42},
        ):
            pass

    def test_exception_propagated(self) -> None:
        tracer = _NoOpTracer()
        with pytest.raises(ValueError, match="test error"), trace_stage(tracer, "failing_stage"):
            raise ValueError("test error")

    def test_span_receives_attributes(self) -> None:
        """Verify attributes are set on the span."""
        recorded: dict[str, object] = {}

        class _TrackingSpan(_NoOpSpan):
            def set_attribute(self, key: str, value: object) -> None:
                recorded[key] = value

        class _TrackingTracer(_NoOpTracer):
            def start_span(self, name: str, **kwargs: object) -> _TrackingSpan:
                return _TrackingSpan()

        tracer = _TrackingTracer()
        with trace_stage(
            tracer,
            "test",
            procedure="rhinoplasty",
            resolution=256,
            intensity=50.0,
        ):
            pass

        assert recorded["landmarkdiff.procedure"] == "rhinoplasty"
        assert recorded["landmarkdiff.resolution"] == 256
        assert recorded["landmarkdiff.intensity"] == 50.0
        assert "landmarkdiff.duration_ms" in recorded

    def test_exception_recorded_on_span(self) -> None:
        """Verify exceptions are recorded on the span."""
        exceptions: list[BaseException] = []

        class _ExcSpan(_NoOpSpan):
            def record_exception(self, exc: BaseException) -> None:
                exceptions.append(exc)

        class _ExcTracer(_NoOpTracer):
            def start_span(self, name: str, **kwargs: object) -> _ExcSpan:
                return _ExcSpan()

        tracer = _ExcTracer()
        with pytest.raises(RuntimeError), trace_stage(tracer, "fail"):
            raise RuntimeError("boom")

        assert len(exceptions) == 1
        assert str(exceptions[0]) == "boom"

    def test_span_ended_after_success(self) -> None:
        ended = []

        class _EndSpan(_NoOpSpan):
            def end(self) -> None:
                ended.append(True)

        class _EndTracer(_NoOpTracer):
            def start_span(self, name: str, **kwargs: object) -> _EndSpan:
                return _EndSpan()

        tracer = _EndTracer()
        with trace_stage(tracer, "test"):
            pass
        assert len(ended) == 1

    def test_span_ended_after_error(self) -> None:
        ended = []

        class _EndSpan(_NoOpSpan):
            def end(self) -> None:
                ended.append(True)

        class _EndTracer(_NoOpTracer):
            def start_span(self, name: str, **kwargs: object) -> _EndSpan:
                return _EndSpan()

        tracer = _EndTracer()
        with pytest.raises(ValueError), trace_stage(tracer, "test"):
            raise ValueError("err")
        assert len(ended) == 1


class TestPipelineSpanNames:
    def test_contains_key_stages(self) -> None:
        assert "landmark_extraction" in PIPELINE_SPAN_NAMES
        assert "diffusion_inference" in PIPELINE_SPAN_NAMES
        assert "postprocessing" in PIPELINE_SPAN_NAMES

    def test_count(self) -> None:
        assert len(PIPELINE_SPAN_NAMES) == 12

    def test_all_strings(self) -> None:
        for name in PIPELINE_SPAN_NAMES:
            assert isinstance(name, str)
            assert len(name) > 0
