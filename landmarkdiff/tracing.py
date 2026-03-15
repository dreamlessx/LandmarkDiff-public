"""OpenTelemetry tracing utilities for the inference pipeline.

Provides span creation and attribute recording for each pipeline stage,
enabling distributed tracing in production deployments. Falls back to
no-op when OpenTelemetry is not installed.

Usage:
    from landmarkdiff.tracing import get_tracer, trace_stage

    tracer = get_tracer()

    with trace_stage(tracer, "landmark_extraction", procedure="rhinoplasty"):
        landmarks = extract_landmarks(image)
"""

from __future__ import annotations

import contextlib
import logging
import time
from collections.abc import Generator
from contextlib import contextmanager
from typing import Any

logger = logging.getLogger(__name__)

_TRACER_NAME = "landmarkdiff"
_TRACER_VERSION = "0.2.0"


class _NoOpSpan:
    """Minimal span stub when OpenTelemetry is not available."""

    def set_attribute(self, key: str, value: Any) -> None:
        pass

    def set_status(self, status: Any) -> None:
        pass

    def record_exception(self, exc: BaseException) -> None:
        pass

    def end(self) -> None:
        pass

    def __enter__(self) -> _NoOpSpan:
        return self

    def __exit__(self, *args: Any) -> None:
        pass


class _NoOpTracer:
    """Minimal tracer stub when OpenTelemetry is not available."""

    def start_span(self, name: str, **kwargs: Any) -> _NoOpSpan:
        return _NoOpSpan()

    def start_as_current_span(self, name: str, **kwargs: Any) -> _NoOpSpan:
        return _NoOpSpan()


def get_tracer(name: str = _TRACER_NAME) -> Any:
    """Get an OpenTelemetry tracer, falling back to a no-op stub.

    Args:
        name: Tracer instrumentation name.

    Returns:
        An OpenTelemetry Tracer or a no-op stub.
    """
    try:
        from opentelemetry import trace

        return trace.get_tracer(name, _TRACER_VERSION)
    except ImportError:
        logger.debug("OpenTelemetry not installed, using no-op tracer")
        return _NoOpTracer()


@contextmanager
def trace_stage(
    tracer: Any,
    stage_name: str,
    *,
    procedure: str = "",
    resolution: int = 0,
    intensity: float = 0.0,
    attributes: dict[str, Any] | None = None,
) -> Generator[Any, None, None]:
    """Context manager that wraps a pipeline stage in a tracing span.

    Records timing, procedure metadata, and any exceptions as span events.
    Works with both real OpenTelemetry tracers and the no-op stub.

    Args:
        tracer: Tracer instance from get_tracer().
        stage_name: Name for the span (e.g. "landmark_extraction").
        procedure: Surgical procedure type.
        resolution: Image resolution in pixels.
        intensity: Procedure intensity (0-100).
        attributes: Extra key-value attributes to record.

    Yields:
        The span object (real or no-op).
    """
    span = tracer.start_span(f"landmarkdiff.{stage_name}")
    start = time.monotonic()

    try:
        if procedure:
            span.set_attribute("landmarkdiff.procedure", procedure)
        if resolution > 0:
            span.set_attribute("landmarkdiff.resolution", resolution)
        if intensity > 0:
            span.set_attribute("landmarkdiff.intensity", intensity)

        if attributes:
            for key, val in attributes.items():
                span.set_attribute(f"landmarkdiff.{key}", val)

        yield span

        elapsed_ms = (time.monotonic() - start) * 1000
        span.set_attribute("landmarkdiff.duration_ms", round(elapsed_ms, 2))

    except Exception as exc:
        elapsed_ms = (time.monotonic() - start) * 1000
        span.set_attribute("landmarkdiff.duration_ms", round(elapsed_ms, 2))
        with contextlib.suppress(Exception):
            span.record_exception(exc)
            span.set_attribute("landmarkdiff.error", str(exc))
        raise
    finally:
        span.end()


# Standard pipeline stage names for consistent instrumentation
PIPELINE_SPAN_NAMES = (
    "model_loading",
    "landmark_extraction",
    "displacement_computation",
    "mesh_deformation",
    "controlnet_conditioning",
    "diffusion_inference",
    "vae_decode",
    "postprocessing",
    "face_restoration",
    "upscaling",
    "compositing",
    "output_encoding",
)
