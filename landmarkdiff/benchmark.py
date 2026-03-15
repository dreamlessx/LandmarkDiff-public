"""Inference benchmarking for deployment sizing.

Measures throughput, latency, and memory usage for ControlNet inference
under various configurations (resolution, batch size, denoising steps).

Usage:
    from landmarkdiff.benchmark import InferenceBenchmark

    bench = InferenceBenchmark()
    bench.add_result("gpu_a6000", latency_ms=142.3, throughput_fps=7.0, vram_gb=4.2)
    bench.add_result("gpu_a6000", latency_ms=138.1, throughput_fps=7.2, vram_gb=4.2)
    print(bench.summary())
"""

from __future__ import annotations

import json
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any


@dataclass
class BenchmarkResult:
    """A single benchmark measurement."""

    config_name: str
    latency_ms: float
    throughput_fps: float = 0.0
    vram_gb: float = 0.0
    batch_size: int = 1
    resolution: int = 512
    num_inference_steps: int = 20
    device: str = ""
    metadata: dict[str, Any] = field(default_factory=dict)


class InferenceBenchmark:
    """Collect and analyze inference benchmarks.

    Args:
        model_name: Name of the model being benchmarked.
    """

    def __init__(self, model_name: str = "LandmarkDiff-ControlNet") -> None:
        self.model_name = model_name
        self.results: list[BenchmarkResult] = []

    def add_result(
        self,
        config_name: str,
        latency_ms: float,
        throughput_fps: float = 0.0,
        vram_gb: float = 0.0,
        batch_size: int = 1,
        resolution: int = 512,
        num_inference_steps: int = 20,
        device: str = "",
        **metadata: Any,
    ) -> None:
        """Add a benchmark result."""
        if throughput_fps == 0.0 and latency_ms > 0:
            throughput_fps = 1000.0 / latency_ms * batch_size

        self.results.append(
            BenchmarkResult(
                config_name=config_name,
                latency_ms=latency_ms,
                throughput_fps=throughput_fps,
                vram_gb=vram_gb,
                batch_size=batch_size,
                resolution=resolution,
                num_inference_steps=num_inference_steps,
                device=device,
                metadata=metadata,
            )
        )

    def mean_latency(self, config_name: str | None = None) -> float:
        """Mean latency in ms, optionally filtered by config."""
        results = self._filter(config_name)
        if not results:
            return float("nan")
        return sum(r.latency_ms for r in results) / len(results)

    def p99_latency(self, config_name: str | None = None) -> float:
        """P99 latency in ms."""
        results = self._filter(config_name)
        if not results:
            return float("nan")
        sorted_latencies = sorted(r.latency_ms for r in results)
        idx = max(0, int(len(sorted_latencies) * 0.99) - 1)
        return sorted_latencies[idx]

    def mean_throughput(self, config_name: str | None = None) -> float:
        """Mean throughput in FPS."""
        results = self._filter(config_name)
        if not results:
            return float("nan")
        return sum(r.throughput_fps for r in results) / len(results)

    def max_vram(self, config_name: str | None = None) -> float:
        """Maximum VRAM usage in GB."""
        results = self._filter(config_name)
        if not results:
            return 0.0
        return max(r.vram_gb for r in results)

    def _filter(self, config_name: str | None) -> list[BenchmarkResult]:
        if config_name is None:
            return self.results
        return [r for r in self.results if r.config_name == config_name]

    @property
    def config_names(self) -> list[str]:
        """Unique config names in order."""
        seen: dict[str, None] = {}
        for r in self.results:
            seen.setdefault(r.config_name, None)
        return list(seen.keys())

    def summary(self) -> str:
        """Generate text summary table."""
        configs = self.config_names
        if not configs:
            return "No benchmark results."

        header = (
            f"{'Config':>20s} | {'Mean(ms)':>10s} | {'P99(ms)':>10s}"
            f" | {'FPS':>8s} | {'VRAM(GB)':>8s} | {'N':>4s}"
        )
        lines = [
            f"Inference Benchmark: {self.model_name}",
            header,
            "-" * len(header),
        ]

        for cfg in configs:
            results = self._filter(cfg)
            lines.append(
                f"{cfg:>20s} | "
                f"{self.mean_latency(cfg):>10.1f} | "
                f"{self.p99_latency(cfg):>10.1f} | "
                f"{self.mean_throughput(cfg):>8.2f} | "
                f"{self.max_vram(cfg):>8.1f} | "
                f"{len(results):>4d}"
            )

        return "\n".join(lines)

    def to_json(self, path: str | Path | None = None) -> str:
        """Export results as JSON."""
        data = {
            "model_name": self.model_name,
            "results": [
                {
                    "config_name": r.config_name,
                    "latency_ms": r.latency_ms,
                    "throughput_fps": round(r.throughput_fps, 2),
                    "vram_gb": r.vram_gb,
                    "batch_size": r.batch_size,
                    "resolution": r.resolution,
                    "num_inference_steps": r.num_inference_steps,
                    "device": r.device,
                }
                for r in self.results
            ],
            "summary": {
                cfg: {
                    "mean_latency_ms": round(self.mean_latency(cfg), 1),
                    "p99_latency_ms": round(self.p99_latency(cfg), 1),
                    "mean_fps": round(self.mean_throughput(cfg), 2),
                    "max_vram_gb": round(self.max_vram(cfg), 1),
                    "n_samples": len(self._filter(cfg)),
                }
                for cfg in self.config_names
            },
        }
        j = json.dumps(data, indent=2)
        if path:
            Path(path).parent.mkdir(parents=True, exist_ok=True)
            Path(path).write_text(j)
        return j


class Timer:
    """Simple context manager for timing code blocks.

    Usage:
        with Timer() as t:
            run_inference()
        print(f"Took {t.elapsed_ms:.1f} ms")
    """

    def __init__(self) -> None:
        self.start_time: float = 0.0
        self.end_time: float = 0.0

    @property
    def elapsed_ms(self) -> float:
        return (self.end_time - self.start_time) * 1000

    @property
    def elapsed_s(self) -> float:
        return self.end_time - self.start_time

    def __enter__(self) -> Timer:
        self.start_time = time.perf_counter()
        return self

    def __exit__(self, *args: Any) -> None:
        self.end_time = time.perf_counter()


# ---------------------------------------------------------------------------
# Pipeline profiler
# ---------------------------------------------------------------------------


@dataclass
class StageProfile:
    """Timing profile for a single pipeline stage."""

    name: str
    times_ms: list[float] = field(default_factory=list)

    @property
    def mean_ms(self) -> float:
        return sum(self.times_ms) / len(self.times_ms) if self.times_ms else 0.0

    @property
    def min_ms(self) -> float:
        return min(self.times_ms) if self.times_ms else 0.0

    @property
    def max_ms(self) -> float:
        return max(self.times_ms) if self.times_ms else 0.0

    @property
    def total_ms(self) -> float:
        return sum(self.times_ms)

    @property
    def count(self) -> int:
        return len(self.times_ms)


class PipelineProfiler:
    """Profile each stage of the LandmarkDiff pipeline.

    Collects per-stage wall-clock timings to identify bottlenecks
    across landmark extraction, manipulation, masking, and inference.

    Usage:
        profiler = PipelineProfiler()
        with profiler.stage("landmark_extraction"):
            face = extract_landmarks(image)
        with profiler.stage("manipulation"):
            modified = apply_procedure_preset(face, "rhinoplasty")
        print(profiler.summary())
    """

    def __init__(self) -> None:
        self._stages: dict[str, StageProfile] = {}
        self._order: list[str] = []

    def stage(self, name: str) -> _StageTimer:
        """Context manager that times a named pipeline stage."""
        if name not in self._stages:
            self._stages[name] = StageProfile(name=name)
            self._order.append(name)
        return _StageTimer(self._stages[name])

    def record(self, name: str, elapsed_ms: float) -> None:
        """Manually record a stage timing."""
        if name not in self._stages:
            self._stages[name] = StageProfile(name=name)
            self._order.append(name)
        self._stages[name].times_ms.append(elapsed_ms)

    @property
    def stages(self) -> list[StageProfile]:
        return [self._stages[n] for n in self._order]

    @property
    def total_ms(self) -> float:
        return sum(s.total_ms for s in self._stages.values())

    def bottleneck(self) -> str | None:
        """Name of the slowest stage by mean time."""
        if not self._stages:
            return None
        return max(self._stages.values(), key=lambda s: s.mean_ms).name

    def summary(self) -> str:
        """Text summary of pipeline stage timings."""
        if not self._stages:
            return "No profile data."

        total = self.total_ms
        header = (
            f"{'Stage':>25s} | {'Mean(ms)':>10s} | {'Min':>8s}"
            f" | {'Max':>8s} | {'%':>6s} | {'N':>4s}"
        )
        lines = ["Pipeline Profile", header, "-" * len(header)]

        for name in self._order:
            s = self._stages[name]
            pct = (s.total_ms / total * 100) if total > 0 else 0
            lines.append(
                f"{s.name:>25s} | "
                f"{s.mean_ms:>10.1f} | "
                f"{s.min_ms:>8.1f} | "
                f"{s.max_ms:>8.1f} | "
                f"{pct:>5.1f}% | "
                f"{s.count:>4d}"
            )

        lines.append("-" * len(header))
        lines.append(f"{'Total':>25s} | {total:>10.1f}")
        bn = self.bottleneck()
        if bn:
            lines.append(f"Bottleneck: {bn}")
        return "\n".join(lines)

    def to_dict(self) -> dict[str, Any]:
        """Export profile data as a dictionary."""
        return {
            "stages": {
                name: {
                    "mean_ms": round(self._stages[name].mean_ms, 2),
                    "min_ms": round(self._stages[name].min_ms, 2),
                    "max_ms": round(self._stages[name].max_ms, 2),
                    "total_ms": round(self._stages[name].total_ms, 2),
                    "count": self._stages[name].count,
                }
                for name in self._order
            },
            "total_ms": round(self.total_ms, 2),
            "bottleneck": self.bottleneck(),
        }

    def reset(self) -> None:
        """Clear all recorded timings."""
        self._stages.clear()
        self._order.clear()


class _StageTimer:
    """Internal context manager for PipelineProfiler.stage()."""

    def __init__(self, profile: StageProfile) -> None:
        self._profile = profile
        self._start: float = 0.0

    def __enter__(self) -> _StageTimer:
        self._start = time.perf_counter()
        return self

    def __exit__(self, *args: Any) -> None:
        elapsed = (time.perf_counter() - self._start) * 1000
        self._profile.times_ms.append(elapsed)


# ---------------------------------------------------------------------------
# Progress reporting
# ---------------------------------------------------------------------------

# Standard pipeline stages with human-readable descriptions and
# approximate percentage of total runtime.
PIPELINE_STAGES: list[tuple[str, str, int]] = [
    ("loading", "Loading model weights", 5),
    ("landmark_extraction", "Extracting facial landmarks", 10),
    ("manipulation", "Applying surgical deformation", 15),
    ("conditioning", "Generating conditioning image", 20),
    ("masking", "Computing surgical mask", 25),
    ("tps_warp", "Warping image (TPS)", 35),
    ("inference", "Running diffusion model", 70),
    ("postprocessing", "Post-processing output", 90),
    ("compositing", "Blending result", 95),
    ("complete", "Done", 100),
]

_STAGE_INDEX = {name: i for i, (name, _, _) in enumerate(PIPELINE_STAGES)}


class ProgressReporter:
    """Callback-based progress reporter for the LandmarkDiff pipeline.

    Provides stage-by-stage progress updates with human-readable
    descriptions and percentage estimates.  Accepts a callback function
    that receives (stage_name, description, percent) on each update.

    Usage:
        def on_progress(stage, desc, pct):
            print(f"[{pct}%] {desc}")

        reporter = ProgressReporter(callback=on_progress)
        reporter.update("landmark_extraction")
        reporter.update("manipulation")
        # ...
        reporter.update("complete")

    For Gradio integration:
        reporter = ProgressReporter(callback=gr.Progress())
    """

    def __init__(
        self,
        callback: Any | None = None,
    ) -> None:
        self._callback = callback
        self._current_stage: str = ""
        self._start_time: float = 0.0
        self._stage_times: dict[str, float] = {}

    def update(self, stage: str, description: str | None = None) -> None:
        """Report progress for a named stage.

        Args:
            stage: Stage identifier (e.g. "landmark_extraction").
                   Unknown stages are accepted with 0% progress.
            description: Override the default stage description.
        """
        now = time.perf_counter()
        if self._start_time == 0:
            self._start_time = now

        # Record timing for previous stage
        if self._current_stage:
            elapsed = (now - self._stage_start) * 1000
            self._stage_times[self._current_stage] = elapsed

        self._current_stage = stage
        self._stage_start = now

        # Look up stage info
        idx = _STAGE_INDEX.get(stage)
        if idx is not None:
            _, default_desc, pct = PIPELINE_STAGES[idx]
        else:
            default_desc = stage.replace("_", " ").title()
            pct = 0

        desc = description or default_desc

        if self._callback is not None:
            try:
                self._callback(stage, desc, pct)
            except TypeError:
                # Gradio Progress() takes (fraction, desc) instead
                import contextlib

                with contextlib.suppress(Exception):
                    self._callback(pct / 100.0, desc=desc)

    @property
    def elapsed_ms(self) -> float:
        """Total elapsed time since first update."""
        if self._start_time == 0:
            return 0.0
        return (time.perf_counter() - self._start_time) * 1000

    @property
    def stage_times(self) -> dict[str, float]:
        """Per-stage timing in milliseconds."""
        return dict(self._stage_times)
