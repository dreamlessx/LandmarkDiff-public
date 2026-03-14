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

        self.results.append(BenchmarkResult(
            config_name=config_name,
            latency_ms=latency_ms,
            throughput_fps=throughput_fps,
            vram_gb=vram_gb,
            batch_size=batch_size,
            resolution=resolution,
            num_inference_steps=num_inference_steps,
            device=device,
            metadata=metadata,
        ))

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

        header = f"{'Config':>20s} | {'Mean(ms)':>10s} | {'P99(ms)':>10s} | {'FPS':>8s} | {'VRAM(GB)':>8s} | {'N':>4s}"
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
