"""
benchmark.py — Timing, memory, and quality metrics benchmark command.

Runs the full generate→validate pipeline N times and reports:
  - mean/std time per image
  - peak RAM on local CPU
  - model size on disk
  - job success rate
  - quality metrics aggregated from spec_delta / adherence results

Designed to be repeatable: same seeds, same specs, same pipeline.
"""
from __future__ import annotations

import json
import time
import tracemalloc
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable


@dataclass
class BenchmarkResult:
    repetitions: int
    successful: int
    failed: int
    job_success_rate: float
    times_sec: list[float]
    mean_time_sec: float
    std_time_sec: float
    peak_ram_mb: float
    model_size_mb: float | None
    quality_metrics: dict = field(default_factory=dict)

    def to_dict(self) -> dict:
        return {
            "repetitions":      self.repetitions,
            "successful":       self.successful,
            "failed":           self.failed,
            "job_success_rate": self.job_success_rate,
            "times_sec": {
                "all":  [round(t, 3) for t in self.times_sec],
                "mean": round(self.mean_time_sec, 3),
                "std":  round(self.std_time_sec, 3),
            },
            "peak_ram_mb":     round(self.peak_ram_mb, 1),
            "model_size_mb":   self.model_size_mb,
            "quality_metrics": self.quality_metrics,
        }

    def summary_table(self) -> str:
        msize_str = f"{self.model_size_mb} MB" if self.model_size_mb else "N/A"
        lines = [
            "┌─────────────────────────────────────────────────┐",
            "│              BENCHMARK RESULTS                  │",
            "├─────────────────────────────────────────────────┤",
            f"│  Repetitions:       {self.repetitions:<28}│",
            f"│  Successful:        {self.successful:<28}│",
            f"│  Job success rate:  {self.job_success_rate:.1%}{'':>22}│",
            f"│  Mean time/image:   {self.mean_time_sec:.3f}s{'':>23}│",
            f"│  Std time/image:    {self.std_time_sec:.3f}s{'':>23}│",
            f"│  Peak RAM:          {self.peak_ram_mb:.1f} MB{'':>22}│",
            f"│  Model size:        {msize_str:<28}│",
            "└─────────────────────────────────────────────────┘",
        ]
        return "\n".join(lines)


def measure_model_size(cache_dir: str | Path | None) -> float | None:
    """Return the total size of model files in cache_dir in MB, or None."""
    if not cache_dir:
        return None
    cache = Path(cache_dir)
    if not cache.exists():
        return None
    total = sum(f.stat().st_size for f in cache.rglob("*") if f.is_file())
    return round(total / 1e6, 1)


def run_benchmark(
    generate_fn: Callable[[], dict],   # callable that runs one full job, returns metrics dict
    repetitions: int = 3,
    warmup_runs: int = 1,
    cache_dir: str | Path | None = None,
    output_path: Path | None = None,
) -> BenchmarkResult:
    """Execute generate_fn multiple times and aggregate performance metrics.

    Args:
        generate_fn:  Callable that runs one full avatar generation job.
                      Must return a dict with at least {"success": bool}.
        repetitions:  Number of benchmark repetitions (after warmup).
        warmup_runs:  Number of warmup runs before timing starts.
        cache_dir:    Directory containing model weights (for size measurement).
        output_path:  If given, write JSON results to this path.

    Returns:
        BenchmarkResult with aggregated statistics.
    """
    import numpy as np

    # Warmup
    for _ in range(warmup_runs):
        try:
            generate_fn()
        except Exception:
            pass

    times: list[float] = []
    successful = 0
    failed = 0
    quality_accumulator: dict[str, list[float]] = {}

    # RAM tracking
    tracemalloc.start()

    for _ in range(repetitions):
        t0 = time.perf_counter()
        try:
            metrics = generate_fn()
            elapsed = time.perf_counter() - t0
            times.append(elapsed)
            if metrics.get("success", True):
                successful += 1
                # Accumulate any numeric quality metrics
                for k, v in metrics.items():
                    if isinstance(v, (int, float)) and k != "success":
                        quality_accumulator.setdefault(k, []).append(float(v))
            else:
                failed += 1
        except Exception:
            failed += 1
            elapsed = time.perf_counter() - t0
            times.append(elapsed)

    _, peak = tracemalloc.get_traced_memory()
    tracemalloc.stop()
    peak_ram_mb = peak / 1e6

    times_arr = np.array(times) if times else np.array([0.0])
    quality_metrics = {
        k: {"mean": round(float(np.mean(v)), 4), "std": round(float(np.std(v)), 4)}
        for k, v in quality_accumulator.items()
    }

    result = BenchmarkResult(
        repetitions=repetitions,
        successful=successful,
        failed=failed,
        job_success_rate=successful / max(repetitions, 1),
        times_sec=list(times),
        mean_time_sec=float(times_arr.mean()),
        std_time_sec=float(times_arr.std()),
        peak_ram_mb=round(peak_ram_mb, 1),
        model_size_mb=measure_model_size(cache_dir),
        quality_metrics=quality_metrics,
    )

    if output_path:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(json.dumps(result.to_dict(), indent=2), encoding="utf-8")

    return result
