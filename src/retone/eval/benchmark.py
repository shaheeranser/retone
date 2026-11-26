"""Measure model latency, render latency and peak memory on the target machine.

The model step is an injected callable so this module measures the real pipeline
without importing an inference client. Times are measured, never estimated.
"""

from __future__ import annotations

import sys
import time
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

import numpy as np
from numpy.typing import NDArray

from retone.executor import Params, apply
from retone.executor.errors import ImageError

Fit = Callable[[NDArray[np.integer[Any]]], Params]
Clock = Callable[[], float]
PeakMemory = Callable[[], int]


@dataclass(frozen=True)
class Benchmark:
    """One measured configuration: both stages, on an image of known size."""

    runs: int
    megapixels: float
    mean_model_seconds: float
    mean_render_seconds: float
    peak_rss_bytes: int

    @property
    def mean_total_seconds(self) -> float:
        return self.mean_model_seconds + self.mean_render_seconds


def peak_rss_bytes() -> int:
    """Peak resident set size in bytes, or 0 where the platform cannot report it."""
    try:
        import resource
    except ImportError:
        return 0
    value = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    # Linux reports KiB, macOS reports bytes.
    return value if sys.platform == "darwin" else value * 1024


def measure(
    image: NDArray[np.integer[Any]],
    fit: Fit,
    *,
    repeats: int = 1,
    clock: Clock = time.perf_counter,
    peak: PeakMemory = peak_rss_bytes,
) -> Benchmark:
    """Run `fit` then `apply` on the full-resolution image and report both times.

    `fit` receives the full-resolution image and is expected to do only the model
    step (thumbnail and request); the render is timed separately because it scales
    with megapixels while the model sees a fixed-size thumbnail.
    """
    if repeats < 1:
        raise ValueError(f"repeats must be positive, got {repeats}")
    if image.ndim != 3 or image.shape[2] != 3:
        raise ImageError(f"expected an HxWx3 image, got shape {image.shape}")

    model_total = 0.0
    render_total = 0.0
    for _ in range(repeats):
        started = clock()
        params = fit(image)
        model_done = clock()
        apply(image, params)
        render_total += clock() - model_done
        model_total += model_done - started

    height, width = image.shape[:2]
    return Benchmark(
        runs=repeats,
        megapixels=height * width / 1_000_000.0,
        mean_model_seconds=model_total / repeats,
        mean_render_seconds=render_total / repeats,
        peak_rss_bytes=peak(),
    )
