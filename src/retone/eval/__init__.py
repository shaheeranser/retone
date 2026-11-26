"""Colour metrics and benchmarks: CIEDE2000, parameter error and the ceiling check."""

from retone.eval.benchmark import Benchmark, measure, peak_rss_bytes
from retone.eval.ceiling import CeilingResult, reconstruction_residual
from retone.eval.delta_e import ciede2000_from_lab, mean_delta_e, srgb_to_lab
from retone.eval.human import (
    HumanPair,
    HumanReport,
    load_human_pairs,
    render_predictions,
)
from retone.eval.predictions import PredictionScore, parameter_mae, score_predictions
from retone.eval.results import ResultRow, results_markdown, write_results_csv
from retone.eval.review import ReviewBundle, build_review_bundle
from retone.eval.similarity import psnr, ssim
from retone.eval.thumbnail_validity import (
    ThumbnailValidity,
    thumbnail_validity,
)

__all__ = [
    "Benchmark",
    "CeilingResult",
    "HumanPair",
    "HumanReport",
    "PredictionScore",
    "ResultRow",
    "ReviewBundle",
    "ThumbnailValidity",
    "build_review_bundle",
    "ciede2000_from_lab",
    "load_human_pairs",
    "mean_delta_e",
    "measure",
    "parameter_mae",
    "peak_rss_bytes",
    "psnr",
    "reconstruction_residual",
    "render_predictions",
    "results_markdown",
    "score_predictions",
    "srgb_to_lab",
    "ssim",
    "thumbnail_validity",
    "write_results_csv",
]
