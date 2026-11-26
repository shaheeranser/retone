"""The enhance pipeline: thumbnail, ask the model, clamp, render, and report.

This is the library the CLI and a future GUI both call. It owns no argument parsing
and writes nothing to stdout; it returns a result and the warnings to show.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
from numpy.typing import NDArray

from retone.cli.compare import compare
from retone.cli.errors import CliError
from retone.cli.exit_codes import ExitCode
from retone.cli.naming import default_output
from retone.executor import Params, apply, load_image, save_image
from retone.executor.errors import ImageError
from retone.executor.icc import is_non_srgb
from retone.inference import (
    InferenceClient,
    ModelUnavailableError,
    thumbnail,
    validate_model_output,
)


@dataclass(frozen=True)
class EnhanceResult:
    """What one input produced: the paths and the parameters that were rendered."""

    input_path: Path
    output_path: Path | None
    params: Params
    warnings: tuple[str, ...]


def choose_params(
    image: NDArray[np.integer[Any]],
    client: InferenceClient,
    *,
    explicit: Params | None = None,
) -> tuple[Params, tuple[str, ...]]:
    """Return the parameters to render and any warnings.

    When `explicit` parameters are given the model is never called, which is what
    `--set` and apply-explicit-params use.
    """
    if explicit is not None:
        return explicit, ()
    shot = thumbnail(image)
    try:
        raw = client.choose_params(shot)
    except ModelUnavailableError as err:
        raise CliError(str(err), ExitCode.MODEL) from err
    validated = validate_model_output(raw)
    return validated.params, validated.warnings


def enhance_image(
    image: NDArray[np.integer[Any]],
    client: InferenceClient,
    *,
    explicit: Params | None = None,
) -> tuple[NDArray[np.integer[Any]], Params, tuple[str, ...]]:
    """Produce the enhanced render, its parameters and warnings, from a loaded image."""
    params, warnings = choose_params(image, client, explicit=explicit)
    return apply(image, params), params, warnings


def resolve_output(
    input_path: Path,
    requested: Path | None,
    *,
    overwrite: bool,
) -> Path:
    """Pick the destination and refuse to clobber the input or an existing file."""
    output = requested if requested is not None else default_output(input_path)
    if output.resolve() == input_path.resolve():
        raise CliError(
            f"refusing to overwrite the input {input_path}; choose another output path",
            ExitCode.USAGE,
        )
    if output.exists() and not overwrite:
        raise CliError(
            f"{output} already exists; pass --force to overwrite",
            ExitCode.WRITE,
        )
    return output


def enhance_file(
    input_path: Path,
    client: InferenceClient,
    *,
    output: Path | None = None,
    write: bool = True,
    overwrite: bool = False,
    explicit: Params | None = None,
    compare_style: str | None = None,
) -> EnhanceResult:
    """Load one photo, choose parameters, render it and optionally write the result."""
    original = _load(input_path)
    rendered, params, warnings = enhance_image(original, client, explicit=explicit)
    warnings = _profile_warnings(input_path) + warnings

    if not write:
        return EnhanceResult(
            input_path=input_path, output_path=None, params=params, warnings=warnings
        )

    output_path = resolve_output(input_path, output, overwrite=overwrite)
    _save(output_path, rendered)
    if compare_style is not None:
        _write_comparisons(original, rendered, output_path, compare_style)
    return EnhanceResult(
        input_path=input_path, output_path=output_path, params=params, warnings=warnings
    )


_COMPARE_STYLES = {"both": ("side-by-side", "split")}


def _write_comparisons(
    original: NDArray[np.integer[Any]],
    rendered: NDArray[np.integer[Any]],
    output: Path,
    compare_style: str,
) -> None:
    styles = _COMPARE_STYLES.get(compare_style, (compare_style,))
    for style in styles:
        _save(_compare_path(output, style), compare(original, rendered, style))


def _compare_path(output: Path, style: str) -> Path:
    tag = "compare" if style == "side-by-side" else "split"
    return output.with_name(f"{output.stem}_{tag}{output.suffix}")


def _load(path: Path) -> NDArray[np.integer[Any]]:
    try:
        return load_image(path)
    except ImageError as err:
        raise CliError(str(err), ExitCode.INPUT) from err


def _profile_warnings(path: Path) -> tuple[str, ...]:
    # The executor assumes sRGB and never converts; a colour-managed input would be
    # rendered incorrectly, so say so rather than silently changing the pixels.
    if is_non_srgb(path):
        return (f"{path} has a non-sRGB ICC profile; interpreting it as sRGB, colours may shift",)
    return ()


def _save(path: Path, image: NDArray[np.integer[Any]]) -> None:
    try:
        save_image(path, image)
    except ImageError as err:
        raise CliError(str(err), ExitCode.WRITE) from err
