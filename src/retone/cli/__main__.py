"""The `retone` command: flags, stream routing and exit codes over the library pipeline."""

from __future__ import annotations

import argparse
import json
import sys
from collections.abc import Sequence
from pathlib import Path
from typing import TextIO

from retone.cli.errors import CliError
from retone.cli.exit_codes import ExitCode
from retone.cli.render import render_json, render_table
from retone.cli.run import EnhanceResult, enhance_file
from retone.executor import ParamError, Params, parse_params
from retone.inference import InferenceClient, LlamaServerClient

COMPARE_STYLES = ("side-by-side", "split", "both")
DEFAULT_SERVER_URL = "http://127.0.0.1:8080"


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="retone",
        description="Enhance photos offline: the model picks values, OpenCV applies them.",
    )
    parser.add_argument("inputs", nargs="+", type=Path, help="one or more image paths")
    parser.add_argument(
        "-o", "--out", type=Path, default=None, help="output path (single input only)"
    )
    parser.add_argument(
        "--params", action="store_true", help="print the chosen parameters without writing an image"
    )
    parser.add_argument(
        "--json", action="store_true", help="print the chosen parameters as JSON and write nothing"
    )
    parser.add_argument(
        "--compare",
        choices=COMPARE_STYLES,
        default=None,
        help="also write a before/after image",
    )
    parser.add_argument(
        "--set",
        action="append",
        default=[],
        metavar="NAME=VALUE",
        help="apply explicit parameters without the model; repeat per parameter",
    )
    parser.add_argument("--force", action="store_true", help="overwrite an existing output file")
    parser.add_argument(
        "--server-url",
        default=DEFAULT_SERVER_URL,
        help=f"llama-server base URL (default {DEFAULT_SERVER_URL})",
    )
    return parser


def _explicit_params(pairs: Sequence[str]) -> Params | None:
    if not pairs:
        return None
    raw: dict[str, object] = {}
    for pair in pairs:
        name, separator, value = pair.partition("=")
        if not separator or not name:
            raise CliError(f"--set {pair!r} must look like name=value", ExitCode.USAGE)
        try:
            raw[name] = int(value)
        except ValueError as err:
            raise CliError(f"--set {pair!r}: {value!r} is not an integer", ExitCode.USAGE) from err
    try:
        return parse_params(raw).params
    except ParamError as err:
        raise CliError(f"invalid --set parameters: {err}", ExitCode.USAGE) from err


def _report(
    results: Sequence[EnhanceResult], *, as_json: bool, as_table: bool, out: TextIO
) -> None:
    multiple = len(results) > 1
    for result in results:
        if as_json:
            if multiple:
                payload = {"input": str(result.input_path), "params": result.params.as_dict()}
                out.write(json.dumps(payload, sort_keys=True) + "\n")
            else:
                render_json(result.params, out)
        elif as_table:
            if multiple:
                out.write(f"{result.input_path}:\n")
            render_table(result.params, out)
        elif result.output_path is not None:
            out.write(f"{result.output_path}\n")


def main(
    argv: Sequence[str] | None = None,
    *,
    client: InferenceClient | None = None,
    out: TextIO | None = None,
    err: TextIO | None = None,
) -> int:
    """Parse arguments, run the pipeline and return a stable exit code.

    `client`, `out` and `err` are injection seams for tests; the real command uses a
    `LlamaServerClient`, stdout and stderr.
    """
    stdout = out if out is not None else sys.stdout
    stderr = err if err is not None else sys.stderr
    args = _parser().parse_args(argv)
    try:
        explicit = _explicit_params(args.set)
        if args.out is not None and len(args.inputs) > 1:
            raise CliError(
                "--out takes a single input; omit it to write beside each", ExitCode.USAGE
            )
        results = _run(args, client, explicit)
    except CliError as failure:
        stderr.write(f"retone: {failure.message}\n")
        return int(failure.code)
    for result in results:
        for warning in result.warnings:
            stderr.write(f"retone: {warning}\n")
    _report(results, as_json=args.json, as_table=args.params, out=stdout)
    return int(ExitCode.SUCCESS)


def _run(
    args: argparse.Namespace, client: InferenceClient | None, explicit: Params | None
) -> list[EnhanceResult]:
    active = client if client is not None else LlamaServerClient(base_url=args.server_url)
    write = not (args.params or args.json)
    results: list[EnhanceResult] = []
    for index, input_path in enumerate(args.inputs):
        output = args.out if index == 0 else None
        results.append(
            enhance_file(
                input_path,
                active,
                output=output,
                write=write,
                overwrite=args.force,
                explicit=explicit,
                compare_style=args.compare,
            )
        )
    return results


if __name__ == "__main__":
    raise SystemExit(main())
