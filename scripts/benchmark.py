#!/usr/bin/env python3
"""Measure the real quantized model on the target laptop (04 T-3.1, T-3.2, T-4.1).

This script is the wiring seam: the eval harnesses take injected callables and do
not import an inference client, so the real `LlamaServerClient` is composed here.
It is operational, not part of the package.

    MODEL_DIR=~/.retone/model scripts/serve.sh &
    python scripts/benchmark.py --image photo24mp.jpg \\
        --thumbnail-image photo24mp.jpg --repeats 3
"""

from __future__ import annotations

import argparse
import json
from collections.abc import Sequence
from dataclasses import asdict
from pathlib import Path
from typing import Any, cast

import numpy as np
from numpy.typing import NDArray

from retone.eval import measure, thumbnail_validity
from retone.executor import Params, load_image
from retone.inference import LlamaServerClient, thumbnail, validate_model_output

Image = NDArray[np.integer[Any]]


def _choose(client: LlamaServerClient, image: Image) -> Params:
    # The client re-encodes any integer array, so a 16-bit photo is fine here.
    encoded = cast(NDArray[np.uint8], image)
    return validate_model_output(client.choose_params(encoded)).params


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Benchmark the real model pipeline")
    parser.add_argument("--image", type=Path, required=True, help="full-resolution photo")
    parser.add_argument(
        "--thumbnail-image",
        action="append",
        type=Path,
        default=[],
        help="photo for the thumbnail-validity check; repeatable",
    )
    parser.add_argument("--server-url", default="http://127.0.0.1:8080")
    parser.add_argument("--repeats", type=int, default=1)
    args = parser.parse_args(argv)

    client = LlamaServerClient(base_url=args.server_url)
    image = load_image(args.image)

    def pipeline_fit(passed: Image) -> Params:
        return _choose(client, thumbnail(passed))

    report: dict[str, object] = {
        "benchmark": asdict(measure(image, pipeline_fit, repeats=args.repeats))
    }

    if args.thumbnail_image:
        images = [load_image(path) for path in args.thumbnail_image]
        report["thumbnail_validity"] = asdict(
            thumbnail_validity(images, lambda passed: _choose(client, passed), thumbnail)
        )

    print(json.dumps(report, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
