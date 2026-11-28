#!/usr/bin/env python3
"""Run one served GGUF over the human pairs and record fidelity, latency and RAM.

Operational wiring like benchmark.py: serve one quant, run this, repeat per
quant, then compare the JSON files (f16 is the "before", Q8_0/Q4_K_M the "after").

    python scripts/quant_compare.py --label Q4_K_M --gguf ~/.retone/model/model-Q4_K_M.gguf \\
        --server-pid <pid> --out artifacts/eval/quant/2026-11-28
"""

from __future__ import annotations

import argparse
import json
import statistics
import time
from collections.abc import Sequence
from dataclasses import asdict
from pathlib import Path
from typing import cast

import numpy as np
from numpy.typing import NDArray

from retone.eval.human import load_human_pairs, render_predictions
from retone.executor import Params, load_image
from retone.inference import LlamaServerClient, thumbnail, validate_model_output


def _vmhwm(pid: int) -> int:
    for line in Path(f"/proc/{pid}/status").read_text().splitlines():
        if line.startswith("VmHWM:"):
            return int(line.split()[1]) * 1024
    raise SystemExit(f"no VmHWM for pid {pid}")


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--label", required=True)
    parser.add_argument("--gguf", type=Path, required=True)
    parser.add_argument("--pairs", type=Path, default=Path("artifacts/data/human/pairs.jsonl"))
    parser.add_argument("--server-url", default="http://127.0.0.1:8080")
    parser.add_argument("--server-pid", type=int, required=True)
    parser.add_argument("--timeout", type=float, default=600.0)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args(argv)

    client = LlamaServerClient(base_url=args.server_url, timeout=args.timeout)
    pairs = load_human_pairs(args.pairs)
    predictions: dict[str, Params] = {}
    seconds: list[float] = []
    for pair in pairs:
        image = cast(NDArray[np.uint8], thumbnail(load_image(pair.before)))
        start = time.perf_counter()
        predictions[pair.pair_id] = validate_model_output(client.choose_params(image)).params
        seconds.append(time.perf_counter() - start)
        print(f"{args.label} {pair.pair_id} {seconds[-1]:.1f}s", flush=True)

    args.out.mkdir(parents=True, exist_ok=True)
    report = render_predictions(pairs, predictions, args.out / f"render-{args.label}")
    # Cold first request pays prompt prefill; report it apart from the rest.
    summary = {
        "label": args.label,
        "gguf_bytes": args.gguf.stat().st_size,
        "peak_rss_bytes": _vmhwm(args.server_pid),
        "mean_s": statistics.mean(seconds),
        "median_s": statistics.median(seconds),
        "mean_delta_e": report.mean_delta_e,
        "median_delta_e": report.median_delta_e,
        "per_pair_delta_e": {s.pair_id: s.delta_e for s in report.per_pair},
        "params": {
            k: asdict(v) if hasattr(v, "__dataclass_fields__") else v
            for k, v in predictions.items()
        },
    }
    (args.out / f"{args.label}.json").write_text(json.dumps(summary, indent=2, default=str))
    print(json.dumps({k: v for k, v in summary.items() if k not in ("params", "per_pair_delta_e")}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
