#!/usr/bin/env python3
"""Benchmark the whole retone pipeline on one machine, per thread count.

For each --threads value it starts its own llama-server, then records:
  load      seconds until the server answers /health (cold model load)
  cold      the first request after load (includes prompt and image prefill)
  stages    decode, thumbnail, model, validate and render time per image (warm requests)
  end2end   the sum, which is what a user waits for
  memory    server RSS when idle and its peak (VmHWM)
  quality   mean delta-E to the human edit, when --pairs is given
  validity  answers rejected by validation (these silently become "no change")
  thermals  hottest sensor seen; the run pauses above --max-temp instead of risking a shutdown

    python scripts/bench_pipeline.py --gguf ~/.retone/model/model-Q4_K_M.gguf \\
        --images artifacts/data/human/before --threads 4 8 --out artifacts/eval/bench/run1

stdout carries only the markdown table; progress goes to stderr.
"""

from __future__ import annotations

import argparse
import json
import math
import shutil
import statistics
import subprocess
import sys
import time
import urllib.error
import urllib.request
from collections.abc import Sequence
from pathlib import Path
from typing import Any

from retone.eval.delta_e import mean_delta_e
from retone.eval.human import load_human_pairs
from retone.executor import apply, load_image
from retone.inference import LlamaServerClient, thumbnail, validate_model_output

IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png"}


def log(message: str) -> None:
    print(message, file=sys.stderr, flush=True)


def percentile(values: Sequence[float], q: float) -> float:
    """Nearest-rank percentile; fine for the dozens of samples a run produces."""
    ordered = sorted(values)
    return ordered[max(0, math.ceil(q / 100 * len(ordered)) - 1)]


def summarize(values: Sequence[float]) -> dict[str, float]:
    return {
        "mean": statistics.fmean(values),
        "median": statistics.median(values),
        "p95": percentile(values, 95),
        "min": min(values),
        "max": max(values),
    }


def hottest_celsius() -> float:
    temps = []
    for path in Path("/sys/class/thermal").glob("thermal_zone*/temp"):
        try:
            temps.append(int(path.read_text()) / 1000)
        except (OSError, ValueError):
            continue
    return max(temps, default=0.0)


def cool_down(limit: float, resume_below: float, waited: list[float]) -> None:
    """Block while the machine is hotter than `limit`, so a long run cannot shut it down."""
    if hottest_celsius() <= limit:
        return
    start = time.monotonic()
    log(f"  {hottest_celsius():.0f} C is above {limit:.0f} C, pausing until {resume_below:.0f} C")
    while hottest_celsius() > resume_below:
        time.sleep(5)
    waited.append(time.monotonic() - start)


def server_memory(pid: int) -> dict[str, int]:
    values = {}
    for line in Path(f"/proc/{pid}/status").read_text().splitlines():
        if line.startswith(("VmRSS:", "VmHWM:")):
            values[line.split(":")[0]] = int(line.split()[1]) * 1024
    return values


def wait_ready(url: str, process: subprocess.Popen[bytes], timeout: float = 300.0) -> None:
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if process.poll() is not None:
            raise SystemExit(f"llama-server exited early with code {process.returncode}")
        try:
            with urllib.request.urlopen(f"{url}/health", timeout=2):
                return
        except (urllib.error.URLError, OSError):
            time.sleep(0.5)
    raise SystemExit("llama-server did not become ready in time")


def collect_images(sources: Sequence[Path]) -> list[Path]:
    found: list[Path] = []
    for source in sources:
        files = sorted(source.rglob("*")) if source.is_dir() else [source]
        found += [f for f in files if f.suffix.lower() in IMAGE_SUFFIXES]
    return found


def run_config(args: argparse.Namespace, threads: int, images: list[Path]) -> dict[str, Any]:
    binary = args.llama_server or shutil.which("llama-server")
    if binary is None:
        raise SystemExit("llama-server not found; pass --llama-server or add it to PATH")
    url = f"http://127.0.0.1:{args.port}"
    command = [
        binary, "--model", str(args.gguf), "--mmproj", str(args.mmproj), "--port", str(args.port),
        "--threads", str(threads), "--ctx-size", str(args.ctx), "--n-gpu-layers", "0",
        "--image-min-tokens", str(args.image_min_tokens), "--jinja",
    ]  # fmt: skip
    log(f"== {args.label}: {threads} threads")
    started = time.monotonic()
    server = subprocess.Popen(command, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    try:
        wait_ready(url, server)
        load_s = time.monotonic() - started
        idle = server_memory(server.pid)
        client = LlamaServerClient(base_url=url, timeout=args.timeout)
        waited: list[float] = []
        rows: list[dict[str, float]] = []
        rejected = 0
        pairs = (
            {p.before.resolve(): p.after for p in load_human_pairs(args.pairs)}
            if args.pairs
            else {}
        )
        deltas: list[float] = []
        cold: float | None = None

        def one(path: Path) -> tuple[dict[str, float], Any]:
            cool_down(args.max_temp, args.max_temp - 10, waited)
            t0 = time.perf_counter()
            image = load_image(path)
            t1 = time.perf_counter()
            small = thumbnail(image)
            t2 = time.perf_counter()
            answer = client.choose_params(small)
            t3 = time.perf_counter()
            checked = validate_model_output(answer)
            t4 = time.perf_counter()
            rendered = apply(image, checked.params)
            t5 = time.perf_counter()
            nonlocal rejected
            rejected += any(w.startswith("model output rejected") for w in checked.warnings)
            row = {"decode": t1 - t0, "thumbnail": t2 - t1, "model": t3 - t2,
                   "validate": t4 - t3, "render": t5 - t4, "end2end": t5 - t0}  # fmt: skip
            return row, rendered

        for _ in range(args.warmup):  # first one is the cold request; none count in the stats
            row, _unused = one(images[0])
            cold = row["end2end"] if cold is None else cold
            log(f"  warm-up {images[0].name}: {row['end2end']:.1f}s")
        order = [path for _ in range(args.repeats) for path in images]
        for index, path in enumerate(order):
            row, rendered = one(path)
            rows.append(row)
            if path.resolve() in pairs and index < len(images):
                deltas.append(mean_delta_e(rendered, load_image(pairs[path.resolve()])))
            log(f"  {len(rows)}/{len(order)} {path.name}: {row['end2end']:.1f}s")
        memory = server_memory(server.pid)
    finally:
        server.terminate()
        server.wait(timeout=30)
    stats = {k: summarize([r[k] for r in rows]) for k in rows[0]}
    return {
        "label": args.label, "threads": threads, "images": len(images), "samples": len(rows),
        "load_s": load_s, "cold_first_request_s": cold,
        "stages": stats, "rss_idle_bytes": idle.get("VmRSS"), "rss_peak_bytes": memory.get("VmHWM"),
        "rejected_answers": rejected, "mean_delta_e": statistics.fmean(deltas) if deltas else None,
        "hottest_c": hottest_celsius(), "thermal_pause_s": sum(waited),
    }  # fmt: skip


def table(results: Sequence[dict[str, Any]]) -> str:
    lines = [
        "| threads | load s | model s (median / p95) | render s | end-to-end s (median / p95) "
        "| peak RAM GB | rejected | dE | max C |",
        "|---|---|---|---|---|---|---|---|---|",
    ]
    for r in results:
        s = r["stages"]
        delta = f"{r['mean_delta_e']:.2f}" if r["mean_delta_e"] is not None else "n/a"
        lines.append(
            f"| {r['threads']} | {r['load_s']:.1f} | {s['model']['median']:.1f} / "
            f"{s['model']['p95']:.1f} | {s['render']['median']:.2f} | {s['end2end']['median']:.1f}"
            f" / {s['end2end']['p95']:.1f} | {r['rss_peak_bytes'] / 2**30:.2f} "
            f"| {r['rejected_answers']} | {delta} | {r['hottest_c']:.0f} |"
        )
    return "\n".join(lines)


def selftest() -> None:
    assert percentile([1, 2, 3, 4, 5], 95) == 5 and percentile([1, 2, 3, 4, 5], 50) == 3
    assert summarize([2.0, 4.0])["mean"] == 3.0 and summarize([7.0])["p95"] == 7.0


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawTextHelpFormatter
    )
    parser.add_argument("--gguf", type=Path)
    parser.add_argument("--mmproj", type=Path)
    parser.add_argument("--images", type=Path, nargs="+", default=[])
    parser.add_argument("--pairs", type=Path, help="human pairs.jsonl; adds delta-E to the report")
    parser.add_argument("--threads", type=int, nargs="+", default=[4])
    parser.add_argument("--repeats", type=int, default=1)
    parser.add_argument("--warmup", type=int, default=1, help="requests discarded from the stats")
    parser.add_argument("--label", default="run")
    parser.add_argument("--port", type=int, default=8090)
    parser.add_argument("--ctx", type=int, default=4096)
    parser.add_argument("--image-min-tokens", type=int, default=256)
    parser.add_argument("--timeout", type=float, default=600.0)
    parser.add_argument("--max-temp", type=float, default=88.0, help="pause above this, in C")
    parser.add_argument("--cooldown", type=float, default=60.0, help="seconds between configs")
    parser.add_argument("--llama-server", type=Path)
    parser.add_argument("--out", type=Path)
    parser.add_argument("--selftest", action="store_true")
    args = parser.parse_args(argv)
    if args.selftest:
        selftest()
        log("selftest ok")
        return 0
    if not (args.gguf and args.images and args.out):
        parser.error("--gguf, --images and --out are required")
    args.mmproj = args.mmproj or args.gguf.with_name("mmproj-f16.gguf")
    images = collect_images(args.images)
    if not images:
        parser.error("no .jpg/.jpeg/.png images found")
    results = []
    for position, threads in enumerate(args.threads):
        if position:
            log(f"cooling down {args.cooldown:.0f}s")
            time.sleep(args.cooldown)
        results.append(run_config(args, threads, images))
    args.out.mkdir(parents=True, exist_ok=True)
    (args.out / f"{args.label}.json").write_text(json.dumps(results, indent=2))
    report = table(results)
    (args.out / f"{args.label}.md").write_text(report + "\n")
    print(report)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
