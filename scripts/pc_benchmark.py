# ruff: noqa: E501  (inline HTML strings and long help lines)
#!/usr/bin/env python3
"""Run f16, Q8_0 and Q4_K_M the way a normal PC would, then write one HTML report.

    python scripts/pc_benchmark.py prepare            # gather camera + phone photos in one folder
    python scripts/pc_benchmark.py run                # benchmark all three, every core, write report

`run` starts one llama-server per quantization with all logical cores (override with --threads),
sends every photo through the real pipeline (decode, thumbnail, model, validate, full-resolution
render) and records, per model: load time, cold first request, per-image time (median, p95),
RAM (idle and peak), CPU cores actually used, hottest temperature, and delta-E to your edit for
the camera photos. The report is <out>/index.html with a summary table and original / reference /
f16 / Q8_0 / Q4_K_M side by side. The run pauses if the machine gets too hot (--max-temp).
"""

from __future__ import annotations

import argparse
import html
import json
import os
import shutil
import statistics
import subprocess
import sys
import threading
import time
from collections.abc import Sequence
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np
from bench_pipeline import cool_down, hottest_celsius, log, summarize, wait_ready
from PIL import Image

from retone.eval.delta_e import mean_delta_e
from retone.executor import Params, apply, load_image
from retone.inference import LlamaServerClient, thumbnail, validate_model_output

QUANTS = ("f16", "Q8_0", "Q4_K_M")
IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png"}
CORPUS = Path("artifacts/eval/corpus")
_TICK = os.sysconf("SC_CLK_TCK")


def _link_or_copy(source: Path, target: Path) -> None:
    target.parent.mkdir(parents=True, exist_ok=True)
    if target.exists():
        return
    try:
        os.link(source, target)  # same disk: no extra space, and the source stays untouched
    except OSError:
        shutil.copy2(source, target)


def prepare(args: argparse.Namespace) -> int:
    """Build <corpus>/images, <corpus>/reference and manifest.json from both photo sources."""
    from retone.eval.human import load_human_pairs

    entries: list[dict[str, Any]] = []
    for pair in load_human_pairs(args.camera_pairs):
        name = f"camera_{pair.pair_id}{pair.before.suffix.lower()}"
        reference = f"camera_{pair.pair_id}{pair.after.suffix.lower()}"
        _link_or_copy(pair.before, args.corpus / "images" / name)
        _link_or_copy(pair.after, args.corpus / "reference" / reference)
        entries.append(
            {"file": f"images/{name}", "kind": "camera", "reference": f"reference/{reference}"}
        )
    phone = sorted(p for p in args.phone.rglob("*") if p.suffix.lower() in IMAGE_SUFFIXES)
    for path in phone:
        name = f"phone_{path.name}"
        _link_or_copy(path, args.corpus / "images" / name)
        entries.append({"file": f"images/{name}", "kind": "phone", "reference": None})
    (args.corpus / "manifest.json").write_text(json.dumps(entries, indent=2))
    cameras = sum(e["kind"] == "camera" for e in entries)
    print(f"{args.corpus}: {cameras} camera + {len(phone)} phone photos")
    return 0


class Sampler(threading.Thread):
    """Polls the server's CPU time, RSS and the machine temperature while requests run."""

    def __init__(self, pid: int) -> None:
        super().__init__(daemon=True)
        self.pid, self.stop = pid, threading.Event()
        self.rss_peak, self.temp_peak = 0, 0.0
        self.cpu_start = self._cpu()
        self.wall_start = time.monotonic()

    def _cpu(self) -> float:
        fields = Path(f"/proc/{self.pid}/stat").read_text().rsplit(")", 1)[1].split()
        return (int(fields[11]) + int(fields[12])) / _TICK  # utime + stime

    def run(self) -> None:
        while not self.stop.wait(0.5):
            try:
                for line in Path(f"/proc/{self.pid}/status").read_text().splitlines():
                    if line.startswith("VmRSS:"):
                        self.rss_peak = max(self.rss_peak, int(line.split()[1]) * 1024)
                self.temp_peak = max(self.temp_peak, hottest_celsius())
            except OSError:
                return

    def cores_used(self) -> float:
        return (self._cpu() - self.cpu_start) / (time.monotonic() - self.wall_start)


def thumb(source: Path | np.ndarray[Any, Any], target: Path) -> str:
    image = Image.open(source) if isinstance(source, Path) else Image.fromarray(source)
    image = image.convert("RGB")
    image.thumbnail((640, 640))
    target.parent.mkdir(parents=True, exist_ok=True)
    image.save(target, quality=80)
    return str(target)


def fmt(params: Params) -> str:
    return ", ".join(f"{k} {v:+d}" for k, v in params.as_dict().items() if v) or "no change"


def run_model(
    args: argparse.Namespace, quant: str, entries: list[dict[str, Any]]
) -> dict[str, Any]:
    gguf = args.model_dir / f"model-{quant}.gguf"
    mmproj = args.model_dir / "mmproj-f16.gguf"
    binary = args.llama_server or shutil.which("llama-server")
    if binary is None or not gguf.exists() or not mmproj.exists():
        raise SystemExit(f"need llama-server on PATH (or --llama-server) plus {gguf} and {mmproj}")
    url = f"http://127.0.0.1:{args.port}"
    command = [
        str(binary), "--model", str(gguf), "--mmproj", str(mmproj), "--port", str(args.port),
        "--threads", str(args.threads), "--ctx-size", str(args.ctx), "--n-gpu-layers", "0",
        "--image-min-tokens", "256", "--jinja",
    ]  # fmt: skip
    log(f"== {quant}: {args.threads} threads")
    started = time.monotonic()
    server = subprocess.Popen(command, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    try:
        wait_ready(url, server)
        load_s = time.monotonic() - started
        idle = next(
            int(line.split()[1]) * 1024
            for line in Path(f"/proc/{server.pid}/status").read_text().splitlines()
            if line.startswith("VmRSS:")
        )
        client = LlamaServerClient(base_url=url, timeout=args.timeout)
        waited: list[float] = []

        def one(entry: dict[str, Any]) -> tuple[dict[str, float], Any, Any]:
            cool_down(args.max_temp, args.max_temp - 10, waited)
            t0 = time.perf_counter()
            image = load_image(args.corpus / entry["file"])
            t1 = time.perf_counter()
            small = thumbnail(image)
            t2 = time.perf_counter()
            answer = client.choose_params(small)
            t3 = time.perf_counter()
            checked = validate_model_output(answer)
            t4 = time.perf_counter()
            rendered = apply(image, checked.params)
            t5 = time.perf_counter()
            row = {"model": t3 - t2, "render": t5 - t4, "end2end": t5 - t0,
                   "decode": t1 - t0, "rejected": float(bool(checked.warnings))}  # fmt: skip
            return row, checked, rendered

        row, _unused, _image = one(entries[0])  # warm-up; its time is the cold request
        cold = row["end2end"]
        sampler = Sampler(server.pid)
        sampler.start()
        rows, outputs = [], {}
        for index, entry in enumerate(entries):
            row, checked, rendered = one(entry)
            rows.append(row)
            delta = None
            if entry["reference"]:
                delta = mean_delta_e(rendered, load_image(args.corpus / entry["reference"]))
            name = Path(entry["file"]).stem
            outputs[entry["file"]] = {
                "params": checked.params.as_dict(), "text": fmt(checked.params), "delta_e": delta,
                "image": thumb(rendered, args.out / "img" / f"{name}-{quant}.jpg"),
            }  # fmt: skip
            log(f"  {index + 1}/{len(entries)} {name}: {row['end2end']:.1f}s")
        sampler.stop.set()
        sampler.join()
        cores = sampler.cores_used()
    finally:
        server.terminate()
        server.wait(timeout=30)
    deltas = [o["delta_e"] for o in outputs.values() if o["delta_e"] is not None]
    return {
        "quant": quant, "threads": args.threads, "gguf_bytes": gguf.stat().st_size,
        "load_s": load_s, "cold_request_s": cold, "idle_rss": idle, "peak_rss": sampler.rss_peak,
        "cpu_cores_used": cores, "hottest_c": sampler.temp_peak, "thermal_pause_s": sum(waited),
        "images": len(rows), "rejected_answers": int(sum(r["rejected"] for r in rows)),
        "end2end": summarize([r["end2end"] for r in rows]),
        "model_stage": summarize([r["model"] for r in rows]),
        "render_stage": summarize([r["render"] for r in rows]),
        "mean_delta_e": statistics.fmean(deltas) if deltas else None, "outputs": outputs,
    }  # fmt: skip


_STYLE = """body{font-family:system-ui,sans-serif;margin:2rem;background:#111;color:#eee}
table{border-collapse:collapse;width:100%}
th,td{border:1px solid #444;padding:.5rem;vertical-align:top;text-align:left}
img{max-width:260px;height:auto;display:block}figure{margin:0}
figcaption{font-size:.8rem;color:#aaa;margin-top:.3rem}
td.n,th.n{text-align:right;font-variant-numeric:tabular-nums}h2{margin-top:2.5rem}"""


def report(
    args: argparse.Namespace, results: list[dict[str, Any]], entries: list[dict[str, Any]]
) -> str:
    gb = 2**30
    head = ("<tr><th>Model</th><th class=n>File GB</th><th class=n>Load s</th><th class=n>Cold s</th>"
            "<th class=n>Per image s (median / p95)</th><th class=n>Model step s</th>"
            "<th class=n>Render s</th><th class=n>Peak RAM GB</th><th class=n>CPU cores used</th>"
            "<th class=n>Hottest C</th><th class=n>dE (camera)</th><th class=n>Rejected</th></tr>")  # fmt: skip
    rows = []
    for r in results:
        delta = f"{r['mean_delta_e']:.2f}" if r["mean_delta_e"] is not None else "n/a"
        rows.append(
            f"<tr><td>{r['quant']}</td><td class=n>{r['gguf_bytes'] / 1e9:.2f}</td>"
            f"<td class=n>{r['load_s']:.1f}</td><td class=n>{r['cold_request_s']:.1f}</td>"
            f"<td class=n>{r['end2end']['median']:.1f} / {r['end2end']['p95']:.1f}</td>"
            f"<td class=n>{r['model_stage']['median']:.1f}</td>"
            f"<td class=n>{r['render_stage']['median']:.2f}</td>"
            f"<td class=n>{r['peak_rss'] / gb:.2f}</td><td class=n>{r['cpu_cores_used']:.1f}</td>"
            f"<td class=n>{r['hottest_c']:.0f}</td><td class=n>{delta}</td>"
            f"<td class=n>{r['rejected_answers']}</td></tr>"
        )
    body = []
    for entry in entries:
        name = Path(entry["file"]).stem
        cells = [f"<td>{html.escape(name)}<br><small>{entry['kind']}</small></td>"]
        original = thumb(args.corpus / entry["file"], args.out / "img" / f"{name}-original.jpg")
        cells.append(figure(Path(original).relative_to(args.out), "Original"))
        if entry["reference"]:
            ref = thumb(
                args.corpus / entry["reference"], args.out / "img" / f"{name}-reference.jpg"
            )
            cells.append(figure(Path(ref).relative_to(args.out), "Your edit"))
        else:
            cells.append("<td></td>")
        for r in results:
            out = r["outputs"][entry["file"]]
            note = f"{r['quant']}: {out['text']}" + (
                f" (dE {out['delta_e']:.2f})" if out["delta_e"] is not None else ""
            )
            cells.append(figure(Path(out["image"]).relative_to(args.out), note))
        body.append(f"<tr>{''.join(cells)}</tr>")
    cols = "".join(f"<th>{r['quant']}</th>" for r in results)
    return (
        f'<!doctype html><html lang="en"><head><meta charset="utf-8"><title>retone PC benchmark</title>'
        f"<style>{_STYLE}</style></head><body><h1>retone PC benchmark</h1>"
        f"<p>{datetime.now():%Y-%m-%d %H:%M}, {os.cpu_count()} logical cores, {args.threads} threads, "
        f"CPU only, model folder {html.escape(str(args.model_dir))}. Times are warm per-image "
        f"(decode, thumbnail, model, validate, full-resolution render) after one warm-up.</p>"
        f"<table>{head}{''.join(rows)}</table><h2>Photos</h2>"
        f"<table><tr><th>Photo</th><th>Original</th><th>Your edit</th>{cols}</tr>{''.join(body)}</table>"
        f"</body></html>"
    )  # fmt: skip


def figure(path: Path, caption: str) -> str:
    return (f'<td><figure><img src="{html.escape(str(path))}" alt="{html.escape(caption)}">'
            f"<figcaption>{html.escape(caption)}</figcaption></figure></td>")  # fmt: skip


def run(args: argparse.Namespace) -> int:
    entries = json.loads((args.corpus / "manifest.json").read_text())
    if args.limit:
        entries = entries[: args.limit]
    (args.out / "img").mkdir(parents=True, exist_ok=True)
    results = []
    for position, quant in enumerate(args.quants):
        if position:
            log(f"cooling down {args.cooldown:.0f}s")
            time.sleep(args.cooldown)
        results.append(run_model(args, quant, entries))
        (args.out / "results.json").write_text(json.dumps(results, indent=2))  # survives a crash
    (args.out / "index.html").write_text(report(args, results, entries))
    print(args.out / "index.html")
    return 0


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawTextHelpFormatter
    )
    sub = parser.add_subparsers(dest="command", required=True)
    prep = sub.add_parser("prepare", help="gather camera and phone photos into one folder")
    prep.add_argument("--camera-pairs", type=Path, default=Path("artifacts/data/human/pairs.jsonl"))
    prep.add_argument("--phone", type=Path, default=Path("artifacts/eval/phone_pictures"))
    go = sub.add_parser("run", help="benchmark the three quantizations and write the report")
    go.add_argument("--model-dir", type=Path, default=Path.home() / ".retone/model-base")
    go.add_argument("--quants", nargs="+", default=list(QUANTS), choices=QUANTS)
    go.add_argument("--threads", type=int, default=os.cpu_count() or 4)
    go.add_argument("--limit", type=int, default=0, help="only the first N photos (smoke test)")
    go.add_argument(
        "--out",
        type=Path,
        default=Path("artifacts/eval/pc-benchmark") / datetime.now().strftime("%Y-%m-%d-%H%M"),
    )
    go.add_argument("--port", type=int, default=8090)
    go.add_argument("--ctx", type=int, default=4096)
    go.add_argument("--timeout", type=float, default=900.0)
    go.add_argument("--max-temp", type=float, default=90.0, help="pause above this, in C")
    go.add_argument("--cooldown", type=float, default=90.0, help="seconds between models")
    go.add_argument("--llama-server", type=Path)
    for p in (prep, go):
        p.add_argument("--corpus", type=Path, default=CORPUS)
    args = parser.parse_args(argv)
    return prepare(args) if args.command == "prepare" else run(args)


if __name__ == "__main__":
    sys.exit(main())
